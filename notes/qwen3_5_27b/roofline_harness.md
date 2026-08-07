# Roofline harness

Decode is bandwidth-bound. A throughput number on its own cannot tell "good"
from "as good as this device gets", so every decode measurement wants a
denominator: the bytes that had to move, and the bandwidth actually available.

Three pieces:

| what | where | needs a model |
|---|---|---|
| `arcaine_kbench device-bandwidth` | `src/benchmarks/device_bandwidth_bench.cpp` | no |
| `arcaine_kbench qwen35-attention-shape` | `modeling/qwen3_5/benchmarks/attention_shape_bench.cpp` | no |
| `arcaine_mbench --roofline` | `modeling/qwen3_5/benchmarks/model_bench.cpp` | yes |

The shared probe is `src/benchmarks/device_bandwidth.hpp`, used by both
`--roofline` and `nvfp4-roofline`.

## Measuring bandwidth is easy to get wrong

Two mistakes, both of which produced a plausible number:

**A compressible buffer measures the compressor, not the bus.** This GPU
compresses memory losslessly, so a `memset` or uniform fill reports ~2420 GB/s
where the same probe over random data reports ~595 GB/s on the same card, at
every buffer size. Weights are incompressible, so the random figure is the one a
roofline should divide by. `device-bandwidth` prints both side by side so the
gap stays visible:

```
 4096 MiB  uniform: read  2422.1 GB/s (event  2449.3) | copy  2766.8 | checksum ok
 4096 MiB  random : read   594.8 GB/s (event   597.1) | copy   530.9
```

**A conditional store lets the compiler delete the loads.** A guard of the form
`if ((id & mask) == 0 && sum == magic)` short-circuits on the id term, so the
accumulation is dead code for every lane that fails the mask. Every lane must
store its own partial. The bench verifies the reduction checksum for exactly
this reason.

Buffers under ~1 GiB also understate — 64 MiB reads about 20% low against the
figure 1-4 GiB converge on — so the probe defaults to 1 GiB.

A datasheet constant is not a substitute. `nvfp4_roofline_bench` used to default
to 456 GB/s, which understates this device by 29%: a kernel at 62% of peak
reported 80% and looked finished. It now measures by default and prints
`(measured)` or `(given)` so the source of the denominator is never ambiguous.

## Bytes per token

The architecture reports its own per-decode-step traffic through
`ModelInfo::decode_traffic`, computed by walking the resident tensors rather
than re-deriving from the config, so layer truncation and fused projections are
reflected without a second source of truth.

Only bytes a decode step really moves are counted. The embedding table is the
case that matters: 2.5 GiB resident, but decode gathers one row of it, so
counting the table would inflate the denominator about tenfold and make every
efficiency figure meaningless.

Models that have not implemented the accounting pass nothing and the columns
stay off, so this does not disturb the other three.

## Measured, Qwen3.6-27B-NVFP4 on one BMG G31

```
traffic per decode step (analytic, from resident tensors)
  lm_head               1.272 GB     6.5%
  mlp                  10.562 GB    54.3%
  delta_proj            5.641 GB    29.0%
  attn_proj             1.678 GB     8.6%
  delta_state_rw        0.308 GB     1.6%
  fixed total          19.463 GB
  kv                     64.0 KiB per cached position

  GPU 0   read   590.0 GB/s   copy   534.3 GB/s

 test          kv-depth      t/s     ms/tok    GB/tok     GB/s   %roof
 pp 512               —   650.14      1.538         —        —       —
 tg 32                0    12.80     78.122    19.464    249.1   42.2%
 tg 32             2048     9.52    105.090    19.598    186.5   31.6%
```

Prefill leaves the columns blank on purpose: its weights are amortized across
the whole batch, so a per-token byte count would mislead rather than merely be
absent.

**Decode runs at 42.2% of achievable bandwidth at depth 0, falling to 31.6% at
depth 2048.** That is the headroom figure. It says roughly 2.4x remains in the
plain decode path before the hardware is the limit — worth weighing against
further speculation work, which attacks a different term.

The MLP is 54.3% of per-token traffic and the DeltaNet projections another
29.0%, so those two are where bytes actually go. The LM head, at 6.5%, is
smaller than it looks.

## Attention shape sweep

`qwen35-attention-shape --sweep` drives `qwen35_xmx_attention` directly with
random data over the shapes the engine produces: decode, prefill, chunked
prefill, and a small batch against a large history — the last of which nothing
generated until the MTP head began advancing its own cache a window at a time.

It exists because a device hang was once blamed on this kernel. Driving it
directly showed every accused shape passing, which moved the search to the
caller and found the real cause: a host buffer freed while an async `memcpy`
over it was still pending. A kernel is easier to exonerate than to convict, and
doing it behind a 22 GB model load is too slow to iterate on.

## Reproduce

```
./build/arcaine_kbench device-bandwidth
./build/arcaine_kbench qwen35-attention-shape --sweep
./build/arcaine_mbench --model <dir> --roofline -p 512 -n 32 -d 0,2048 -r 3 -w 1
./build/arcaine_kbench nvfp4-roofline -p 2048 --experts 8    # measured peak now
```
