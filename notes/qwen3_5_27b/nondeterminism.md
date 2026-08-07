# The decode path is not deterministic

Same model, same prompt, same flags, same seed, two processes: the logits
differ, and by enough to change roughly 4% of tokens over 200 steps.

This invalidates every perplexity comparison in these notes that was measured
across processes. It is also a correctness problem in its own right — the engine
does not currently produce reproducible output.

## Evidence

Three golden captures, identical configuration, `FP8_LAYERS=0`, `MTP=0`,
200 records:

```
197.8519
198.1423
202.7083      the golden files also differ byte for byte
```

Comparing a configuration against its own golden — a control that must report
`max |dlogit| = 0`:

```
MTP=0, FP8 off    max |dlogit| 3.8125   top-1 0.9600   perplexity +17.37
MTP=1, FP8 off    max |dlogit| 3.6250   top-1 0.9600   perplexity +17.00
```

## Which path

Each arm captured and compared against its own golden:

| configuration | max abs delta logit | |
|---|---:|---|
| default (ESIMD delta, fused decode) | 5.8125 | nondeterministic |
| `FUSED_ESIMD_DELTA_DECODE=0` | **0** | **deterministic** |
| `ESIMD_DELTA=0` + fused off (scalar recurrence) | 13.7188 | nondeterministic |
| `XMX_ATTENTION=0` (fused decode still on) | 19.9688 | nondeterministic |

`qwen35_delta_decode_fused_esimd` is not deterministic, and neither is the
scalar `qwen35_recurrent_delta` fallback. `qwen35_recurrent_delta_esimd` — fused
decode off, ESIMD on — is the one clean path.

Attention is not implicated: turning XMX off leaves the fused delta decode on
and the run stays nondeterministic.

The divergence first appears deep into the sequence, around step 147-181. That
is why it went unnoticed: the numerical gate's original control used 11 records
and passed bit-exact. Errors accumulate through the DeltaNet recurrent state,
which carries forward across steps even under teacher forcing, so a single
early bit difference compounds.

## Workaround

`ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE=0` is deterministic and costs 1.1%:

| | tok/s | ms/token |
|---|---:|---:|
| fused decode on (default) | 12.64 | 79.10 |
| fused decode off | 12.51 | 79.96 |

One percent for reproducible output is a trade worth making, and there is a case
for flipping the default until the race is found.

## What this retracts

Any perplexity delta measured across processes at this sequence length sits
inside a noise band of roughly 198-215, about 9%. That covers:

- **the FP8 requantization quality figure** in `decode_bandwidth_gap.md`. The
  "+6% perplexity" and the clip sweep that produced it are not supported. A
  later run of the same comparison gave -3.3 instead of +12.2.
- **the "no quality regression" conclusion** in `kernel_flag_numerics.md`. Its
  202.59 -> 197.80 is well inside the noise.

Throughput numbers are unaffected. Timing is reproducible to well under a
percent across runs, and the speedups at issue are 1.4x to 2.3x.

## Measuring quality until this is fixed

Use `ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE=0` for both arms of any perplexity
comparison. That arm is bit-exact, so a nonzero delta is the change under test
rather than the engine.

Always run the control first — compare a configuration against its own golden
and confirm `max |dlogit| = 0` — and run it at the full record count, not a
short one. Eleven records hid this for the entire session.

## Reproduce

```
P=$(head -70 notes/qwen3_5_27b/architecture.md | tr '\n' ' ')
for i in 1 2 3; do
  ./build/arcaine_mbench --model <dir> --golden capture --out /tmp/g$i.bin \
      --steps 200 --prefill 32 --prompt "$P"
done            # perplexity differs run to run

./build/arcaine_mbench --model <dir> --golden compare --golden-file /tmp/g1.bin
                # max |dlogit| nonzero against its own golden

ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE=0 ...   # both steps, now bit-exact
```
