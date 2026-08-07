# Where the decode bandwidth gap is

Decode moves 19.464 GB per token and the device sustains 590 GB/s, so the bytes
alone need 33 ms. A step takes 79 ms. This locates the missing 46 ms.

**Status: diagnosed, not closed.** The whole gap is one code path. Three
candidate fixes were measured and eliminated; two remain, both with real costs.

## It is not fixed overhead

Decode time against layer count, `ARCAINE_QWEN35_MAX_LAYERS`:

| layers | ms/token |
|---:|---:|
| 16 | 22.473 |
| 32 | 42.604 |
| 48 | 62.883 |
| 64 | 79.026 |

Least squares gives **4.3 ms fixed, 1.19 ms per layer**. Fixed covers the
embedding gather, final norm, the 1.27 GB LM head and the logits download —
about 5% of the step, and roughly what the head's bytes alone should cost. The
inefficiency is inside the layers.

## It is the NVFP4 MLP

The intervals are not equal, and the checkpoint explains why: layers 0-55 carry
NVFP4 MLP weights, layers 56-63 E4M3 FP8. Every group of 8 layers holds the same
mixer composition (2 full attention, 6 DeltaNet), so groups differ only in MLP
weight format.

| layers | MLP format | ms per group of 8 | ms/layer |
|---|---|---:|---:|
| 40 → 48 | NVFP4 | 10.171 | 1.271 |
| 48 → 56 | NVFP4 | 10.118 | 1.265 |
| 56 → 64 | **FP8** | **5.978** | **0.747** |

Bytes per group, from the same accounting the roofline uses (MLP is
3 x 17408 x 5120 params; NVFP4 is 0.5 B/param plus an E4M3 scale per 16, FP8 is
1 B/param plus a per-channel BF16 scale; mixer averages 119.2 MB/layer):

| group | bytes | time | achieved | of 590 GB/s |
|---|---:|---:|---:|---:|
| NVFP4 x8 | 2156 MB | 10.14 ms | 213 GB/s | **36%** |
| FP8 x8 | 3093 MB | 5.98 ms | 517 GB/s | **88%** |

**FP8 moves 44% more bytes and is 1.69x faster per layer.** NVFP4 halves the
traffic and gives back 2.4x in efficiency, so at M=1 it is a net loss. The FP8
layers show the hardware and the rest of the engine are fine: 88% of achievable
bandwidth is close to done.

Projected if every MLP were FP8: 79.0 → 49.9 ms/token, about **1.58x**.

## Three fixes that do not work

Each measured on the full model, `-n 32 -d 0 -r 3`:

| change | ms/token | verdict |
|---|---:|---|
| baseline, oneDNN f4 | 79.02 | — |
| `ARCAINE_QWEN35_NVFP4_DPAS=1` (Xe2 pack kernels) | 78.90 | no change |
| `DIFF_NVFP4_WEIGHT_LAYOUT=any` (oneDNN reorder) | 79.82 | no change |
| `DIFF_NVFP4_GPU_LAYOUT=1` | 79.03 | no change |
| `ARCAINE_QWEN35_NVFP4_DECODE_GEMV=1` | **245.72** | 3.1x worse |

The last one deserves explanation because it looks like the obvious answer.
`matmul_nvfp4_decode_gemv_esimd` and `matmul_nvfp4_decode_swiglu_esimd` already
exist in `runtime/quantization/nvfp4.hpp`, are M=1 specializations, and were
never called from qwen3_5. Wiring them in makes decode three times slower: they
launch `nd_range<1>(N, 1)`, one work-item per work-group, which suits the MoE
model's small per-expert shapes and badly under-occupies the device at this
model's intermediate size of 17408. The wiring is kept behind the flag, default
off, so the next person reads this instead of rediscovering it.

That the layout knobs and the DPAS kernels are all inert points the same way:
every available f4 path is shaped for large M, and none of them is a decode
GEMV.

## Two that could

**Requantize the NVFP4 MLP to FP8 at load.** Directly buys the 1.69x per layer
the FP8 layers already demonstrate. Two costs. It needs +6.56 GB for all 56
layers, which does not fit beside a 22 GB model on a 30 GiB card — a partial
conversion of ~40 layers fits and projects about 1.36x, and the two-GPU split
has room for all of it. And `Fp8Linear` carries one scale per output channel
where NVFP4 carries one per 16 inputs, so requantizing coarsens the scale grid;
the quality cost is unknown and must be measured with the golden gate
(`arcaine_mbench --golden`) rather than assumed.

**Write an M=1 f4 GEMV with proper occupancy.** No VRAM cost and no numerical
change, since it reads the same weights and scales. This is the better answer if
it works; it is also a real kernel, and the two existing ESIMD attempts show the
occupancy shape matters more than the arithmetic.

## Reproduce

```
# layer scaling
for L in 16 32 48 64; do
  ARCAINE_QWEN35_MAX_LAYERS=$L ARCAINE_QWEN35_MTP=0 \
    ./build/arcaine_mbench --model <dir> -p 8 -n 32 -d 0 -r 3 -w 1 --max-seq 512
done

# NVFP4 vs FP8 groups: 56 is the last NVFP4 MLP layer
for L in 40 48 56 64; do ... ; done

# the eliminated fixes
ARCAINE_QWEN35_NVFP4_DPAS=1        ./build/arcaine_mbench ...
DIFF_NVFP4_WEIGHT_LAYOUT=any       ./build/arcaine_mbench ...
ARCAINE_QWEN35_NVFP4_DECODE_GEMV=1 ./build/arcaine_mbench ...
```
