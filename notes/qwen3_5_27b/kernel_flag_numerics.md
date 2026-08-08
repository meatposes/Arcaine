# Qwen3.5 kernel flag numerics

Every fast path in this model is selected by an environment flag. Timing them
against each other only means something if they compute the same thing, and the
existing kernel benchmarks check shapes and pointers against random weights,
which catches wiring but not arithmetic.

`arcaine_mbench --golden capture|compare` records a teacher-forced logit
trajectory under one configuration and replays it under another.
`scripts/qwen35_kernel_flag_gate.sh` drives the whole sweep.

## Method

Capture with every fast path off, then compare with one enabled at a time and
finally with the production default. Teacher forcing means the comparison does
not depend on the sampler and perplexity over the same tokens is directly
comparable between runs.

Read the metrics in this order:

- **control** — the baseline compared against its own golden. It must be
  bit-exact. At 11 records it is; at 200 it is not, because the fused DeltaNet
  decode kernel is nondeterministic. Run the control at the record count you
  intend to use. See .
- **perplexity** — the only metric that tracks output quality. Use it to decide
  whether a change is acceptable.
- **KL(golden || current)** — distributional distance, weighted by the golden's
  own probabilities. Sensitive to the prompt's entropy, so compare KL only
  across runs on the same prompt.
- **max |dlogit|** — nearly useless on its own. One irrelevant token deep in a
  248320-wide vocabulary dominates it, and it sits around 15 even for
  configurations whose top-1 agreement is perfect.

`FUSED_ESIMD_DELTA_DECODE` requires `ESIMD_DELTA`; the fused decode path and the
scalar baseline use different recurrent-state layouts and mixing them produces
garbage rather than an error, so the sweep enables the two together.

## Measured 2026-08-06, Qwen3.6-27B-NVFP4, one BMG G31

After the conv-state layout fix. Short prompt, 11 records, low-entropy text
(golden perplexity 5.47):

| configuration | mean KL | top-1 | ppl delta |
|---|---:|---:|---:|
| control | 0 | 1.0000 | +0.0000 |
| XMX attention | 0.0069 | 1.0000 | -0.184 |
| ESIMD DeltaNet | 0.0310 | 1.0000 | +0.855 |
| fused decode (+ESIMD) | 0.0331 | 1.0000 | +0.710 |
| fused BA projection | 0.0281 | 1.0000 | +1.455 |
| production default | 0.0298 | 1.0000 | +1.524 |

Long run, 200 records of technical prose (golden perplexity 202.59), production
against conservative:

| | |
|---|---:|
| top-1 agreement | 0.8800 (176/200) |
| KL max / mean | 3.72 / 0.181 nats |
| perplexity | 202.59 -> **197.80** (-4.79, -2.4%) |

## The sweep once found the bug and this note talked itself out of it

Before rewriting the conclusions, the important part. An earlier version of
this section read:

> **An earlier signal did not reproduce.** Before the conv-state layout fix, on
> the pre-refactor branch, fused BA projection stood out at 0.062 nats mean KL
> against 0.007 for the other flags [...] Both are gone [...] The layout fix
> accounted for most of it.

Fused BA projection was the one flag that stood out, by roughly 9x, and the
note attributed it to something else and moved on. It was in fact a real
defect — the fused `[b|a]` matmul's output layout does not match what the
prefill consumers read, corrupting the DeltaNet gates for every token after
the first (see `nondeterminism.md`). It took another month and a different
instrument to find.

The failure was not the measurement. It was accepting "the other fix probably
explains it" for an outlier the method had specifically surfaced. **An
unexplained outlier is an open question, not a resolved one.**

## Conclusions, re-measured 2026-08-07 on a deterministic engine

The retraction above is discharged: the nondeterminism it depended on is fixed,
the control now reads `max |dlogit| 0` with top-1 400/400, and
`ARCAINE_QWEN35_PERSISTENT_IO=0` is bit-exact over 400 records — a real
equivalence proof for that path.

**No defect was proven in any kernel flag.** Six flags compared against the
production default over 400 teacher-forced records of prose:

| flag | max \|dlogit\| | top-1 |
|---|---:|---:|
| control | 0 | 1.0000 |
| `PERSISTENT_IO=0` | 0 | 1.0000 |
| `FUSED_DECODE_MAX_SEQ=64` | 17.03 | 0.9250 |
| `XMX_ATTENTION=0` | 17.94 | 0.9225 |
| `SUBGROUP_ATTENTION=1` | 17.14 | 0.9350 |
| `FUSED_ESIMD_DELTA_DECODE=0` | 17.34 | 0.9200 |
| `NVFP4_DPAS=1` | 19.72 | 0.7825 |
| `ESIMD_DELTA=0` | 36.88 | 0.3375 |

**Read this as "they differ", not "they are broken".** Two kernels summing the
same products in different orders are expected to disagree, and DeltaNet's
recurrence amplifies any disagreement. Divergence between two implementations
names neither as wrong. Two discriminators that look useful and are not:

- *Magnitude at record 0.* The genuine fused-BA bug showed only
  `max |dlogit| 0.25` at record 0 — smaller than every arm above. Small early
  error neither exonerates nor convicts.
- *Growth with prefill length.* `XMX_ATTENTION=0` goes 0.95 at prefill 32 to
  5.38 at prefill 512 with top-1 unchanged. That is the signature of
  accumulation order, i.e. the null hypothesis.

`ESIMD_DELTA=0` is the visible outlier — top-1 0.25 by 8 records while every
other arm holds 0.875 or better — and, per the lesson at the top of this
section, an outlier gets an answer rather than an explanation.

**The answer: both DeltaNet kernels are correct.** `deltanet_bench` gained an
fp64 host oracle (`ARCAINE_DELTANET_ORACLE=1`) that computes the gated delta
rule in double precision from the same bf16 inputs, so the two GPU paths can be
ranked against truth instead of contrasted with each other:

```
fp64 reference, 32 tokens, state carried across tokens
  baseline  max_abs 0.000004  rms 0.000000   0.67 bf16 ULP of peak
  esimd     max_abs 0.000004  rms 0.000000   0.67 bf16 ULP of peak
```

Identical, both at the bf16 output-quantization floor. Neither departs from the
mathematical recurrence, so the end-to-end divergence is amplification of
sub-ULP state differences, not a defect. Note what this cannot do: it has no
resolution below one bf16 output ULP, and it covers one shape (heads=48,
K=V=128) with synthetic normalized inputs at peak |truth| 0.0014.

**Attention is verified too.** `attention_bench` gained the same treatment
(`ARCAINE_ATTENTION_ORACLE=1`): fp64 causal GQA from the same bf16 inputs, with
the query→kv-head mapping and the causal mask written out independently of the
kernels rather than shared with them, since getting either wrong is the failure
being hunted.

79 comparisons over shapes chosen to be awkward — `seq` ∈ {1,2,3,7,8,9,15,16,
17,33}, which straddles both the XMX kernel's 8-query tile and its 16-wide
subgroup, crossed with `past` ∈ {0,1,7,13,15,31,64,100} so the cache offset is
rarely aligned either. Worst error per kernel, in units of one bf16 output ULP:

| kernel | worst |
|---|---:|
| baseline | 0.90 |
| subgroup | 0.90 |
| xmx (default) | 1.04 |
| xmx-gqa | 0.84 |

Nothing above 1.04, and `seq=1 past=0` is exact for all four. A wrong mask or a
wrong kv-head mapping would land orders of magnitude above 1, not at 1, so the
ragged final tile and the unaligned `past` are both handled. XMX being
consistently a little worse than the scalar paths (0.73-1.04 against 0.50-0.90)
is what tiled accumulation costs, not a defect.

That leaves `NVFP4_DPAS` as the only swept flag with no oracle, and it is off by
default — a defect there reaches nobody today. Its default counterpart is
oneDNN's f4 matmul, where an oracle would be testing Intel's library rather than
this engine, so it is deliberately not built. Worth doing only if someone plans
to turn DPAS on.

## The guard gap held a second layout bug

Chasing the fused-decode guard turned up a real defect, and the sweep had
already been pointing at it. `ESIMD_DELTA=0` was the one arm far outside the
pack — top-1 0.3375 against 0.92 or better for everything else — while the fp64
oracle said both DeltaNet kernels are individually correct. Both statements are
true, because the fault was never in either kernel.

The three DeltaNet implementations do not agree on the recurrent state's
layout:

```
qwen35_recurrent_delta         state[head*K*V + k*V + v]   [head][key][value]
qwen35_recurrent_delta_esimd   state[head*V*K + v*K + k]   [head][value][key]
qwen35_delta_decode_fused      recurrent_state[... v*K]    [head][value][key]
```

Both arms of the guard advance the *same* `cache.recurrent_state`. The
per-token arm was gated on `qwen35_fused_esimd_delta_decode_enabled()` alone,
so `ARCAINE_QWEN35_ESIMD_DELTA=0` selected the scalar kernel for prefill while
leaving the fused ESIMD core on for decode. Prefill then wrote
`[head][key][value]` and decode read `[head][value][key]` out of the same
buffer. K and V are both 128, so the mismatch is exactly size-compatible:
nothing faults, the state is silently transposed.

This is the fused-BA bug's twin — dimensions coincide, so it is invisible
except numerically — and `kernels.hpp` already carried the warning ("a cache
must not switch layouts mid-stream") with nothing enforcing it.

**Fix:** `qwen35_esimd_delta_enabled()` joins the guard condition, so the flag
switches the whole DeltaNet path coherently instead of splitting it across two
layouts. The default never mixed (ESIMD on both sides), so the default is
bit-exact unchanged — verified, `max |dlogit| 0`, top-1 400/400 against the
pre-fix golden. The repaired arm moves as predicted:

| | before | after |
|---|---:|---:|
| `ESIMD_DELTA=0` top-1 | 0.3375 | **0.9275** |

0.9275 sits inside the 0.920-0.935 band every other kernel swap occupies, which
is the point: it is now an ordinary alternative implementation rather than a
corrupted mixture. The deeper fix is to give the scalar kernel the ESIMD
layout so no combination *can* mix them; this makes the unsafe one unreachable
meanwhile.

**What is still not proven.** With the default flags both arms use
`[head][value][key]`, and routing an identical prefill through each
(`FUSED_DECODE_MAX_SEQ=64`) leaves `max |dlogit| 0.859` at record 0 and top-1
0.9250 over 400. That is now statistically indistinguishable from every other
legitimate kernel swap — the arm sits in the same band as XMX (0.9225) and
fused-delta-decode (0.9200) — which is evidence, not proof. Proving the two
arms compute the same function needs an fp64 oracle spanning the whole DeltaNet
block (conv, gating, recurrence, norm), not just the recurrence. Recorded as
open. The distinction worth keeping: an outlier got a cause, and what remains
is a population, not an outlier.

## Where the M=1 pattern still lives

The confirmed bug had a shape worth searching for directly: a buffer written
with one layout and read with another, coinciding only at one sequence length.
Every site was checked.

| site | verdict |
|---|---|
| `swiglu_strided` on the fused `gate_up` | correct — indexes `[tok*2*inter + dim]` and `[tok*2*inter + inter + dim]` |
| `qwen35_split_q_gate`, `..._q_gate_kv`, `qwen35_extract_qkv` | correct — all use a per-token row stride |
| `operators.hpp` BA split | the bug, fixed |
| fused-decode guard, `operators.hpp` | two arms, equivalence unverified (above) |
| `seq == 1` NVFP4 GEMV, `operators.hpp` | off by default, separately measured as a 3.1x regression |
| `qwen35_xmx_attention_decode_gqa` | defined but never called from `operators.hpp`; bench-only |

The BA site was the only place a fused matmul's output was split with raw
pointer arithmetic. Everywhere else the split goes through a kernel that takes
`seq` and indexes per token, which is what makes those correct. That is the
distinction to preserve: **if a fused output must be split, split it in a kernel
that knows the row stride, not by offsetting a pointer.**

**Practical consequence, unchanged:** the fast paths are not numerically
identical to the conservative ones, so any A/B that swaps these flags is not
comparing identical computations. A throughput number should say which
configuration produced it.

## Reproduce

```
ZE_AFFINITY_MASK=2 scripts/qwen35_kernel_flag_gate.sh <model_dir> \
    --steps 24 --prefill 24 --prompt "<text>"
```

For a quality verdict rather than a wiring check, use a few hundred records:

```
ARCAINE_QWEN35_XMX_ATTENTION=0 ARCAINE_QWEN35_ESIMD_DELTA=0 \
ARCAINE_QWEN35_FUSED_ESIMD_DELTA_DECODE=0 ARCAINE_QWEN35_FUSED_BA_PROJECTION=0 \
  ./build/arcaine_mbench --model <dir> --golden capture --out golden.bin \
      --steps 200 --prefill 32 --prompt "<long text>"
./build/arcaine_mbench --model <dir> --golden compare --golden-file golden.bin
```

A golden file is `records * vocab * 4` bytes — 190 MiB at 200 records here, so
they are not worth keeping.
