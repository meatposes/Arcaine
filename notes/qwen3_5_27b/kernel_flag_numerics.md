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
  bit-exact, and it is: max |dlogit| 0, KL 0. Without that the rest is noise.
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

## Conclusions

**No quality regression in the production kernel set.** Perplexity over 200
records is slightly lower than the conservative baseline, not higher. The
+1.52 seen on the 11-record run has no statistical power behind it and does not
survive a longer measurement — treat short runs as wiring checks only.

**The fast paths are nevertheless not numerically equivalent to the
conservative ones**: top-1 agreement 88%, mean KL 0.18 nats. That is a property
to be aware of, not a defect. It does mean any A/B that swaps these flags is not
comparing identical computations, so a throughput result across them should say
which configuration it was measured under.

**An earlier signal did not reproduce.** Before the conv-state layout fix, on
the pre-refactor branch, fused BA projection stood out at 0.062 nats mean KL
against 0.007 for the other flags, and top-1 agreement across the set ran
0.63-0.94. Both are gone: every flag now agrees at top-1 1.0000 on the short
prompt and the spread in KL is small. The layout fix accounted for most of it.

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
