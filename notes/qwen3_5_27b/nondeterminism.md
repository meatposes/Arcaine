# The decode path is not deterministic

Same model, same prompt, same flags, same seed, two processes: the logits
differ, sometimes by enough to change several percent of tokens over 200 steps.

It is **intermittent**. Some runs are bit-exact and some are not, with the same
binary and the same configuration.

This invalidates every perplexity comparison in these notes that was measured
across processes, and it is a correctness problem in its own right — the engine
does not reliably produce reproducible output.

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

Divergence first appears deep into the sequence — observed at steps 59, 147,
169 and 181 across runs. That is why it went unnoticed: the numerical gate's
original control used 11 records and passed bit-exact. Errors accumulate
through the DeltaNet recurrent state, which carries across steps even under
teacher forcing, so one early bit difference compounds.

## It predates the changes on this branch

The obvious suspicion is that the performance work on this branch introduced it.
It did not. Checking out `0ac12a9`, the base commit this branch forked from, and
adding *only* the golden benchmark — a measurement tool that does not touch the
inference path — gives five runs out of five nondeterministic:

```
max |dlogit| 0.15625  (step 156)
max |dlogit| 9.4375   (step 142)
max |dlogit| 0.15625  (step 156)
max |dlogit| 5.4502   (step 67)
max |dlogit| 3.98438  (step 173)
```

Nothing on this branch caused it. What this branch did was build the instrument
that exposes it: a 200-record teacher-forced golden comparison. The engine had
no way to notice before, because nothing compared logit trajectories across
processes at that length.

To reproduce the base measurement:

```
git worktree add --detach <path> 0ac12a9
cp src/modeling/qwen3_5/benchmarks/golden_bench.cpp <path>/src/modeling/qwen3_5/benchmarks/
# add golden_bench.cpp to MODEL_BENCH_SOURCES in that worktree's
# src/modeling/qwen3_5/CMakeLists.txt, and the --golden dispatch to its
# benchmarks/model_bench.cpp; both are benchmark files, neither is on the
# inference path
```

## Not localized

An earlier version of this note claimed the fault was in
`qwen35_delta_decode_fused_esimd`, on the strength of one run per configuration:
the arm with `FUSED_ESIMD_DELTA_DECODE=0` reported `max |dlogit| = 0` and the
others did not. **That was wrong.** Repeating the same arm three times:

```
compare 1   max |dlogit| 4.76562  (step 59)   perplexity +26.01
compare 2   max |dlogit| 5.18750  (step 59)   perplexity  +8.07
compare 3   max |dlogit| 0                    perplexity  +0.00
```

The clean result was luck. One run per configuration cannot localize a
stochastic fault, and the per-flag table that used to be here has been removed
rather than corrected, because every row in it had the same defect.

The fused kernel is also clean in isolation. `arcaine_kbench
qwen35-delta-decode-fusion` compares it against the unfused baseline over 32
tokens with synthetic inputs and reports `core_max_abs=0.000000`,
`z_max_abs=0.000000` on every run. Whatever the cause is, it does not reproduce
at that scale.

There is currently **no known workaround** and no identified culprit.

## What this retracts

Any perplexity delta measured across processes at this sequence length sits
inside a noise band of roughly 198-245. That covers:

- **the FP8 requantization quality figure** in `decode_bandwidth_gap.md`. The
  "+6% perplexity" and the clip sweep that produced it are not supported.
- **the "no quality regression" conclusion** in `kernel_flag_numerics.md`. Its
  202.59 -> 197.80 is well inside the noise.

Throughput is unaffected. Timing reproduces to well under a percent across runs
and the speedups at issue are 1.4x to 2.3x.

## How to investigate this properly

The mistake to avoid is the one made twice above: concluding from a single run.
The fault fires on some fraction of runs, so any comparison between
configurations needs **repeats and a failure rate**, not one sample.

A workable shape:

- fix a golden, then run `compare` N times per configuration and report how many
  of the N were bit-exact
- N large enough to separate rates that differ by a few tenths; the observed
  rate is roughly one clean run in three, so N=10 is a floor
- vary one thing at a time across configurations, and keep the record count at
  200, since shorter runs hide it entirely

Worth checking early, since none of it has been done: whether the fault survives
`ARCAINE_QWEN35_MAX_LAYERS=1` (isolating a single layer), whether it appears at
all with a single decode step rather than 200, and whether it depends on the
DeltaNet path at all once repeats are used.

## Reproduce

```
P=$(head -70 notes/qwen3_5_27b/architecture.md | tr '\n' ' ')
./build/arcaine_mbench --model <dir> --golden capture --out /tmp/g.bin \
    --steps 200 --prefill 32 --prompt "$P"

for i in $(seq 10); do
  ./build/arcaine_mbench --model <dir> --golden compare --golden-file /tmp/g.bin \
    | grep "max |dlogit|"
done                 # some runs report 0, some do not
```
