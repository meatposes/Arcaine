# Numerical verification tools for Qwen3.5

Three instruments for answering "is this kernel computing the right thing?"
None of them is on the inference path.

They exist because the existing kernel benchmarks compare one GPU kernel
against another. That establishes only that two implementations disagree — it
never says which is wrong, and two kernels summing the same products in a
different order are *expected* to disagree. Every tool here was built to close
that gap, and two of them found real defects doing it.

## 1. The golden gate — `arcaine_mbench --golden`

Records a teacher-forced logit trajectory under one configuration and replays
the identical token sequence under another.

```
M=/path/to/model
P=$(sed -n '90,700p' some_prose.txt | tr '\n' ' ' | cut -c1-24000)

arcaine_mbench --model $M --golden capture --out golden.bin \
    --steps 400 --prefill 32 --max-seq 2048 --prompt "$P"

arcaine_mbench --model $M --golden compare --golden-file golden.bin --max-seq 2048
```

Read the output in this order:

- **control first.** Compare a configuration against *its own* golden. It must
  report `max |dlogit| 0`. If it does not, the engine is nondeterministic and
  nothing else in the run means anything. Run the control at the record count
  you actually intend to use — short runs hide it.
- **top-1 agreement** — the stable statistic. It moved by less than a point
  between 200 and 1000 records in every configuration measured.
- **perplexity** — heavy-tailed and dominated by a handful of high-loss steps.
  Not reliable for ranking small differences even at 1000 records.
- **`max |dlogit|`** — nearly useless alone. One irrelevant token deep in the
  vocabulary can dominate it.

## 2. The in-process repeat probe — `--repeat N`

Replays the same trajectory N times **inside one process** and compares each
pass against the first.

```
arcaine_mbench --model $M --golden capture --out /tmp/probe.bin \
    --repeat 20 --steps 1 --prefill 32
```

This is the tool that matters most, and it is the cheapest. Comparing across
processes cannot distinguish "memory started out different" from "the ordering
is not stable"; comparing inside one process can. Ask it before anything else
when chasing nondeterminism — it splits the hypothesis space in half for the
price of one run.

It is also how a slow repro becomes a fast one. A fault visible on one run in
three over 200 records became visible on nineteen runs in twenty over a single
32-token prefill, which turned an intractable bisect into a quick one.

Always report a **rate** — `k/N`, N ≥ 10. A stochastic fault cannot be
localized from one sample; the mistake has been made here more than once.

## 3. fp64 host oracles

Both live in the existing kernel benchmarks and are off unless asked for.

```
ARCAINE_DELTANET_ORACLE=1  arcaine_kbench qwen35-deltanet -p 32 -n 32 -w 0 -r 1 \
    --kernels baseline,esimd
ARCAINE_ATTENTION_ORACLE=1 arcaine_kbench qwen35-attention \
    --kernels baseline,subgroup,xmx,xmx-gqa
```

Each computes the operation in double precision on the host from the *same*
bf16 inputs the kernels receive, so every kernel can be ranked against the
value it is approximating rather than against a sibling. Error is quoted in
units of one bf16 output ULP: a kernel that merely stores its answer in bf16
lands near 1, while a structural error — wrong mask, wrong head mapping,
transposed state — lands orders of magnitude above it.

The attention oracle writes out the query→kv-head mapping and the causal mask
independently of the kernels rather than sharing them, because getting either
wrong is the failure it exists to catch. It sweeps deliberately awkward shapes:
`seq` ∈ {1,2,3,7,8,9,15,16,17,33}, straddling both the XMX kernel's 8-query
tile and its 16-wide subgroup, crossed with `past` ∈ {0,1,7,13,15,31,64,100} so
the cache offset is usually unaligned.

Current results, all four attention kernels and both DeltaNet kernels:

```
attention   baseline 0.90   subgroup 0.90   xmx 1.04   xmx-gqa 0.84   (worst ULPs)
deltanet    baseline 0.67   esimd    0.67                             (worst ULPs)
```

Everything verified. What an oracle cannot do: resolve below one bf16 output
ULP, or cover shapes it was not given.

## 4. Two allocation diagnostics

Both off by default, neither implicated in anything so far. Kept because they
cheaply kill a hypothesis that otherwise costs a day.

- `ARCAINE_GPU_INIT_FILL=<0-255>` fills every fresh `GpuBuffer` with a byte
  value. `sycl::malloc_device` does not initialize, so a buffer read before it
  is written takes whatever the driver last left there — which differs between
  processes and sometimes happens to match. Zero every allocation and the
  engine becomes reproducible if that is the cause; poison with `205` (`0xCD`)
  and it becomes reliably wrong. If neither changes the failure rate, it is not
  an uninitialized read.
- `ARCAINE_QWEN35_ZERO_KV=1` zeroes the KV cache, which unlike the DeltaNet
  state is zeroed on neither allocation nor reset. `reset()` only rewinds
  `filled`. Attention should never read past that mark, so this should make no
  difference — which is why being able to test it is worth the six lines.

## Building

`arcaine_mbench` and `arcaine_kbench` link every enabled model. If another
model fails to compile in your environment, scope the build:

```
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -G Ninja -DARCAINE_MODELS="qwen3_5"
```

## What these found

- A fused `[b|a]` projection whose output layout did not match what its
  consumers read, corrupting DeltaNet gates for every prefill token after the
  first on the default configuration. Also the cause of intermittent
  cross-process nondeterminism.
- A recurrent state written as `[head][key][value]` by one kernel and read as
  `[head][value][key]` by another, silently transposed because K and V are both
  128 so the sizes coincide.

Both are the same shape: **a buffer written with one layout and read with
another, where the sizes happen to agree.** That is the thing to go looking for
next, and the reason to reach for an oracle rather than a diff.
