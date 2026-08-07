# The decode path was not deterministic — found and fixed

Same model, same prompt, same flags, same seed, two processes: the logits
differed, sometimes by enough to change several percent of tokens over 200
steps. Roughly one run in three was bit-exact and the rest were not.

**Fixed.** The fused `[b|a]` projection was used on the multi-token prefill
path, where its output layout does not match what its consumers read. The fix
is in `operators.hpp`: prefill now issues the two separate matmuls it always
should have. The engine is bit-exact 10/10 across processes at 200 records.

This was never only a determinism bug. It fed the DeltaNet recurrence the wrong
gate values for **every prefill token after the first**, on the default
configuration, in every request the engine has ever served with a prompt longer
than 8 tokens.

## The bug

`matmul_bf16` computes `C(M,N)` row-major (`tag::ab`). The fused weight stacks
b's rows then a's, so `C(seq, 2*heads)` comes back with each token's b and a
adjacent:

```
token t occupies [t*2*heads, (t+1)*2*heads)  =  [b0..b_{h-1}, a0..a_{h-1}]
```

The consumers index two separate contiguous blocks:

```cpp
bf16* beta = workspace.tmp1.data();
bf16* g    = workspace.tmp1.data() + head_values;   // head_values = seq*heads
// ... later indexed as beta[token * heads + head], g[token * heads + head]
```

Those describe the same bytes only when `seq == 1`. For longer sequences
`beta[t*heads + j]` reads element `(t*heads + j)` of a `2*heads`-strided
matrix, which is some other token's b or a.

Three things kept it hidden:

- the decode path builds `ba` per token at M=1, where both layouts agree, so
  the fused weight is correct there and is still used there;
- the fused-decode guard routes `seq <= 8` through that same per-token loop, so
  short sequences never reach the broken path;
- the result is a plausible-looking perturbation of the gates rather than an
  obvious break, so generated text stayed coherent.

## How it was found

The instrument that mattered was `--repeat N` on the golden bench: replay the
same trajectory N times **inside one process** and compare each pass to the
first. Every earlier attempt compared across processes, which cannot separate
"memory started out different" from "the ordering is not stable".

```
in-process repeat, 64 layers, 200 records     2/4 bit-exact
```

Not a startup-memory fault — the same allocations, after `reset_cache()`,
disagree with themselves. And the first differing record was **0**, the prefill
output, not the deep decode steps (59, 147, 169, 181) an earlier version of
this note reported. Those were downstream compounding of a prefill error.

That turned a 1-in-3 failure over a 200-record run into a ~95% failure over a
single 32-token prefill, which made everything after it cheap:

| probe | clean / 19 |
|---|---:|
| prefill 32, 1 record | 1 |
| prefill 1, 1 record | 19 |
| prefill 1, 8 records (decode path) | 19 |
| prefill 32, `MAX_LAYERS=1` | 19 |

Multi-token prefill only. Then a layer sweep, with layers 0-1 full attention
and 2+ DeltaNet:

| `MAX_LAYERS` | 1 | 2 | 3 | 4 | 6 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean / 19 | 19 | 19 | 17 | 17 | 12 | 10 | 2 | 1 |
| DeltaNet layers | 0 | 0 | 1 | 2 | 4 | 6 | 12 | 48 |

Onset is exactly at the first DeltaNet layer, and the rate is consistent with
independent per-layer failure at ~10.5%: that model predicts 12.2/19 at L=6
(observed 12) and 9.8/19 at L=8 (observed 10).

Then the flag sweep at `MAX_LAYERS=16`:

| arm | clean / 19 |
|---|---:|
| baseline | 10 |
| `ESIMD_DELTA=0` | 5 |
| `GPU_INIT_FILL=0` (zero every allocation) | 5 |
| **`FUSED_BA_PROJECTION=0`** | **19** |
| `ESIMD_DELTA=0` + `FUSED_BA_PROJECTION=0` | 19 |

`ARCAINE_GPU_INIT_FILL` was added for this sweep (`runtime/gpu/buffer.hpp`) to
fill every fresh device allocation. It killed the leading hypothesis — that
`sycl::malloc_device` handing back dirty pages explained the intermittency —
and it is worth keeping for the next time something looks like an
uninitialized read. `ARCAINE_QWEN35_ZERO_KV` (`cache.hpp`) was added the same
way, to zero the KV cache, which unlike the DeltaNet state is not zeroed on
allocation or reset; it made no difference here.

## Validation

`FUSED_BA_PROJECTION=0`, before the code fix:

```
in-process, 64 layers, 1 record       19/19 bit-exact
in-process, 64 layers, 200 records      4/4 bit-exact
cross-process, 200 records            10/10 bit-exact
```

The code fix, measured the same way:

```
in-process, 64 layers, 200 records      4/4 bit-exact
cross-process, 200 records            10/10 bit-exact
against the FUSED_BA_PROJECTION=0 golden:
    max |dlogit| 0, top-1 200/200      identical
```

That last line matters: the committed change reproduces the validated arm
exactly rather than merely resembling it. The cross-process gate is the one
that used to fail about two runs in three.

## How wrong it was

The pre-fix binary on its default settings, against the fixed engine's golden,
200 records:

```
max |dlogit|      19.4688   (step 169)
top-1 agreement    0.8700   (174/200)
perplexity       186.1591 -> 191.9893   (+5.8302, 3.1% worse)
```

**26 of 200 generated tokens changed.** This is the size of the quality bug the
default configuration was carrying on every prompt longer than 8 tokens, and it
is far outside the noise band the retracted measurements were lost in.

## Method, for the next stochastic fault

The earlier version of this note recorded two wrong localizations, both from a
single run per configuration. The rule that fixed it:

- **Ask whether one process disagrees with itself before comparing processes.**
  It splits the hypothesis space in half for the price of one run and it is
  what unblocked this.
- **Shrink the repro before bisecting.** 200 records at 1-in-3 is unusable;
  one 32-token prefill at 19-in-20 is a fast, sharp instrument.
- **Report a rate, never a sample.** Every table above is k/19 or k/10.
- Bisect structure (layers) before flags — the layer sweep named DeltaNet
  before any flag was touched, which made the flag list short.

## What this restores

Every cross-process perplexity comparison was blocked by this. Now unblocked:

- **the FP8 requantization quality figure.** `decode_bandwidth_gap.md`'s "+6%
  perplexity" was retracted; it can now be measured. This gates enabling
  `ARCAINE_QWEN35_MLP_FP8_LAYERS=56` by default, which is 1.57x decode and
  2.34x prefill.
- **the kernel-flag numerics gate** in `kernel_flag_numerics.md`.

Both must be re-measured rather than un-retracted: the old numbers were taken
against a prefill that was computing the wrong thing.

## Reproduce

```
M=/path/to/Qwen3.6-27B-NVFP4

# the fast probe: one prefill, twenty passes, one process
ARCAINE_QWEN35_MTP=0 ./build/arcaine_mbench --model $M \
    --golden capture --out /tmp/probe.bin --repeat 20 --steps 1 --prefill 32

# the original gate
P=$(head -70 notes/qwen3_5_27b/architecture.md | tr '\n' ' ')
./build/arcaine_mbench --model $M --golden capture --out /tmp/g.bin \
    --steps 200 --prefill 32 --prompt "$P"
for i in $(seq 10); do
  ./build/arcaine_mbench --model $M --golden compare --golden-file /tmp/g.bin \
    | grep "max |dlogit|"
done
```
