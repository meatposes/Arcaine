# Qwen3.5 speculative decoding

The checkpoint ships an MTP head: one full-attention decoder layer that predicts
the token two positions ahead from the backbone hidden state and the embedding
of the token after it. It costs one layer against sixty four, so a round that
lands returns two tokens for one backbone pass.

`run_decode_loop` takes the speculative path whenever the head is present.
`ARCAINE_QWEN35_MTP=0` keeps the weights resident and the head idle, which
selects the ordinary loop and is the A/B.

## A round

State on entry: the caches cover positions `0..past-1`, and `pending` is the
token at `past`, already sampled and not yet consumed by a forward.

1. draft `q = warp(mtp_draft(pending, past))`, draw `x ~ q`
2. snapshot the caches
3. `verified = forward_verify({pending, x}, past)` — logits for `past+1` and
   `past+2`
4. accept or correct `x` against `p = warp(verified[0])`
5. accept → emit `{pending, x}`, next pending from `verified[1]`
   reject → restore, draw the correction from the residual, replay
   `forward_verify({pending, corrected}, past)`, emit `{pending, corrected}`,
   next pending from the replay

A miss costs a second backbone pass but still yields two tokens, so it degrades
to ordinary decoding rather than below it.

## Sampling correctness

The acceptance rule is `min(1, p(x)/q(x))`, and on rejection the emitted token
is drawn from the normalized residual `max(0, p - q)`. That is what makes the
output distribution exactly the target's.

**Accepting when the draft matches the target's argmax is only correct at
temperature 0.** Under sampling it over-weights whatever the draft head prefers
and serves a distribution that differs from the one the request asked for —
silently, and invisibly to any single-sequence comparison. At temperature 0 both
distributions are point masses and the general rule reduces to the argmax match,
so there is one implementation rather than two.

Both `p` and `q` are warped by the request's temperature / top-k / top-p before
the comparison. Comparing raw logits would be wrong: the target distribution the
caller asked for is the warped one.

`speculative_correct` in `inference/sampling.cpp` is split out from the decode
path so it can be tested against a known target without a GPU or a model.

### The rng stream

Emitted tokens are distributed exactly as the ordinary sampler's, but for a
given seed the *sequence* differs, because a round consumes randomness
differently (a draft draw, an acceptance coin, then the next token). Both are
valid draws from the same distribution; neither reproduces the other. At
temperature 0 nothing is drawn at all, so greedy output is identical either way
— which is what the end-to-end tests below check.

## Streaming and cancellation

A round yields one or two tokens, so both move from per-token to per-round
granularity. A round is not interrupted partway: its tokens are already verified
against the target distribution, and dropping the second would waste a backbone
pass already paid for. Deltas are still emitted one token at a time, so a
streaming client sees no change in granularity.

## Validation

`tests/modeling/qwen3_5/test_speculative_sampling.cpp` (ctest, no GPU) draws
200k corrected tokens per case and compares the empirical distribution against
the target. Accept rates match the theoretical `sum(min(p, q))` exactly:

| case | TV distance | accept rate | theory |
|---|---:|---:|---:|
| identical draft and target | 0.0018 | 1.000 | 1.000 |
| skewed draft | 0.0020 | 0.399 | 0.400 |
| partially disjoint supports | 0.0002 | 0.300 | 0.300 |
| point-mass draft, broad target | 0.0012 | 0.250 | 0.250 |

Plus two greedy cases: disagreement always rejects and emits the target's token,
and agreement accepts without touching the rng.

End to end, Qwen3.6-27B-NVFP4 on one BMG G31.

Engine level, `arcaine_mbench --model <dir> --spec --spec-tokens 128`:

```
acceptance        0.827        draft cost 0.052 of a step (= break-even)
baseline          12.39 tok/s
speculative       18.92 tok/s   1.53x
passes            75 for 128 tokens, 1.707 tokens per pass
sequences         identical to plain greedy, batched control 128/128
```

Serving level, same request through `/v1/chat/completions` at temperature 0,
96 completion tokens:

```
ARCAINE_QWEN35_MTP=0   8242 ms   11.65 tok/s
ARCAINE_QWEN35_MTP=1   6098 ms   15.74 tok/s   1.35x
text                   identical
```

The serving figure is lower than the engine figure because it includes prompt
processing, tokenization and HTTP overhead over a short completion.

Streaming at temperature 0 produced 39 content deltas for the same request and
reassembled to the expected text; temperature 0.8 produced coherent output that
varies with the seed.

## Reproduce

```
# engine level, includes the identical-sequence check
./build/arcaine_mbench --model <dir> --spec --spec-tokens 128

# serving level A/B
ARCAINE_QWEN35_MTP=0 ./build/arcaine_server --model <dir> --served-model-name m --port 8801
ARCAINE_QWEN35_MTP=1 ./build/arcaine_server --model <dir> --served-model-name m --port 8802
# same request to each at temperature 0; the text must match

ctest --test-dir build -R speculative_sampling
```

## Not done

Depth is fixed at one draft token per round. Compute at batch 1 is a few percent
utilized, so width is nearly free and a draft *tree* is the obvious next step —
but it has to beat depth-1, not beat no speculation. Linear acceptance compounds
(0.83^3 is 57%), which is the argument for trees over deeper linear drafts, and
is why llama.cpp lowered its own default from 16 to 3.
