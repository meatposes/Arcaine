# Branch inventory: what is on the fork, what it is worth, what still applies

Written 2026-08-08 against `upstream/arch-refactor` @ `08be071` and open PR #11.

The fork carries 33 branches, ~24 of them Claude-authored across several
sessions. This is an honest audit of what each is, whether its claims survived
verification, and whether it still applies after upstream's service refactor
and this session's fixes.

**Read the "truth" column with the confidence label attached.** Several claims
made in earlier sessions turned out to be wrong when finally tested, and one
whole class of measurement is now suspect for a reason that was not known when
it was taken. Those are called out rather than quietly dropped.

## Confidence labels

| label | meaning |
|---|---|
| **verified** | measured on the current, corrected environment, with a control |
| **measured** | measured on an older environment; the measurement is timing-only, so the fused-BA bug below does not affect it |
| **suspect** | measured before the fused-BA bug was found, and the quantity measured *depends* on model output being correct |
| **superseded** | upstream or later work already does this, by the same or another mechanism |
| **retracted** | previously claimed, since shown wrong |
| **unverified** | never validated; the claim is the commit message's word only |

### The thing that makes older measurements suspect

The fused `[b|a]` projection fed the DeltaNet recurrence the wrong gate values
for every prefill token after the first, on the default configuration, for any
prompt longer than 8 tokens (fixed in PR #11). Every measurement taken before
2026-08-07 was taken on an engine computing the wrong thing.

Timing measurements survive this — the wrong math costs the same work. Anything
that depends on *output quality* does not. That specifically includes **MTP
draft acceptance**, which is a measure of how often two model predictions agree
and is therefore invalid until re-measured.

---

## Disposition summary

| disposition | branches |
|---|---|
| **Live, clean, in flight** | `fix/qwen35-prefill-ba-layout` (PR #11), `test/qwen35-numerical-verification`, `fix/qwen35-conv-layout-and-decode-guard` |
| **Delete: merged** | `fix/qwen35-conv-state-layout` (landed as PR #10) |
| **Delete: superseded** | `fix/qwen35-recurrent-state-reset`, `bench/golden-gate-roofline`, `nvfp4/config-probe`, `nvfp4/loader-probe`, `fix/onednn-runtime-path` |
| **Delete: explicit do-not-merge probes** | `perf/qwen35-gemv-geometry-probe` |
| **Duplicates of one another** | `integration/nvfp4-27b` = `perf/qwen35-decode-attention`; `perf/qwen35-dequant-bf16-prefill` ≈ `perf/qwen35-dense-projection-bench` |
| **Real unlanded value, needs rebase** | the four server fixes, tool-call handling, `perf/qwen35-prefill-dequant-bf16`, `perf/qwen35-prof-scopes` |
| **Real value, needs rework + re-measurement** | MTP speculative decoding |

---

## 1. Live branches (clean against current upstream)

### `fix/qwen35-prefill-ba-layout` — PR #11 — 2 commits, 2 files, 0 conflicts
**Category:** correctness fix.
**Claims:** two DeltaNet state-layout bugs.
**Truth: verified.** The fused `[b|a]` output layout mismatch is provable by
reading `matmul_bf16`'s memory descriptor — no measurement required. Its impact
was measured at top-1 0.87 over 200 records, and the fix restores 10/10
cross-process bit-exactness. The second (`ESIMD_DELTA=0` splitting the
recurrent state across two layouts) was found by a falsifiable prediction that
held: the affected arm moved 0.3375 → 0.9275 exactly as predicted.
**Applies:** yes, cleanly. Highest-value work on the fork.

### `test/qwen35-numerical-verification` — 1 commit, 9 files, 0 conflicts
**Category:** validation/testing.
**Contents:** golden gate (`--golden capture|compare`), the `--repeat`
in-process probe, fp64 oracles for the DeltaNet recurrence and causal GQA
attention, two allocation diagnostics.
**Truth: verified.** These found both bugs above. The oracles reproduce
identical numbers across two toolchains (attention 0.90/0.90/1.04 bf16 ULP,
deltanet 0.67).
**Applies:** yes, cleanly. Benchmark-only, nothing on the inference path.
**Note:** builds need `-DARCAINE_MODELS="qwen3_5"` only if the environment
cannot build MoE; on a correct image the default build works.

### `fix/qwen35-conv-layout-and-decode-guard` — 20 commits, 34 files, 0 conflicts, 2 behind
**Category:** mixed — this is the session's working line and is not a coherent
unit. It contains the PR #11 fixes, the verification harness, MTP speculative
decoding, FP8 requantization, the roofline work, and all the notes.
**Applies:** merges cleanly but **should not be proposed as-is** — it is too
broad to review. It is the archive; the shippable pieces are being split out of
it (PR #11 and the test branch already were).

---

## 2. Work units inside the working line, assessed individually

### FP8 MLP requantization (`ARCAINE_QWEN35_MLP_FP8_LAYERS`)
**Category:** performance, opt-in.
**Claim:** 1.57x decode, 2.34x prefill.
**Truth: verified for speed, verified for cost, verdict is "leave off".**
Re-measured on the corrected deterministic engine: changes ~5.5% of predicted
tokens (top-1 0.945 over 1000 records). The earlier "+6% perplexity" figure was
**retracted** — it was measured on a nondeterministic engine and was too kind.
Perplexity is not monotonic in the clip parameter at any sample size tried, so
clip is not tunable with this instrument.
**Applies:** yes. Keep as an opt-in flag, default off.

### Decode bandwidth analysis + the M=1 GEMV dead end
**Category:** performance investigation (negative result).
**Truth: measured.** Pure timing, unaffected by the BA bug. NVFP4 MLP layers
run at 36% of achievable bandwidth against 88% for FP8 layers in the same
model. Five candidate fixes measured and eliminated, including a hand-written
ESIMD GEMV that ties oneDNN within 3%. Two independent implementations
converging at ~43% of roofline is the durable result.
**Applies:** yes, as documentation. The code is behind default-off flags.

### Measured roofline denominator (590 GB/s)
**Category:** validation.
**Truth: measured.** Replaces a hardcoded 456 GB/s. Two traps documented
(short-circuit DCE killing the loop, and GPU memory compression making a
uniform fill read ~2400 GB/s).
**Applies:** yes.

### Server: defer the streaming role chunk (`ac29eab`)
**Category:** reporting fix.
**Truth: verified.** Upstream still emits the role chunk on `StartedEvent`
(`sse_event_sink.cpp:79-82`), which makes any harness timing first-response
report prefill throughput inflated ~10x. Confirmed by reading llama-benchy's
source (`client.py:180` sets `first_response_ts` on the first chunk with a
non-empty `choices` array) and by running it: 5,339 → 517 t/s, against the
engine's own 582.4.
**Applies:** yes, **not in upstream**. Good standalone PR candidate.

---

## 3. MTP speculative decoding — real, but needs re-measurement

Spread across `bench/golden-gate-roofline` and the working line
(`d1cf798`, `23ada87`, `7c7666e`, `f214902`, `66bd47a`, `c31a8a6`).

**Category:** feature + performance.
**Claim:** 1.51x engine, 1.35x serving, acceptance 76-93%.

**Truth: split.**
- The *implementation* is real: an MTP head, speculative rejection sampling
  that provably preserves the target distribution (200k-draw test, TV
  0.0002-0.0020), cache snapshot/restore for rollback, and a genuine bug fix
  along the way (a `DEVICE_LOST` that was a host use-after-free from async
  `queue.memcpy` over loop-local vectors, not a kernel fault).
- The *acceptance rate is suspect.* Acceptance measures how often the draft
  head agrees with the backbone. Both were computed on an engine whose prefill
  gates were corrupted. The number must be re-measured before it is quoted.
- The throughput figures are also **stale**: measured pre-fix and on the old
  environment.

**Applies:** the code needs rework against upstream's service refactor —
`session.cpp` now owns per-request cache lifetime, which is exactly what
speculative rollback interacts with. This is the largest unshipped feature and
the one needing the most care.

---

## 4. Unlanded work with real value, needs rebase (all 21 commits behind)

| branch | category | truth | note |
|---|---|---|---|
| `fix/defer-stream-role-chunk` | reporting fix | **verified** | 1 file, 1 conflict. Also carried on the working line. Not upstream. |
| `fix/stream-chunk-null-usage` | protocol fix | unverified | Omits `usage`/`metrics` rather than sending `null`. Small, plausible; llama-benchy does `chunk['usage'].get(...)`, which would crash on null — so this likely matters. Never explicitly tested. |
| `fix/server-error-body` | correctness fix | unverified | Stops the error handler overwriting real error bodies. Never confirmed present or tested. |
| `feat/per-request-chat-template-kwargs` | feature | unverified | 1 file. |
| `fix/qwen35-tool-call-tolerance` / `fix/arch-dispatched-output-parsing` | feature/robustness | unverified | Tool-call parsing tolerance; has tests on the branch. |
| `perf/qwen35-prefill-dequant-bf16` | performance | **measured** | Dequantize NVFP4 → BF16 for large-M projections; crossover measured near M~512. Timing-only, so survives the BA bug. |
| `perf/qwen35-prof-scopes` | validation | unverified | `DIFF_PROFILE` instrumentation scopes. |
| `perf/qwen35-attention-kernel-by-phase` | performance (negative) | unverified | Includes "the measurement that rejected phase selection" — a negative result worth keeping as documentation. |
| `perf/qwen35-decode-attention` = `integration/nvfp4-27b` | performance | unverified | Split-KV decode attention. 13 conflicts. Identical commits — one of these should be deleted outright. |

---

## 5. Delete

| branch | why |
|---|---|
| `fix/qwen35-conv-state-layout` | **merged** as PR #10; 0 ahead |
| `fix/qwen35-recurrent-state-reset` | **superseded.** It added `if (past_len == 0) reset_cache()` because nothing reset DeltaNet state between server requests. Upstream's service refactor now constructs a `Qwen35Cache` and calls `cache.reset()` per request (`session.cpp:48-49`). Landing this would be dead code. |
| `bench/golden-gate-roofline` | superseded by the working line's newer versions of the same commits |
| `nvfp4/config-probe`, `nvfp4/loader-probe` | investigation probes, findings already in notes |
| `perf/qwen35-gemv-geometry-probe` | commit says "investigation, do not merge" |
| `fix/onednn-runtime-path` | the oneDNN version pin it sets (v3.13) already matches upstream's `docker-compose.yml`; the image has since been rebuilt to the author's full spec |
| `perf/qwen35-dense-projection-bench`, `perf/qwen35-dequant-bf16-prefill`, `perf/qwen35-mtp-acceptance-spike` | intermediate points on the same chain; their distinct content is captured above |

---

## 6. What I would do, in order

1. **Land PR #11.** Verified, small, fixes a live default-path correctness bug.
2. **Offer the test branch as PR #2.** It is what found #1, and without it the
   next bug of that class goes unnoticed.
3. **Split out the server role-chunk fix** as a small standalone PR — verified,
   not upstream, one file.
4. **Delete the 9 branches in section 5.** They are noise that makes the fork
   hard to reason about.
5. **Triage section 4 by cheap verification**, not by reading commit messages.
   Most are one file; several can be confirmed or dropped in minutes each.
6. **Re-measure MTP acceptance last**, on the corrected engine, before deciding
   whether the feature is worth the rework against the service refactor.

## 7. Standing caveat

Every number in this file that is not labelled **verified** was produced by a
process that has since been shown to make confident, wrong claims — including
five separate assertions this session that upstream code or the author's
configuration was broken, all five of which were my own error. Treat
**unverified** as "unknown", not as "probably fine".
