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

Started at 33 branches, ~24 Claude-authored. After the 2026-08-08 pass: 15
Claude-authored branches remain.

| disposition | branches |
|---|---|
| **Live, clean, in flight** | `fix/qwen35-prefill-ba-layout` (PR #11) · `test/qwen35-numerical-verification` · `fix/qwen35-conv-layout-and-decode-guard` (archive of the session, not shippable as-is) |
| **New, validated tooling** | `bench/qwen35-dense-projection` — ported and running on current upstream |
| **Deleted** | 10 branches, §5. Two archived as remote tags first. |
| **Dead: upstream already fixed it** | `fix/server-error-body` · `fix/stream-chunk-null-usage` — proven, §4 |
| **Live, proven, not upstream** | `fix/defer-stream-role-chunk` |
| **Novel content, needs rewrite not rebase** | `feat/per-request-chat-template-kwargs` · `fix/qwen35-tool-call-tolerance` · `fix/arch-dispatched-output-parsing` |
| **Premise proven, feature unported** | `perf/qwen35-prefill-dequant-bf16` |
| **Absent upstream, unmeasured** | `perf/qwen35-prof-scopes` · `perf/qwen35-attention-kernel-by-phase` · `integration/nvfp4-27b` = `perf/qwen35-decode-attention` (identical) |
| **Verified win, needs port to the service refactor** | MTP speculative decoding — 1.464x, lossless, re-measured 2026-08-08, §3 |

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

## 3. MTP speculative decoding — RE-MEASURED 2026-08-08, claim holds

**Verified.** Re-run on `upstream/arch-refactor` HEAD merged with PR #11's
fixes, built on the rebuilt image (oneAPI 2026.1, oneDNN v3.13 with grouped
memory), pinned to the scratch GPU. Control first: the engine is bit-exact
2/2 in-process, so the acceptance figure is trustworthy.

```
backbone step         94.15 ms  (median of 128)
mtp draft              5.04 ms  = 0.054 of a step
acceptance            0.8661    (110/127)
break-even            0.0536

end to end, greedy, 128 tokens, 1164-token prompt
baseline              13963.6 ms    9.17 tok/s
speculative            9535.1 ms   13.42 tok/s
speedup                                1.464x
backbone passes       73 for 128 tokens  (1.753 tok/pass)
draft/verify/rollback  322.6 / 7304.5 / 49.0 ms
batched control       128/128 tokens reproduced
sequences             identical (128 tokens)
```

Acceptance sits **16x above break-even**, and the speculative path emits
byte-identical output to the greedy baseline — losslessness demonstrated, not
argued. Rollback costs 49 ms against 9.5 s of generation.

The suspicion recorded below was correct to raise and did not survive contact:
the fused-BA bug did not materially distort acceptance (86.6% now against
76-93% claimed before). The earlier throughput figures were nonetheless taken
on a different code state; **1.464x is the number to quote.**

**This is the only measured win on the fork that costs nothing in output
quality.** FP8 buys more decode but changes 5.5% of tokens; this changes none.
Remaining work is porting it onto upstream's service refactor, where
`session.cpp` now owns per-request cache lifetime — precisely what speculative
rollback interacts with.

### Original assessment, kept for the record

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

## 4. Section-4 branches, now validated

Each claim here was tested the same way: the claim is "X is broken or missing
upstream", so it is proved by showing the code in current upstream, and the fix
is proved by running it. Two claims died on contact.

### Dead — upstream already fixed it

| branch | proof |
|---|---|
| `fix/server-error-body` | `server_app.cpp:83` already carries `if (!res.body.empty()) { ...; return; }` — the same guard with the same rationale. Only residual difference is that upstream synthesizes "not found" for any empty-body status where the branch made the message status-aware. Cosmetic. |
| `fix/stream-chunk-null-usage` | `sse_event_sink.cpp` already carries `if (!usage.is_null()) out["usage"] = ...` and the same for metrics. Exactly the branch's change. |

Both were labelled *unverified* in the first pass. Both would have been
re-proposed as fixes for bugs that no longer exist.

### Live and proven

| branch | proof |
|---|---|
| `fix/defer-stream-role-chunk` | Upstream still emits the role chunk on `StartedEvent` (`sse_event_sink.cpp:79-82`). Measured impact: llama-benchy reports PP 5,339 t/s against the engine's own 582.4; with the fix, 517. Also present on the working line as `ac29eab`. |

### Novel content, obsolete mechanism — rewrite, do not rebase

| branch | proof | what survives |
|---|---|---|
| `feat/per-request-chat-template-kwargs` | Upstream reads kwargs only from the server-wide option (`request_decoder.cpp:265` — `gen.chat_template.kwargs = app.opts.chat_template_kwargs`), never from the request body. | The feature. The patch targets `src/arcaine_server.cpp`, which no longer exists. |
| `fix/qwen35-tool-call-tolerance`, `fix/arch-dispatched-output-parsing` | Arch dispatch is already upstream via per-model `output_parser.cpp`, so that half is obsolete. But upstream's qwen3_5 parser has no `True`/`False`/`None` handling, and `json::parse("True")` throws — Python-style tool arguments fail today. | The literal coercion and spelling tolerance. Needs porting into `src/modeling/qwen3_5/output_parser.cpp`. |

### Premise validated, feature unconfirmed

**`perf/qwen35-prefill-dequant-bf16`** — claim: route large-M projections
through a BF16 expansion instead of oneDNN's f4 matmul. Claimed pp512 515→781,
pp2048 491→983, decode unchanged, no extra VRAM.

The premise reproduces on the current toolchain. `out_proj` (6144x5120), ms:

| M | nvfp4 | bf16 | dequant+bf16 |
|---:|---:|---:|---:|
| 1 | 0.1183 | 0.0985 | 0.5893 |
| 64 | 0.1663 | 0.1092 | 0.5973 |
| 256 | 0.4403 | 0.1699 | 0.5913 |
| 512 | 0.9228 | 0.2680 | **0.6680** |
| 2048 | 3.5694 | 0.8326 | **1.2489** |

f4 against bf16 at M=2048 is **4.29x**, matching the 4.3x measured in July.
dequant+bf16 overtakes f4 between M=256 and M=512 and reaches 2.86x at M=2048.

**A port was attempted and deliberately abandoned.** The MLP path conflicts
non-mechanically — upstream refactored it into per-format variant branches —
and forcing the merge would have silently reverted the PR #11 fixes in the same
file. The 1.5-2.0x prefill figure is a *model-level* claim and remains
unconfirmed; the same July series contains an attention change that looked 20%
faster in isolation and measured 2% slower end to end.

The measuring tool was ported instead: **`bench/qwen35-dense-projection`**
(`44bec09`), builds and runs on current upstream.

### Absent upstream, still unmeasured

`perf/qwen35-prof-scopes` (DIFF_PROFILE instrumentation, no perf claim),
`perf/qwen35-attention-kernel-by-phase` (a negative result — its value is the
documentation of what was rejected), and
`perf/qwen35-decode-attention` = `integration/nvfp4-27b` (identical commits;
split-KV decode attention, never measured end to end).

---

## 5. Deleted 2026-08-08

Ten branches removed. Two held commits reachable from nowhere else and were
archived as tags on the remote first, verified before deletion:
`archive/bench-golden-gate-roofline` (11 commits) and
`archive/perf-qwen35-gemv-geometry-probe` (1).

`fix/qwen35-conv-state-layout` (merged as PR #10) ·
`fix/qwen35-recurrent-state-reset` (superseded — upstream's `session.cpp:48-49`
resets the cache per request) · `bench/golden-gate-roofline` ·
`nvfp4/config-probe` · `nvfp4/loader-probe` · `perf/qwen35-gemv-geometry-probe`
(marked do-not-merge) · `fix/onednn-runtime-path` ·
`perf/qwen35-dense-projection-bench` · `perf/qwen35-dequant-bf16-prefill` ·
`perf/qwen35-mtp-acceptance-spike`

---

## 6. What can be tested for performance right now

Three levers exist. Only one is both large and free.

### MEASURED — MTP speculative decoding, 1.464x and lossless

Done, §3. Acceptance 0.8661 against a break-even of 0.0536, end to end 9.17 →
13.42 tok/s, output byte-identical to the greedy baseline, on latest upstream +
PR #11 + the rebuilt toolchain, with a bit-exact control.

**This is the best performance option on the fork** and the only one that costs
nothing in output quality. It is no longer a measurement question; it is a
porting question — onto upstream's service refactor, where `session.cpp` owns
per-request cache lifetime and speculative rollback has to fit around it.

### Testable today, no new code — FP8 MLP requantization

`ARCAINE_QWEN35_MLP_FP8_LAYERS=56`, plus `16` / `32` as a partial dial.
Speed measured at 1.57x decode / 2.34x prefill, but on the previous
environment — worth re-running on oneAPI 2026.1, which takes minutes.

Quality is already re-measured on the corrected engine and is the problem:
~5.5% of predicted tokens change. Fine for throughput-shaped work, not for
general serving. Verdict stands at "off by default".

### Needs a port first — prefill dequant to BF16

Premise proved above (4.29x f4-vs-bf16, crossover M 256-512). If the
model-level claim holds it is the best prefill option available: 1.5-2.0x with
no quality cost and no resident VRAM, where FP8 buys 2.34x for +6.5 GB and 5.5%
token drift. Blocked on integrating with upstream's per-format MLP dispatch.

### Not a lever yet — concurrency

Every measurement in this repository is at concurrency 1. For a serving engine
that is the largest untouched throughput dimension, and it needs no new
kernels. Correctness first: the workspace buffers `tmp0`-`tmp4` are reused
across stages within a request, and whether concurrent requests share a
workspace has never been checked.

### Already ruled out, do not re-litigate

Writing an M=1 f4 GEMV (two independent implementations converge at ~43% of
roofline; the format is the ceiling, not the kernel), `NVFP4_DPAS=1`, oneDNN
weight-layout reorders, and attention-kernel-by-phase selection. See
`decode_bandwidth_gap.md` and `kernel_flag_numerics.md`.

---

## 7. Standing caveat

Every number in this file that is not labelled **verified** was produced by a
process that has repeatedly made confident, wrong claims — including several
assertions that upstream code or the author's configuration was broken, every
one of which turned out to be my own error. In this pass alone, two branches
labelled *unverified* turned out to be fixing bugs upstream had already fixed.
Treat **unverified** as "unknown", not "probably fine".
