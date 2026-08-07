# Prefill throughput was reported ~10x too high to streaming harnesses

Fixed in `ac29eab`. This note records how the number is actually derived
downstream, and the measurement that confirms the fix, so neither has to be
guessed at again.

## How llama-benchy derives PP

Read from the container's own source rather than inferred
(`docker exec llama-benchy-web`). Two places matter.

`src/llama_benchy/client.py:180`, which decides what "first response" means:

```python
if 'choices' in chunk and len(chunk['choices']) > 0:
    if result.first_response_ts is None:
        result.first_response_ts = chunk_time
```

`src/llama_benchy/results.py:292`, which turns that into throughput:

```python
ttfr     = res.first_response_ts - res.start_ts
est_ppt  = max(0, ttfr - latency)          # latency = measured API round-trip
pp_speed = prompt_tokens / est_ppt
```

So the reported prefill rate is set by **the arrival time of the first SSE
chunk carrying a non-empty `choices` array** — not the first byte, not the
response headers, not the first chunk with text in it.

The server used to emit a role-only chunk (`delta: {"role": "assistant"}`) on
`StartedEvent`, when the stream opened and before any prefill had run. That
chunk has a `choices` array, so it set `first_response_ts` to roughly socket
setup time — constant regardless of prompt length. `prompt_tokens` grows, the
denominator does not, and the reported rate scales with prompt size instead of
staying near the machine's real prefill rate.

`ac29eab` moves that chunk to immediately before the first chunk that carries
anything (`SseEventSink::ensure_role`). The wire shape is unchanged — clients
still see a role-only chunk first — only its timing.

## Confirming it

Same 2089-token prompt three ways, against the branch build serving on
`level_zero:2`:

```
server prefill_throughput, non-streaming    582.4 tok/s
server prefill_throughput, streaming        580.4 tok/s
llama-benchy's rule replayed by hand        502.4 tok/s   (first choices chunk at 4158 ms)
```

The streaming and non-streaming paths agree to 0.3%, so streaming does not
distort the engine's own metric. The harness rule lands ~14% low because
`ttfr` includes HTTP framing the engine's timer does not see (4158 ms against a
582.4-implied 3720 ms); llama-benchy subtracts its measured `latency` baseline
to compensate for exactly this.

A real llama-benchy run against the same server, pp 2048, tg 128, 2 runs:

```
depth     PP (tok/s)    TG (tok/s)    TTFR (ms)
0             516.7        14.92          3698
4096          427.9         9.56         12976
8192            —             —              —
```

Against what the same harness reported before the fix — 5,339 at depth 0,
11,091 at 4096, 22,098 at 8192, with TTFR pinned near 380 ms at every depth.
TTFR now grows with the work being done, and PP sits within 11% of the
engine's own prefill metric while erring **low**.

## The depth-8192 blank is correct

Not a regression. `--max-seq 8192` cannot hold 8192 of context plus a 2048
prompt plus 128 generated tokens, and the server says so:

```
{"error":{"message":"prompt_tokens + max_tokens exceeds model max_seq", ...}}
```

That error chunk carries no `choices`, so `first_response_ts` stays `None` and
llama-benchy records null rather than a number. A blank cell is the honest
outcome; the 22,098 it used to print for this configuration was measuring a
request the server never could have served. (The error is delivered inside a
200 response rather than as a 4xx — worth revisiting, but it is a separate
issue and the body itself is accurate.)

## Reproducing

llama-benchy is a Flask app wrapping the CLI, and needs an address reachable
from *its own* container, not `127.0.0.1`:

```
curl -X POST http://localhost:5059/api/start -H 'Content-Type: application/json' -d '{
  "base_url":"http://<arcaine-container-ip>:7461",
  "model":"qwen3.6-27b", "tokenizer":"qwen3.6-27b", "test_group":"custom",
  "custom_config":{"pp":[2048],"tg":[128],"depth":[0,4096],
                   "concurrency":[1],"enable_prefix_caching":false,"runs":2}}'
# -> {"run_id": ...}; GET /api/run/<id>/stream until done, then
#    GET /api/results/<id>/export/json
```

Two things to know when reading its output. It cannot resolve a served model
name against HuggingFace, so it falls back to the **gpt2 tokenizer** and its
local token counts are approximations; it prefers the server's reported
`prompt_tokens` when those are within 20% of its target, which is the case
here (2089 against a 2048 target). And a per-request failure is recorded as
null without surfacing a top-level error, so blank cells mean "every request
failed," not "not run."
