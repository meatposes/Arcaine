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

Then the harness itself, pp 2048, tg 128, 2 runs, three ways — driven through
llapdance, driven through the dashboard's job API, and the CLI invoked
directly, the last both with its default tokenizer fallback and with the
model's real one:

```
                                  pp2048          pp2048 @ d4096      tg128
llapdance -> dashboard API         516.7               427.9          14.92
CLI direct, gpt2 fallback     517.59 ± 1.05       425.59 ± 0.59   15.94 ± 0.01
CLI direct, real Qwen tok     514.21 ± 0.24       410.98 ± 2.43   14.25 ± 0.37
```

All three agree within 0.7% at depth 0, and the run-to-run spread inside each
is well under 1%. Nothing about how the harness is driven moves the number.

The tokenizer matters less than expected. Pointing `--tokenizer` at the
model's own files (see below) changes the corpus tokenization materially —
144,677 tokens where gpt2 counted 159,582 — and lengthens `ttfr` at depth 0
from 3729 ms to 4028 ms, but `prompt_tokens` rises with it and the rate barely
moves. Its effect is larger at depth, where the priming context is
mis-sized too: 3.4% at d4096.

Against what the same harness reported before the fix — 5,339 at depth 0,
11,091 at 4096, 22,098 at 8192, with TTFR pinned near 380 ms at every depth.
TTFR now grows with the work being done, and PP sits within 13% of the
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

The CLI is the shortest path, and it prints its own table:

```
docker exec llama-benchy-web llama-benchy \
    --base-url http://<arcaine-container-ip>:7461 --model qwen3.6-27b \
    --runs 2 --pp 2048 --tg 128 --depth 0 4096 --concurrency 1 --format md
```

To use the model's real tokenizer instead of the gpt2 fallback, copy its
tokenizer files into the container's one writable volume and point
`--tokenizer` at them (`--model` stays the served name, which is what the API
calls use):

```
docker exec llama-benchy-web mkdir -p /app/data/qwen-tok
for f in tokenizer.json tokenizer_config.json vocab.json added_tokens.json \
         special_tokens_map.json generation_config.json config.json; do
    docker cp "$MODEL_DIR/$f" llama-benchy-web:/app/data/qwen-tok/
done
# then add: --tokenizer /app/data/qwen-tok
```

The dashboard is the same CLI behind a Flask job API, and needs an address
reachable from *its own* container, not `127.0.0.1`:

```
curl -X POST http://localhost:5059/api/start -H 'Content-Type: application/json' -d '{
  "base_url":"http://<arcaine-container-ip>:7461",
  "model":"qwen3.6-27b", "tokenizer":"qwen3.6-27b", "test_group":"custom",
  "custom_config":{"pp":[2048],"tg":[128],"depth":[0,4096],
                   "concurrency":[1],"enable_prefix_caching":false,"runs":2}}'
# -> {"run_id": ...}; GET /api/run/<id>/stream until done, then
#    GET /api/results/<id>/export/json
```

Two things to know when reading its output. Without `--tokenizer` it cannot
resolve a served model name against HuggingFace, so it falls back to the
**gpt2 tokenizer** and its local token counts are approximations; it prefers
the server's reported `prompt_tokens` when those are within 20% of its target,
which is the case here (2089 against a 2048 target), so the fallback stays
survivable — but it is worth fixing, and the numbers above quantify what it
costs. And a per-request failure is recorded as null without surfacing a
top-level error: in the markdown table the row **vanishes entirely** rather
than printing blanks, which is how depth 8192 disappears above. A missing row
means "every request failed," not "not run."
