# Vision latency baseline

This note records the remote Ollama latency work completed on 2026-09-17. It is
a performance baseline, not a gesture-accuracy result.

## Production defaults

The console and `scripts/run-remote-vision.sh` now use the same rolling input:

```text
window:      0.8 seconds
frames:      3 chronological frames
in-flight:   1 request
queueing:    latest window wins
model:       qwen3.5:9b
Ollama:      OLLAMA_NUM_PARALLEL=1
```

The API can replay the previous input size without a source edit:

```bash
uv run --locked --extra perception --extra vision go2-api \
  --vision-window-s 2.0 --vision-frame-count 8
```

`OllamaVisionInvoker.warmup()` performs an empty generation request. A metadata
lookup alone does not load the model. Vision requests also ignore process proxy
variables so a local SSH tunnel at `127.0.0.1:11435` is not sent through an
HTTP proxy.

## Measurements

Host: RTX 4090 D, Ollama 0.32.5, `qwen3.5:9b` Q4_K_M. The benchmark used fresh
synthetic 640x480 noise images, so it isolates request performance and does not
measure gesture recognition. Warmup requests were excluded.

| Configuration | Median | P95 | Throughput |
| --- | ---: | ---: | ---: |
| 8 frames, one in flight | 1.655 s | 1.665 s | 0.528 req/s |
| 3 frames, one in flight | 1.204 s | 1.251 s | 0.772 req/s |
| 3 frames, two in flight | 1.615 s | 2.137 s | 1.056 req/s |
| 3 frames, final one-in-flight verification (8 requests) | 1.236 s | 1.311 s | 0.753 req/s |

Reducing the rolling input from eight to three frames cut median model latency
by about 27%. Two concurrent requests increased aggregate throughput but made
each reaction slower and worsened tail latency, so production remains single
in-flight.

The empty preload took 4.37 seconds after a service restart. It happens while
the UI reports that the visual model is loading, before the policy worker can
submit robot actions.

## Reproduce

Run on the inference host:

```bash
python3 scripts/benchmark-vision-latency.py --frames 8 3 --requests 8
```

Use actual consecutive camera captures for a representative replay:

```bash
python3 scripts/benchmark-vision-latency.py \
  --frames 3 --requests 8 --images frame-001.jpg frame-002.jpg frame-003.jpg
```

The script reports Ollama load, prompt evaluation and generation durations,
transport residual, schema completion and basic gesture/evidence consistency.
Only a labeled held-out camera replay or a physical-robot run can establish
recognition quality and end-to-end reaction time.

Runtime decision logs separate `client_encode_ms`, `http_round_trip_ms`,
server-side `inference_latency_ms`, and `transport_residual_ms`. Ollama does not
publish separate upload and download clocks, so `transport_residual_ms` (and the
legacy-compatible `network_rtt_ms`) is an upper bound that also contains HTTP
client serialization. `end_to_end_latency_ms` remains the authoritative latest
frame-to-decision measurement.

For the FastAPI camera path, RGB rotation is applied inside the native or
system-Python RealSense capture process before JPEG encoding. This avoids the
previous per-frame JPEG decode, rotate and second encode in the backend while
leaving depth sampling and person-detection coordinates in the native camera
orientation.

A local macOS microbenchmark with a 205,553-byte 640x480 JPEG measured the
removed decode/180-degree-rotate/re-encode path at 3.44 ms per frame, versus
0.08 microseconds to reuse the pre-rotated JPEG. At 30 FPS that removes roughly
103 ms of CPU work per second. These numbers demonstrate the eliminated fixed
work only; Jetson timing still requires an on-device measurement.

## Server service

The dedicated vision instance must remain isolated from the system Ollama
instance and use one parallel slot:

```bash
systemd-run --user --unit=go2-vision-ollama \
  --setenv=OLLAMA_HOST=127.0.0.1:11435 \
  --setenv=OLLAMA_NUM_PARALLEL=1 /usr/local/bin/ollama serve
```

## Go2 UnifoLM decision latency (2026-10-09)

The deployed two-stage pipeline uses UnifoLM for visual observation and
qwen3.5:9b for registered-skill selection. Server logs showed roughly 8,365
input tokens being truncated to 4,098. Later decisions also produced 123-token
explanations, taking 3–4.5 seconds per window; output-budget failures discarded
an entire round. The worker already samples the newest window with one request
in flight and checks frame freshness before execution.

The open visual prompt now keeps every skill, argument constraint and safety
tag while removing duplicate catalog fields and schema titles. The UnifoLM
decider explicitly requests an 8,192-token context, limits reason strings to
64 characters and speech to 160 characters through the output schema, and
uses a 256-token output budget. Runtime validation and rejection of truncated
responses remain in place.

Six alternating recorded waving/empty-image replays on the RTX 3080 server
(with prior failed-action context and local person_count=0) yielded:

| Input | First request | Subsequent requests |
| --- | ---: | ---: |
| Waving person | 3.181 s | 2.126 s, 1.944 s |
| Empty room | 1.377 s | 1.392 s, 1.370 s |

All three waving inputs selected wave and all three empty inputs selected
ignore. Inputs were 4,674–4,703 tokens; outputs were 23–44 tokens. These replay
measurements exclude robot execution and do not establish live gesture
accuracy or physical response time. The existing five-second identical-action
cooldown and two-second SDK timeout remain separate limits on repeated actions.

The later 2026-10-09 update removes the identical-action cooldown and its CLI
option. An identical behavior is still suppressed while active; it may execute
again as soon as completion is recorded. Decision prompts request JSON only,
and the UnifoLM decider schema excludes the reason field entirely. Internal
runtime diagnostic reasons remain available for rejected decisions.

### Current gesture evidence and repeated responses

A later live trace exposed a semantic error: an observation described the hand
being extended in the first frame and lowered in the second, yet the decider
selected wave. Ambiguous "could be interpreted as waving" observations also
selected hello. Live frame-to-decision ages were roughly 1.1–3.0 seconds;
these records did not show a growing backlog of queued camera windows.

The observer now prioritizes the last frame and uses at most eight words. The
decider emits a required gesture_state (wave/heart/none/uncertain) alongside its
JSON action. That metadata is removed before skill parsing. A wave/hello/heart
action must match current gesture evidence; past or ambiguous gestures are
ignored. The worker records one attempted response per continuous gesture,
including attempts with uncertain SDK acknowledgement. wave and hello share
that state. Only a fresh none observation rearms the same gesture; an ignore,
an uncertain observation or an old none observation cannot rearm it. The state
is supplied to the decider as responded_gesture, so repeated commands can be
suppressed before reaching execution as well. No confirmation timer or fixed
cooldown is involved. Console decision freshness is tightened from five to two
seconds; slow results are discarded rather than executed late.

Regression coverage includes gesture release/reappearance, alias switching,
uncertainty, stale release evidence and repeated SDK failure. This reduces
repeated and stale commands but does not eliminate model inference latency or
resolve SDK status 3104.
