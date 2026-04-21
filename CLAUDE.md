# CLAUDE.md

This file gives Claude (Claude Code, Antigravity, or any AI coding assistant) the context needed to work on this project effectively. Read this entire file before writing or modifying any code.

## Project identity

**Name:** edge-vlm-assistant
**One-line:** A smart-glasses AI assistant simulator that runs entirely on-device
 on a MacBook Air, achieving <800ms voice-to-response latency.
**Owner:** Nithilan Sathariels — undergrad CS, applying for ML/systems internships at Meta, Anthropic, and similar.

## Project thesis (read this twice)

This project is **not** "I built a webcam Q&A demo." Anyone can wire GPT-4V to a webcam in an afternoon.

This project **is** "I built a system that mirrors the compute, memory, and latency constraints of a real wearable device (Ray-Ban Meta: ~4GB RAM, Snapdragon AR1, tiny battery), and here is the engineering I did to make it feel instant."

**Every architectural decision must be defensible through the lens of edge-AI constraints.** When in doubt, ask: "Would this work on the actual Ray-Ban Meta hardware?" If the answer is no, reject it.

## Hard constraints — never violate these

1. **No cloud APIs for inference.** No OpenAI, Anthropic, Gemini, AWS, or any hosted model API. All vision, ASR, TTS, translation, and OCR must run locally. The entire point of the project is on-device inference. A cloud call for any inference task invalidates the thesis.
2. **No model larger than what fits the wearable RAM budget.** Working set target is ~4GB at runtime. Loaded models combined should not exceed this when feasible.
3. **No batching.** This is a single-user realtime system. Batching adds latency for no benefit.
4. **No blocking architecture.** Capture, audio, and inference must run in parallel loops. Never block one on another.
5. **No silent failures.** Every stage must be instrumented with timestamps. Latency observability is a core deliverable, not an afterthought.

## Latency budget (the project's headline metric)

**Target:** <800ms from end-of-speech to first audio token of response.

| Stage | Budget | Notes |
|-------|--------|-------|
| VAD end-of-speech detection | 100ms | Inherent in Silero |
| ASR (distil-whisper, ~3s utterance) | 150ms | Via faster-whisper / CTranslate2 |
| Frame retrieval from ring buffer | ~1ms | O(1) lookup |
| VLM first-token latency | 400ms | Largest single budget item |
| TTS first-chunk | 150ms | Piper streaming |
| **Total** | **~800ms** | |

Every measurement must be wall-clock, captured with `time.perf_counter()`, and logged per-query. The latency dashboard is a primary deliverable, not a debug tool.

## Architecture: three parallel loops

This is the single most important design decision. Loops run **concurrently**, not sequentially.

### Loop 1 — Capture loop (continuous, ~30fps)
- Webcam frames → downsample to 384x384 or 448x448 (VLM-native resolution) → push to ring buffer (size 30, fixed allocation)
- Never blocks on inference
- Always provides the freshest frame on demand
- Implementation: `cv2.VideoCapture` in a dedicated thread

### Loop 2 — Audio loop (continuous)
- Microphone (16kHz mono) → Silero VAD → speech segment buffer
- When VAD detects end-of-speech, kick off ASR on the buffered segment
- ASR (distil-whisper-small.en via faster-whisper) runs once per utterance, never on silence
- Implementation: `sounddevice` callback in a dedicated thread

### Loop 3 — Inference loop (event-driven)
- Triggered by ASR transcript availability
- Grabs the latest frame from the ring buffer (NOT the frame at time-of-speech-start)
- Sends frame + transcript to VLM
- Streams VLM tokens directly into Piper TTS as they arrive
- TTS audio chunks stream to output device immediately

### Critical principle: stream everything, block nothing
TTS must start emitting audio before the VLM is done generating. The user perceives latency as "time until first syllable," not "time until generation complete."

## Tech stack

### Backend (Python 3.12, in venv)
- `mlx-vlm` — Moondream 3 inference on Apple Silicon (or `transformers` with `device_map="mps"` for Moondream 2 on 8GB systems)
- `faster-whisper` — ASR via CTranslate2 backend
- `piper-tts` — Streaming neural TTS
- `silero-vad` — Voice activity detection
- `opencv-python` — Webcam capture
- `sounddevice` — Audio I/O
- `numpy` — Array ops
- `fastapi` + `uvicorn` + `websockets` — Frontend bridge
- `pvporcupine` — Wake word (stretch, week 4 only)

### Frontend (Next.js / React)
- Live webcam feed display
- Latency waterfall chart (per-query, real-time)
- Live transcript stream
- Streaming VLM response display
- Visual styling: looks like a wearable debug console, not a consumer app

### What we are NOT using (and why)
- **LLaVA / LLaVA-NeXT**: too big (7B+) for the wearable constraint thesis
- **GPT-4V / Claude Vision / Gemini**: cloud APIs violate the local-only constraint
- **Whisper large**: distil-whisper-small.en is 6x faster with negligible accuracy loss
- **Cloud TTS (ElevenLabs, etc.)**: violates local-only constraint and adds network latency
- **Generic English NMT models**: MarianMT per-language-pair is smaller and faster than NLLB-200 for the single-language demo case

## Models on disk

All models live in `~/models/`. Do not redownload — check existence first.

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| Moondream 3 (MLX, int4) | `~/models/moondream3-mlx` | 6.5 GB | Vision-language model |
| distil-whisper-small.en | `~/models/whisper` | 320 MB | ASR |
| Piper voice (lessac-medium) | `~/models/piper` | 61 MB | TTS |
| MarianMT es→en | `~/models/marian-es-en` | 599 MB | Translation (Phase 2) |
| Silero VAD | `~/.cache/huggingface/...` | ~2 MB | Auto-downloaded by `silero-vad` package |

## Development environment

- **OS:** macOS (MacBook Air 2024, M3)
- **Python:** 3.12.3, in `~/glasses-project/venv/`
- **Always activate venv before running anything:** `source ~/glasses-project/venv/bin/activate`
- **Inference backend:** MLX preferred on Apple Silicon. Fall back to PyTorch MPS if MLX path has issues.
- **Thermal note:** the Air is fanless and throttles after sustained load. Benchmark in short bursts (10-20 queries), let it cool between runs. This is also a feature for the writeup — a real wearable constraint we accidentally simulate.

## Project structure

```
edge-vlm-assistant/
├── CLAUDE.md                    # This file
├── README.md                    # User-facing docs
├── requirements.txt
├── .gitignore
├── .env.example                 # No actual secrets — there are none
├── pyproject.toml               # If we go that route
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, latency targets, model configs
│   ├── ring_buffer.py           # Frame ring buffer (fixed allocation)
│   ├── capture.py               # Loop 1: webcam capture thread
│   ├── audio.py                 # Loop 2: mic + VAD + ASR pipeline
│   ├── inference.py             # Loop 3: VLM + TTS streaming pipeline
│   ├── vlm.py                   # Moondream wrapper (MLX or MPS)
│   ├── asr.py                   # faster-whisper wrapper
│   ├── tts.py                   # Piper streaming wrapper
│   ├── vad.py                   # Silero VAD wrapper
│   ├── ocr.py                   # PaddleOCR wrapper (Phase 2)
│   ├── translate.py             # MarianMT wrapper (Phase 2)
│   ├── orchestrator.py          # Wires the three loops together
│   ├── metrics.py               # Latency timestamping, waterfall data
│   └── server.py                # FastAPI + websocket bridge to frontend
│
├── frontend/                    # Next.js app
│   ├── app/
│   │   ├── page.tsx             # Main demo UI
│   │   └── layout.tsx
│   ├── components/
│   │   ├── WebcamFeed.tsx
│   │   ├── LatencyWaterfall.tsx
│   │   ├── TranscriptStream.tsx
│   │   └── ResponseStream.tsx
│   └── lib/
│       └── websocket.ts
│
├── benchmarks/                  # Latency measurement scripts
│   ├── run_baseline.py          # Pre-optimization numbers
│   ├── run_optimized.py         # Post-optimization numbers
│   └── results/                 # JSON dumps of latency data
│
├── scripts/
│   ├── sanity_check.py          # Verifies all models load
│   ├── download_models.sh       # One-shot model fetcher (idempotent)
│   └── measure_baseline.py
│
└── docs/
    ├── architecture.md          # Diagram + design decisions
    ├── latency-deep-dive.md     # Source for the Medium post
    └── tradeoffs.md             # What we considered and rejected
```

## Build plan — phase-by-phase

Work strictly in this order. Do not skip ahead. Each phase has a measurable deliverable.

### Phase 1: Independent components (week 1)
**Goal:** prove each loop works in isolation. No integration yet.

Deliverables:
- `capture.py`: webcam → display frames at 30fps → measure capture latency
- `audio.py`: mic → VAD → print "speech detected" / "speech ended" with timestamps
- `vlm.py`: load Moondream, run one inference on a static image, print first-token latency
- `tts.py`: text input → audio output, measure first-chunk latency
- `scripts/sanity_check.py`: loads every model, prints ✓ for each

Do NOT integrate yet. Do NOT optimize yet. Just measure baseline numbers and write them down.

### Phase 2: End-to-end integration (week 2)
**Goal:** unoptimized full pipeline working end-to-end. Establish the "before" numbers.

Deliverables:
- `orchestrator.py`: wires all three loops with proper threading
- `ring_buffer.py`: fixed-size, lock-free where possible
- `metrics.py`: per-stage timestamping, output waterfall JSON per query
- Working CLI demo: speak into mic, hear response
- **Document baseline latency numbers in `benchmarks/results/baseline.json`**

### Phase 3: Optimization sprint (week 3)
**Goal:** hit <800ms. Every optimization gets a before/after measurement.

Optimizations in priority order (by expected impact):
1. **Quantize VLM to int4** if not already (Moondream 3 MLX is already int4 — verify)
2. **KV cache warmup** — dummy inference at startup
3. **Streaming TTS** — pipe VLM tokens to TTS as they generate
4. **Image resize before VLM** — never send raw 1080p
5. **Thread pinning** — `os.sched_setaffinity` where applicable on macOS (limited)
6. **Pre-allocated buffers** — no per-frame numpy allocation
7. **uvloop** instead of default asyncio for the websocket layer

Add the OCR + translation feature in this phase.

### Phase 4: Polish & writeup (week 4)
**Goal:** demo-ready project + Medium post.

Deliverables:
- Frontend dashboard showing live waterfall
- 90-second demo video
- README with architecture diagram, latency table, design decisions
- Medium post draft on Stackademic
- Deployed demo page on sathariels.com or Vercel
- Optional: C++ hot-path port if time permits

## How to think about feature requests during the build

When asked to add a feature, evaluate against the thesis:
- **Does it improve perceived latency?** → high priority
- **Does it demonstrate edge-AI engineering?** → high priority
- **Is it a generic LLM feature dressed up?** → low priority (skip unless free)
- **Does it require a cloud call?** → reject
- **Does it add a model that pushes RAM over budget?** → reject or replace existing model

## Coding standards

- **Python:** type hints everywhere. `mypy --strict` should pass. Use `ruff` for linting.
- **No premature abstraction.** This is a 4-week project. Three loops, one orchestrator, done. Don't build a plugin system.
- **Profile before optimizing.** Use `py-spy` or `cProfile`. Never optimize on intuition.
- **Every long-running operation gets a timer.** Use the `metrics.timer(name)` context manager (to be built in Phase 2).
- **Logging:** structured (use `structlog` or stdlib `logging` with JSON formatter). Latency events go to a separate channel from regular logs.
- **Tests:** unit tests for `ring_buffer` and `metrics` (deterministic logic). Integration tests are demo runs — no need to mock the models.

## What "done" looks like

The project is done when:
1. End-to-end voice→vision→speech latency is reliably under 800ms (p50), measured over 100+ queries
2. The latency dashboard renders a per-query waterfall in real-time
3. Three demo flows work: scene description, object Q&A, OCR + translation
4. README has architecture diagram, latency table (before/after), and tradeoffs section
5. Medium post is published
6. Demo video is recorded and linked from the README

## Frequently asked questions Claude should know the answer to

**Q: Why not use OpenAI for vision since it's better?**
A: This project is about local edge inference. Cloud APIs invalidate the thesis. Even one cloud call breaks the wearable simulation premise.

**Q: Why Python instead of C++?**
A: All the heavy inference is already in C++ libraries (llama.cpp, CTranslate2, ONNX Runtime). Python is the orchestration layer, which is <2% of CPU time. C++ rewrite of orchestration would save microseconds while the VLM runs for 400ms. Optional C++ port of the hot path is a Phase 4 stretch.

**Q: Why not LLaVA?**
A: 7B+ params, doesn't fit the 4GB wearable RAM budget. Moondream is purpose-built for edge.

**Q: Why not Whisper-large for better accuracy?**
A: distil-whisper-small.en is 6x faster with negligible accuracy loss for short utterances. Latency budget doesn't allow for large.

**Q: Should I batch inference?**
A: No. Single-user realtime system. Batching adds latency.

**Q: Can I use TypeScript for the backend?**
A: No. Python ML ecosystem is not optional here. TypeScript is for the frontend only.

**Q: Should I add a database?**
A: No, unless implementing the session memory stretch feature, in which case use SQLite. No Postgres, no Redis, nothing networked.

## When asking the user (Nithilan) for clarification

Ask only when truly blocked. Otherwise:
- Make the simplest defensible choice
- Document it in code with a comment explaining the tradeoff
- Surface it in the next response so it can be revised

Never ask permission to:
- Add necessary type hints
- Add docstrings
- Add timing instrumentation (it's required by the thesis)
- Refactor for readability after a feature works

Always ask before:
- Adding a new dependency (especially a heavy one)
- Changing an architectural decision documented above
- Skipping an optimization step from the Phase 3 list
- Introducing any cloud service or external API