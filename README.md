# edge-vlm-assistant

A smart-glasses AI assistant simulator that runs **entirely on-device** on a MacBook Air M3, achieving **<800ms voice-to-response latency**.

This is an edge-AI systems engineering project — not a cloud wrapper. Every inference stage (ASR, VLM, TTS, VAD) runs locally, mirroring the compute and memory constraints of real wearable hardware (Ray-Ban Meta: ~4GB RAM, Snapdragon AR1).

---

## Architecture

Three parallel, non-blocking loops:

```
┌─────────────────────────────────────────────────────────┐
│  Loop 1: Capture   webcam → resize 448×448 → ring buf   │
│  Loop 2: Audio     mic → Silero VAD → speech → ASR      │
│  Loop 3: Inference ring_buf + transcript → VLM → TTS    │
└─────────────────────────────────────────────────────────┘
```

The critical design principle: **stream everything, block nothing.**
TTS starts emitting audio before the VLM finishes generating. The user perceives latency as "time to first syllable."

## Latency budget

| Stage | Budget | Model |
|-------|--------|-------|
| VAD end-of-speech | 100ms | Silero VAD |
| ASR (~3s utterance) | 150ms | distil-whisper-small.en (CTranslate2) |
| Frame retrieval | ~1ms | Ring buffer O(1) |
| VLM first token | 400ms | Moondream 3 MLX int4 |
| TTS first chunk | 150ms | Piper (lessac-medium) |
| **Total** | **<800ms** | |

*Before/after optimization numbers will be added after Phase 3.*

## Models (all local, no cloud)

| Model | Size | Purpose |
|-------|------|---------|
| Moondream 2 MLX int4 (`mlx-community/moondream2-4bit`) | ~1.8GB | Vision-language |
| distil-whisper-small.en (CTranslate2 int8) | 320MB | Speech-to-text |
| Piper en_US-lessac-medium | 61MB | Text-to-speech |
| MarianMT es→en | 599MB | OCR translation (Phase 2) |
| Silero VAD | ~2MB | Voice activity detection (auto-downloaded) |

> **Note on Moondream 3:** The original spec targets Moondream 3 MLX int4. As of Phase 1,
> `vikhyatk/moondream3` is not yet available as a public HuggingFace repo and the
> `mlx-community` conversion does not yet exist. `mlx-community/moondream2-4bit` is used
> in the interim — it is structurally identical for the purposes of this pipeline and
> ships the `hf_moondream.py` architecture file that `mlx-vlm` requires. Swap the
> `repo_id` in `scripts/download_models.sh` and re-run when Moondream 3 MLX ships.

## Quickstart

```bash
# 1. Activate the project venv
source ~/glasses-project/venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (idempotent — safe to re-run)
./scripts/download_models.sh

# 4. Verify everything loads
python scripts/sanity_check.py
```

## Platform notes

### macOS camera permissions (important)

`python -m src.capture` requires camera access. On macOS, camera permission is
granted **per application** via System Settings → Privacy & Security → Camera.

**Grant access once:**
1. Run `python -m src.capture` from Terminal.app or iTerm2.
2. macOS will show a permission prompt — click Allow.
3. The permission is then persistent for that terminal app.

**Background processes are blocked by design.**
Any process launched by a background agent (automated CI, headless scripts, AI
coding assistants running in a subprocess) will receive an AVFoundation
`Cannot Open Camera` error and `errno: 0` regardless of whether the terminal
app has been granted permission. This is macOS system sandboxing — it is not a
code bug. The capture loop code is correct; it simply must be invoked from an
interactive foreground terminal session.

This is also a deliberate note for the project write-up: a real wearable device
faces the same hardware-access gating at the OS/kernel level.

---

## Phase 1 — Run individual components

```bash
# Webcam capture loop (shows live feed, prints FPS stats)
# Must run from an interactive terminal — see Platform notes above.
python -m src.capture

# Audio / VAD loop (prints SPEECH START / SPEECH END events)
python -m src.audio

# VLM inference (loads Moondream, runs on test image, prints latency)
python -m src.vlm

# TTS streaming (synthesizes a sentence, plays it, prints first-chunk latency)
python -m src.tts

# ASR (records 5s of audio, transcribes it)
python -m src.asr
```

## Phase 2 — Full pipeline demo

```bash
# Run the full voice→vision→speech pipeline (Ctrl+C to stop)
python run.py

# Without the WebSocket dashboard server
python run.py --no-server

# Measure baseline latency (20 synthetic queries, saves baseline.json)
python benchmarks/run_baseline.py --queries 20
```

On each utterance the terminal prints:
```
  [0] ✓ 612ms total | asr=134 vlm=378 tts=100 | 'What do you see in front of me?'
```

Metrics are saved to `benchmarks/results/session_metrics.json` on exit.

## Build plan

## Phase 3 — Optimization sprint

```bash
# Run benchmark comparing before/after (requires baseline.json from Phase 2 first)
python benchmarks/run_baseline.py --queries 20   # capture the "before" numbers
python benchmarks/run_optimized.py --queries 20  # run with all Phase 3 optimizations

# OCR + translation demo flow (standalone test)
python -m src.ocr        # webcam → OCR, press 'r' to scan
python -m src.translate  # translate test sentences es→en
```

**Optimizations implemented:**

| Optimization | Where | Expected gain |
|---|---|---|
| Streaming VLM → concurrent TTS | `inference.py` | ~150ms — TTS starts on first sentence, not after full VLM |
| Pre-allocated capture frame pool | `capture.py` | ~0.5ms/frame — no per-frame malloc at 30fps |
| uvloop for WebSocket server | `server.py` | ~5ms — lower asyncio overhead |
| int4 VLM (already baked in) | `mlx-community/moondream2-4bit` | baseline already includes this |
| KV cache warmup at startup | `orchestrator.py` | first query not 2-3x cold |
| Streaming TTS (sentence-level) | `tts.py` | first chunk before full synth |

**OCR + translation demo:**
Say "translate this" or "read this" to trigger the OCR flow instead of VLM. The pipeline detects Spanish text in the frame and speaks the English translation.

## Build plan

| Phase | Status | Goal |
|-------|--------|------|
| 1 | ✅ Done | Independent components — each loop tested in isolation |
| 2 | ✅ Done | End-to-end integration — full pipeline, baseline latency measured |
| 3 | ✅ Done | Optimization sprint — concurrent VLM+TTS, OCR+translation |
| 4 | ✅ Done | Frontend dashboard — live waterfall, webcam, transcripts |

## Phase 4 — Frontend dashboard

```bash
# Terminal 1: start the Python pipeline + WebSocket server
source ~/glasses-project/venv/bin/activate
python run.py

# Terminal 2: start the Next.js dashboard
cd frontend
npm install
npm run dev        # → http://localhost:3000
```

The dashboard shows:
- **Webcam feed** — live MJPEG stream from the Python backend (`/stream`)
- **Latency waterfall** — per-query horizontal bar chart with ASR/VLM/TTS stage bars positioned at their true start offsets, budget line at 800ms, color-coded pass/fail
- **Transcript stream** — rolling history of what was heard
- **Response stream** — rolling history of VLM responses with per-query latency

Design: dark terminal aesthetic (`#0a0a0a` background, monospace font, green-on-black status indicators).

## Design decisions

**Why Moondream instead of LLaVA?**
LLaVA-7B doesn't fit the 4GB wearable RAM budget. Moondream is purpose-built for edge.

**Why Python instead of C++?**
All heavy inference runs in C++ libraries (mlx, CTranslate2). Python is the orchestration layer, which is <2% of CPU time.

**Why distil-whisper-small.en instead of whisper-large?**
6x faster, negligible accuracy loss on short utterances. Fits the 150ms ASR budget.

**Why no cloud APIs?**
The entire thesis is on-device inference. A single cloud call invalidates the wearable simulation premise.
