"""Baseline latency benchmark — Phase 2.

Runs N queries through the full pipeline using synthetic inputs so results
are reproducible without requiring a live microphone session.

Synthetic input strategy:
  1. ASR stage: synthesize a known sentence via Piper, then feed the resulting
     float32 audio to Whisper. This gives real ASR latency on realistic audio.
  2. VLM stage: use a 448×448 test image with text overlay.
  3. TTS stage: synthesize the VLM response, discard audio bytes (don't play).

Results are saved to benchmarks/results/baseline.json.

Usage:
    source ~/glasses-project/venv/bin/activate
    python benchmarks/run_baseline.py [--queries 20] [--warmup 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.asr import ASR
from src.config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    SAMPLE_RATE,
    TARGET_ASR_MS,
    TARGET_TTS_FIRST_CHUNK_MS,
    TARGET_TOTAL_MS,
    TARGET_VLM_FIRST_TOKEN_MS,
)
from src.metrics import QueryMetrics
from src.tts import TTS
from src.vlm import VLM


# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------

_TEST_PROMPTS = [
    "What do you see in front of me?",
    "Can you describe what's in the image?",
    "What objects are visible here?",
    "Tell me what you observe.",
    "What is happening in this scene?",
]

_TEST_IMAGE_TEXTS = [
    "Laptop on a desk",
    "Coffee mug",
    "Bookshelf",
    "Window with curtains",
    "Notebook and pen",
]


def _make_test_frame(label: str = "Test scene") -> np.ndarray:
    """Return a 448×448 BGR frame with a text label, matching capture resolution."""
    import cv2  # type: ignore[import-untyped]

    frame = np.full((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), 40, dtype=np.uint8)
    # White rectangle to give the VLM something to look at
    cv2.rectangle(frame, (40, 40), (CAPTURE_WIDTH - 40, CAPTURE_HEIGHT - 40), (200, 200, 200), 2)
    cv2.putText(frame, label, (60, CAPTURE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def _synthesize_to_float32(tts: TTS, text: str) -> np.ndarray:
    """TTS → raw int16 bytes → float32 array at 16kHz for Whisper.

    Piper outputs at ~22050Hz; we resample to 16kHz before feeding to Whisper.
    """
    # Collect all PCM chunks from TTS
    chunks: list[bytes] = []
    for chunk in tts.synthesize_streaming(text):
        chunks.append(chunk)
    raw = b"".join(chunks)

    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # Resample from Piper's native rate to 16kHz if needed
    piper_rate = tts.sample_rate
    if piper_rate != SAMPLE_RATE:
        try:
            import resampy  # type: ignore[import-untyped]
            audio_f32 = resampy.resample(audio_f32, piper_rate, SAMPLE_RATE)
        except ImportError:
            # Fallback: linear interpolation via numpy (lower quality but no dep)
            n_out = int(len(audio_f32) * SAMPLE_RATE / piper_rate)
            audio_f32 = np.interp(
                np.linspace(0, len(audio_f32) - 1, n_out),
                np.arange(len(audio_f32)),
                audio_f32,
            ).astype(np.float32)

    return audio_f32


# ---------------------------------------------------------------------------
# Single query runner
# ---------------------------------------------------------------------------

def run_one_query(
    query_id: int,
    vlm: VLM,
    tts: TTS,
    asr: ASR,
    prompt: str,
    frame: np.ndarray,
    pre_audio: np.ndarray,
    discard_audio: bool = True,
) -> QueryMetrics:
    """Run one full pipeline query with synthetic inputs.

    Args:
        pre_audio:      float32 16kHz audio (generated from TTS in setup phase).
        discard_audio:  If True, TTS output is discarded rather than played.
                        Set False only in interactive mode.
    """
    m = QueryMetrics(query_id=query_id)

    # Simulated speech timestamps (VAD overhead is negligible in benchmarks)
    m.t_speech_start = time.perf_counter()
    m.t_speech_end = time.perf_counter()

    # Stage 1: ASR
    m.t_asr_start = time.perf_counter()
    transcript = asr.transcribe(pre_audio)
    m.t_asr_end = time.perf_counter()
    m.transcript = transcript or prompt  # use prompt as fallback if silence

    # Stage 2: frame grab (instant in benchmark — no ring buffer needed)
    m.t_frame_grabbed = time.perf_counter()

    # Stage 3: VLM
    m.t_vlm_start = time.perf_counter()
    response, _ = vlm.generate(frame, m.transcript, max_tokens=150)
    m.t_vlm_first_token = time.perf_counter()  # Phase 2: == end time
    m.t_vlm_end = m.t_vlm_first_token
    m.response = response

    # Stage 4: TTS
    m.t_tts_start = time.perf_counter()
    first_tts_chunk = True
    if response.strip():
        for chunk in tts.synthesize_streaming(response):
            if first_tts_chunk:
                m.t_tts_first_chunk = time.perf_counter()
                first_tts_chunk = False
            if not discard_audio:
                pass  # TODO: play via sounddevice in interactive mode
    if first_tts_chunk:
        m.t_tts_first_chunk = time.perf_counter()
    m.t_tts_end = time.perf_counter()

    return m


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 baseline latency benchmark")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to run (default: 20)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup queries to discard (default: 2)")
    parser.add_argument("--output", type=str, default="benchmarks/results/baseline.json")
    args = parser.parse_args()

    print("=" * 60)
    print("edge-vlm-assistant — Phase 2 baseline benchmark")
    print(f"  Queries  : {args.queries} (+ {args.warmup} warmup discarded)")
    print(f"  Output   : {args.output}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\nLoading models...")
    t_load = time.perf_counter()

    vlm = VLM()
    tts = TTS()
    asr = ASR()

    vlm.load()
    tts.load()
    asr.load()

    print(f"  Models loaded in {(time.perf_counter() - t_load)*1000:.0f}ms\n")

    # ------------------------------------------------------------------
    # Pre-generate synthetic audio inputs (done once, not counted in latency)
    # ------------------------------------------------------------------
    print("Pre-generating synthetic audio inputs...")
    pre_audio_clips: list[np.ndarray] = []
    for prompt in _TEST_PROMPTS:
        audio = _synthesize_to_float32(tts, prompt)
        pre_audio_clips.append(audio)
    print(f"  {len(pre_audio_clips)} audio clips ready\n")

    # ------------------------------------------------------------------
    # Warmup (not counted)
    # ------------------------------------------------------------------
    print(f"Running {args.warmup} warmup queries (discarded)...")
    for i in range(args.warmup):
        idx = i % len(_TEST_PROMPTS)
        run_one_query(
            query_id=-1,
            vlm=vlm,
            tts=tts,
            asr=asr,
            prompt=_TEST_PROMPTS[idx],
            frame=_make_test_frame(_TEST_IMAGE_TEXTS[idx % len(_TEST_IMAGE_TEXTS)]),
            pre_audio=pre_audio_clips[idx],
        )
        print(f"  warmup {i+1}/{args.warmup} done")
    print()

    # ------------------------------------------------------------------
    # Measured runs
    # ------------------------------------------------------------------
    print(f"Running {args.queries} measured queries...")
    print(f"{'ID':>4}  {'ASR':>6}  {'VLM':>6}  {'TTS':>6}  {'TOTAL':>7}  {'status'}")
    print("-" * 52)

    results: list[QueryMetrics] = []
    for i in range(args.queries):
        idx = i % len(_TEST_PROMPTS)
        m = run_one_query(
            query_id=i,
            vlm=vlm,
            tts=tts,
            asr=asr,
            prompt=_TEST_PROMPTS[idx],
            frame=_make_test_frame(_TEST_IMAGE_TEXTS[idx % len(_TEST_IMAGE_TEXTS)]),
            pre_audio=pre_audio_clips[idx],
        )
        results.append(m)

        status = "✓" if m.total_ms <= TARGET_TOTAL_MS else "✗ OVER"
        print(
            f"{i:>4}  {m.asr_ms:>5.0f}ms  {m.vlm_first_token_ms:>5.0f}ms"
            f"  {m.tts_first_chunk_ms:>5.0f}ms  {m.total_ms:>6.0f}ms  {status}"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _percentile(vals: list[float], p: float) -> float:
        s = sorted(vals)
        idx = min(int(len(s) * p / 100), len(s) - 1)
        return s[idx]

    stages: dict[str, list[float]] = {
        "asr_ms": [m.asr_ms for m in results],
        "vlm_first_token_ms": [m.vlm_first_token_ms for m in results],
        "tts_first_chunk_ms": [m.tts_first_chunk_ms for m in results],
        "total_ms": [m.total_ms for m in results],
    }

    print("\n" + "=" * 60)
    print("Results summary")
    print(f"{'Stage':<22} {'p50':>7} {'p95':>7} {'min':>7} {'max':>7}  {'budget'}")
    print("-" * 60)
    budgets = {
        "asr_ms": TARGET_ASR_MS,
        "vlm_first_token_ms": TARGET_VLM_FIRST_TOKEN_MS,
        "tts_first_chunk_ms": TARGET_TTS_FIRST_CHUNK_MS,
        "total_ms": TARGET_TOTAL_MS,
    }
    for stage, vals in stages.items():
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        budget = budgets.get(stage, 0)
        hit = "✓" if p50 <= budget else "✗"
        print(
            f"{stage:<22} {p50:>6.0f}ms {p95:>6.0f}ms "
            f"{min(vals):>6.0f}ms {max(vals):>6.0f}ms  "
            f"{hit} ({budget}ms)"
        )
    print("=" * 60)

    total_p50 = _percentile(stages["total_ms"], 50)
    if total_p50 <= TARGET_TOTAL_MS:
        print(f"\n✓ Meets <{TARGET_TOTAL_MS}ms target at p50 ({total_p50:.0f}ms)")
    else:
        gap = total_p50 - TARGET_TOTAL_MS
        print(f"\n✗ Over budget by {gap:.0f}ms at p50 — Phase 3 optimizations needed")

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "phase": "2-baseline",
            "n_queries": len(results),
            "n_warmup_discarded": args.warmup,
            "description": (
                "Unoptimized Phase 2 baseline. "
                "VLM first-token == total VLM time (streaming not yet implemented). "
                "Beat these numbers in Phase 3."
            ),
        },
        "summary": {
            stage: {
                "p50_ms": round(_percentile(vals, 50), 1),
                "p95_ms": round(_percentile(vals, 95), 1),
                "min_ms": round(min(vals), 1),
                "max_ms": round(max(vals), 1),
                "budget_ms": budgets.get(stage, 0),
                "meets_budget": _percentile(vals, 50) <= budgets.get(stage, float("inf")),
            }
            for stage, vals in stages.items()
        },
        "queries": [m.to_dict() for m in results],
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nBaseline saved to {output_path}")
    print("Run benchmarks/run_optimized.py after Phase 3 to compare.\n")


if __name__ == "__main__":
    main()
