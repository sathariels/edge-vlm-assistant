"""Optimized latency benchmark — Phase 3.

Same synthetic workload as run_baseline.py but runs with all Phase 3
optimizations active:
  - Streaming VLM → concurrent TTS (biggest impact)
  - Pre-allocated capture buffers (per-frame malloc eliminated)
  - uvloop for server asyncio (not measured here, server not started)

After running, loads benchmarks/results/baseline.json and prints
a before/after comparison table.

Usage:
    source ~/glasses-project/venv/bin/activate
    python benchmarks/run_optimized.py [--queries 20] [--warmup 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.asr import ASR
from src.config import (
    TARGET_ASR_MS,
    TARGET_TTS_FIRST_CHUNK_MS,
    TARGET_TOTAL_MS,
    TARGET_VLM_FIRST_TOKEN_MS,
)
from src.metrics import QueryMetrics
from src.tts import TTS
from src.vlm import VLM

# Re-use helpers from baseline script
sys.path.insert(0, str(Path(__file__).parent))
from run_baseline import _make_test_frame, _synthesize_to_float32, _TEST_IMAGE_TEXTS, _TEST_PROMPTS  # noqa: E402


# ---------------------------------------------------------------------------
# Optimized single-query runner
# ---------------------------------------------------------------------------

def run_one_query_optimized(
    query_id: int,
    vlm: VLM,
    tts: TTS,
    asr: ASR,
    prompt: str,
    frame: np.ndarray,
    pre_audio: np.ndarray,
) -> QueryMetrics:
    """Run one query using the Phase 3 streaming pipeline.

    The key difference from run_baseline: VLM.stream() feeds tokens to TTS
    concurrently via the InferencePipeline streaming path. We measure the
    true first-token latency and true TTS first-chunk latency separately.
    """
    import queue
    import threading

    m = QueryMetrics(query_id=query_id)
    m.t_speech_start = time.perf_counter()
    m.t_speech_end = time.perf_counter()

    # Stage 1: ASR (same as baseline)
    m.t_asr_start = time.perf_counter()
    transcript = asr.transcribe(pre_audio)
    m.t_asr_end = time.perf_counter()
    m.transcript = transcript or prompt

    # Stage 2: frame grab
    m.t_frame_grabbed = time.perf_counter()

    # Stage 3+4: streaming VLM → concurrent TTS
    from src.inference import _is_sentence_boundary

    sentence_queue: queue.Queue[None | str] = queue.Queue()
    tts_done = threading.Event()

    def tts_worker() -> None:
        first_chunk = False
        chunks_written = 0
        for sentence in iter(sentence_queue.get, None):
            if not sentence or not sentence.strip():
                continue
            for chunk in tts.synthesize_streaming(sentence):
                if not first_chunk:
                    m.t_tts_first_chunk = time.perf_counter()
                    first_chunk = True
                chunks_written += 1
                # Discard audio in benchmark mode (don't play to speakers)
        if not first_chunk:
            m.t_tts_first_chunk = time.perf_counter()
        m.t_tts_end = time.perf_counter()
        tts_done.set()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    m.t_tts_start = time.perf_counter()

    m.t_vlm_start = time.perf_counter()
    first_token = True
    token_buf = ""
    full_response: list[str] = []

    for token in vlm.stream(frame, m.transcript, max_tokens=150):
        if first_token:
            m.t_vlm_first_token = time.perf_counter()
            first_token = False
        token_buf += token
        full_response.append(token)
        if _is_sentence_boundary(token_buf):
            sentence_queue.put(token_buf.strip())
            token_buf = ""

    if token_buf.strip():
        full_response.append(token_buf)
        sentence_queue.put(token_buf.strip())

    sentence_queue.put(None)  # sentinel
    m.t_vlm_end = time.perf_counter()
    if first_token:
        m.t_vlm_first_token = m.t_vlm_end

    tts_done.wait()
    m.response = "".join(full_response)
    return m


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _print_comparison(
    baseline: dict | None,
    optimized_stages: dict[str, list[float]],
    budgets: dict[str, int],
) -> None:
    print("\n" + "=" * 72)
    print(f"{'Stage':<22} {'BEFORE p50':>10} {'AFTER p50':>10} {'DELTA':>8}  {'budget'}")
    print("-" * 72)

    for stage, vals in optimized_stages.items():
        after_p50 = _percentile(vals, 50)
        before_str = "  n/a"
        delta_str = ""

        if baseline and "summary" in baseline and stage in baseline["summary"]:
            before_p50 = baseline["summary"][stage].get("p50_ms", 0)
            before_str = f"{before_p50:>8.0f}ms"
            delta = after_p50 - before_p50
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:.0f}ms"

        budget = budgets.get(stage, 0)
        hit = "✓" if after_p50 <= budget else "✗"
        print(
            f"{stage:<22} {before_str}   {after_p50:>8.0f}ms  {delta_str:>7}  "
            f"{hit} ({budget}ms)"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 optimized latency benchmark")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=str, default="benchmarks/results/optimized.json")
    parser.add_argument(
        "--baseline",
        type=str,
        default="benchmarks/results/baseline.json",
        help="Path to Phase 2 baseline JSON for comparison",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("edge-vlm-assistant — Phase 3 optimized benchmark")
    print(f"  Queries  : {args.queries} (+ {args.warmup} warmup)")
    print(f"  Output   : {args.output}")
    print("=" * 60)

    baseline_data = _load_baseline(Path(args.baseline))
    if baseline_data:
        print(f"  Baseline : {args.baseline} loaded ({baseline_data['meta']['n_queries']} queries)")
    else:
        print(f"  Baseline : {args.baseline} not found — run run_baseline.py first")
    print()

    # Load models
    print("Loading models...")
    t_load = time.perf_counter()
    vlm = VLM()
    tts = TTS()
    asr = ASR()
    vlm.load()
    tts.load()
    asr.load()
    print(f"  Done in {(time.perf_counter() - t_load)*1000:.0f}ms\n")

    # Pre-generate audio inputs
    print("Pre-generating synthetic audio inputs...")
    pre_audio_clips = [_synthesize_to_float32(tts, p) for p in _TEST_PROMPTS]
    print(f"  {len(pre_audio_clips)} clips ready\n")

    # Warmup
    print(f"Warmup ({args.warmup} queries)...")
    for i in range(args.warmup):
        idx = i % len(_TEST_PROMPTS)
        run_one_query_optimized(
            -1, vlm, tts, asr,
            _TEST_PROMPTS[idx],
            _make_test_frame(_TEST_IMAGE_TEXTS[idx % len(_TEST_IMAGE_TEXTS)]),
            pre_audio_clips[idx],
        )
    print()

    # Measured runs
    print(f"Running {args.queries} measured queries...")
    print(f"{'ID':>4}  {'ASR':>6}  {'VLM 1st tok':>11}  {'TTS 1st':>8}  {'TOTAL':>7}  status")
    print("-" * 58)

    results: list[QueryMetrics] = []
    for i in range(args.queries):
        idx = i % len(_TEST_PROMPTS)
        m = run_one_query_optimized(
            i, vlm, tts, asr,
            _TEST_PROMPTS[idx],
            _make_test_frame(_TEST_IMAGE_TEXTS[idx % len(_TEST_IMAGE_TEXTS)]),
            pre_audio_clips[idx],
        )
        results.append(m)
        status = "✓" if m.total_ms <= TARGET_TOTAL_MS else "✗ OVER"
        print(
            f"{i:>4}  {m.asr_ms:>5.0f}ms  {m.vlm_first_token_ms:>10.0f}ms"
            f"  {m.tts_first_chunk_ms:>7.0f}ms  {m.total_ms:>6.0f}ms  {status}"
        )

    # Stats
    stages: dict[str, list[float]] = {
        "asr_ms": [m.asr_ms for m in results],
        "vlm_first_token_ms": [m.vlm_first_token_ms for m in results],
        "tts_first_chunk_ms": [m.tts_first_chunk_ms for m in results],
        "total_ms": [m.total_ms for m in results],
    }
    budgets = {
        "asr_ms": TARGET_ASR_MS,
        "vlm_first_token_ms": TARGET_VLM_FIRST_TOKEN_MS,
        "tts_first_chunk_ms": TARGET_TTS_FIRST_CHUNK_MS,
        "total_ms": TARGET_TOTAL_MS,
    }

    _print_comparison(baseline_data, stages, budgets)

    total_p50 = _percentile(stages["total_ms"], 50)
    if total_p50 <= TARGET_TOTAL_MS:
        print(f"\n✓ Meets <{TARGET_TOTAL_MS}ms target at p50 ({total_p50:.0f}ms)")
    else:
        print(f"\n✗ Still {total_p50 - TARGET_TOTAL_MS:.0f}ms over budget at p50")
        print("  Check which stage is bottlenecked and apply remaining Phase 3 optimizations.")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "phase": "3-optimized",
            "n_queries": len(results),
            "n_warmup_discarded": args.warmup,
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
    print(f"\nOptimized results saved to {output_path}\n")


if __name__ == "__main__":
    main()
