"""Sanity check — verify every model loads and prints ✓ for each.

Run this before starting development on a new machine or after
updating models to confirm nothing is broken.

Usage:
    source ~/glasses-project/venv/bin/activate
    python scripts/sanity_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

# Ensure src/ is importable when run as a top-level script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MARIAN_ES_EN_PATH,
    MODELS_DIR,
    MOONDREAM_PATH,
    PIPER_PATH,
    WHISPER_PATH,
)


def _check(label: str, fn: Callable[[], None]) -> bool:
    """Run *fn*, print ✓/✗, return True on success."""
    t0 = time.perf_counter()
    try:
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  ✓  {label} ({elapsed_ms:.0f}ms)")
        return True
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  ✗  {label} ({elapsed_ms:.0f}ms) — {type(exc).__name__}: {exc}")
        return False


def check_model_dirs() -> bool:
    print("\n[1/5] Model directories")
    ok = True
    for path, name in [
        (MOONDREAM_PATH, "Moondream 3 MLX"),
        (WHISPER_PATH, "distil-whisper-small.en"),
        (PIPER_PATH, "Piper voice (lessac-medium)"),
        (MARIAN_ES_EN_PATH, "MarianMT es→en"),
    ]:
        exists = path.exists()
        status = "✓" if exists else "✗"
        suffix = "" if exists else f" — missing: {path}"
        print(f"  {status}  {name}{suffix}")
        if not exists:
            ok = False
    return ok


def check_silero_vad() -> bool:
    print("\n[2/5] Silero VAD")

    def _load() -> None:
        from src.vad import VAD
        vad = VAD()
        vad.load()
        import numpy as np
        dummy = np.zeros(512, dtype=np.float32)
        prob = vad.speech_prob(dummy)
        assert 0.0 <= prob <= 1.0, f"Unexpected probability: {prob}"

    return _check("Silero VAD (auto-download from HF cache)", _load)


def check_whisper() -> bool:
    print("\n[3/5] ASR — faster-whisper / distil-whisper-small.en")

    def _load() -> None:
        from src.asr import ASR
        asr = ASR()
        asr.load()
        import numpy as np
        dummy = np.zeros(16_000, dtype=np.float32)  # 1 second of silence
        text = asr.transcribe(dummy)
        # Silence produces empty or near-empty transcript — that's fine
        assert isinstance(text, str)

    return _check("faster-whisper (distil-small.en)", _load)


def check_moondream() -> bool:
    print("\n[4/5] VLM — Moondream 3 (MLX)")

    def _load() -> None:
        from src.vlm import VLM
        vlm = VLM()
        vlm.load()
        import numpy as np
        dummy_frame = np.zeros((448, 448, 3), dtype=np.uint8)
        response, latency_ms = vlm.generate(dummy_frame, "What is in this image?", max_tokens=20)
        assert isinstance(response, str)
        print(f"       first inference: {latency_ms:.0f}ms")

    return _check("Moondream 3 MLX inference", _load)


def check_piper() -> bool:
    print("\n[5/5] TTS — Piper (lessac-medium)")

    def _load() -> None:
        from src.tts import TTS
        tts = TTS()
        tts.load()
        audio, first_chunk_ms = tts.synthesize_array("Hello.")
        import numpy as np
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        print(f"       first-chunk: {first_chunk_ms:.0f}ms, {len(audio)} samples")

    return _check("Piper TTS (lessac-medium) streaming synthesis", _load)


def main() -> None:
    print("=" * 60)
    print("edge-vlm-assistant — sanity check")
    print(f"Models dir: {MODELS_DIR}")
    print("=" * 60)

    results = [
        check_model_dirs(),
        check_silero_vad(),
        check_whisper(),
        check_moondream(),
        check_piper(),
    ]

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    if all(results):
        print(f"All {total} checks passed. System ready.")
        sys.exit(0)
    else:
        print(f"{passed}/{total} checks passed. Fix failures above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
