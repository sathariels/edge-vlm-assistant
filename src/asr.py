"""ASR wrapper — faster-whisper with distil-whisper-small.en.

Why distil-whisper-small.en:
  6x faster than whisper-large with negligible accuracy loss for
  short utterances. Fits the 150ms ASR budget at our target utterance
  length (~3s). See CLAUDE.md for full rationale.

Phase 1 standalone usage:
    python -m src.asr
  Records a 5-second clip and transcribes it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_PATH,
)

logger = logging.getLogger(__name__)


class ASR:
    """faster-whisper transcription wrapper."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        # Prefer local path; fall back to HuggingFace model name for first-run download.
        self._model_path: str = (
            str(model_path or WHISPER_PATH)
            if (model_path or WHISPER_PATH).exists()
            else "distil-whisper/distil-small.en"
        )
        self._model = None  # type: ignore[assignment]

    def load(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._model_path,
            device="cpu",       # CTranslate2 CPU backend; MPS not yet supported
            compute_type=WHISPER_COMPUTE_TYPE,
            num_workers=1,      # single-user realtime — no benefit from multiple workers
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Whisper loaded from '%s' in %.0fms", self._model_path, elapsed_ms)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe *audio* (float32, 16kHz, mono) and return the text.

        Timing is logged at INFO level for latency tracking.
        """
        if self._model is None:
            raise RuntimeError("Call ASR.load() before transcribing.")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            beam_size=WHISPER_BEAM_SIZE,
            language="en",
            vad_filter=False,   # we already gate on Silero VAD; avoid double VAD overhead
            word_timestamps=False,
        )
        # Consuming the generator forces completion
        text = " ".join(s.text for s in segments).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "ASR: %.0fms | audio_len=%.2fs | '%s'",
            elapsed_ms,
            len(audio) / SAMPLE_RATE,
            text[:80],
        )
        return text


# ---------------------------------------------------------------------------
# Phase 1 standalone demo
# ---------------------------------------------------------------------------

def _run_phase1_demo() -> None:
    """Record N seconds of audio, then transcribe and print result + latency."""
    import sounddevice as sd  # type: ignore[import-untyped]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    RECORD_SECONDS = 5

    print("Loading Whisper...")
    asr = ASR()
    asr.load()

    print(f"\nRecording for {RECORD_SECONDS}s... speak now!")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio_mono = audio[:, 0]
    print("Recording done. Transcribing...")

    t0 = time.perf_counter()
    text = asr.transcribe(audio_mono)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nTranscript : '{text}'")
    print(f"ASR latency: {elapsed_ms:.0f}ms")


if __name__ == "__main__":
    _run_phase1_demo()
