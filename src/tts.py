"""TTS wrapper — Piper streaming neural TTS.

Streams raw PCM audio chunks from Piper as the model generates them,
allowing the audio output device to start playing before synthesis is complete.
This is the key latency trick: user hears the first syllable long before
the full response is synthesized.

Phase 1 standalone usage:
    python -m src.tts
  Synthesizes a test sentence, plays it, measures first-chunk latency.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterator, Optional, Any

import numpy as np

from src.config import PIPER_PATH, TTS_SENTENCE_SILENCE


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences on '.', '!', '?' boundaries.

    Keeps punctuation attached to the preceding word so Piper's prosody
    model gets complete sentences rather than fragments.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]

logger = logging.getLogger(__name__)


class TTS:
    """Piper TTS wrapper with streaming audio output."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = model_dir or PIPER_PATH
        self._voice: Any = None
        self._sample_rate: int = 22_050  # updated after load

    def load(self) -> None:
        """Load Piper voice from PIPER_PATH (expects *.onnx + *.onnx.json)."""
        from piper.voice import PiperVoice  # type: ignore[import-untyped]

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {self._model_dir}. "
                "Run scripts/download_models.sh first."
            )

        onnx_files = list(self._model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx voice file found in {self._model_dir}"
            )
        onnx_path = onnx_files[0]
        config_path = Path(str(onnx_path) + ".json")

        t0 = time.perf_counter()
        if config_path.exists():
            self._voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))
        else:
            self._voice = PiperVoice.load(str(onnx_path))

        self._sample_rate = self._voice.config.sample_rate
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Piper TTS loaded from '%s' in %.0fms (sample_rate=%d)",
            onnx_path.name,
            elapsed_ms,
            self._sample_rate,
        )

    def synthesize_streaming(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM int16 chunks, one per sentence.

        piper-tts >= 1.4.x API: voice.synthesize(text) returns
        Iterable[AudioChunk] where AudioChunk.audio_float_array is float32.
        Convert to int16 bytes: (arr * 32767).astype(np.int16).tobytes()

        Streaming is achieved by splitting on sentence boundaries and synthesizing
        each sentence independently — the first sentence is ready to play while
        subsequent ones are still being synthesized.

        Yields:
            Raw PCM bytes (int16, mono, self.sample_rate Hz) per sentence.
        """
        if self._voice is None:
            raise RuntimeError("Call TTS.load() before synthesis.")

        t0 = time.perf_counter()
        first_chunk = True

        for sentence in _split_sentences(text):
            if not sentence.strip():
                continue

            # Collect all AudioChunk objects for this sentence
            pcm_parts: list[bytes] = []
            for audio_chunk in self._voice.synthesize(sentence):
                pcm = (audio_chunk.audio_float_array * 32767).astype(np.int16).tobytes()
                pcm_parts.append(pcm)

            if not pcm_parts:
                continue

            if first_chunk:
                first_chunk_ms = (time.perf_counter() - t0) * 1000.0
                logger.info("TTS first chunk: %.0fms (budget 150ms)", first_chunk_ms)
                first_chunk = False

            yield b"".join(pcm_parts)

    def synthesize_array(self, text: str) -> tuple[np.ndarray, float]:
        """Synthesize *text* and return (audio_array_int16, first_chunk_ms).

        Convenience method for Phase 1 testing and benchmarks.
        """
        chunks: list[bytes] = []
        first_chunk_ms = 0.0
        t0 = time.perf_counter()

        for i, chunk in enumerate(self.synthesize_streaming(text)):
            if i == 0:
                first_chunk_ms = (time.perf_counter() - t0) * 1000.0
            chunks.append(chunk)

        audio = np.frombuffer(b"".join(chunks), dtype=np.int16)
        return audio, first_chunk_ms

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


# ---------------------------------------------------------------------------
# Phase 1 standalone demo
# ---------------------------------------------------------------------------

def _run_phase1_demo() -> None:
    """Synthesize a test sentence, play it, print first-chunk latency."""
    import sounddevice as sd  # type: ignore[import-untyped]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    TEST_SENTENCE = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a latency test for the edge VLM assistant."
    )

    print(f"Loading Piper TTS from {PIPER_PATH}...")
    tts = TTS()
    tts.load()

    print(f"\nSynthesizing: '{TEST_SENTENCE[:60]}...'")
    print("Playing audio...")

    # Stream and play in real-time
    t0 = time.perf_counter()
    first_chunk_ms: Optional[float] = None

    with sd.RawOutputStream(
        samplerate=tts.sample_rate,
        channels=1,
        dtype="int16",
    ) as stream:
        for i, chunk in enumerate(tts.synthesize_streaming(TEST_SENTENCE)):
            if i == 0:
                first_chunk_ms = (time.perf_counter() - t0) * 1000.0
            stream.write(chunk)

    total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nTTS first-chunk latency: {first_chunk_ms:.0f}ms (budget 150ms)")
    print(f"TTS total synthesis time: {total_ms:.0f}ms")
    if first_chunk_ms is not None and first_chunk_ms <= 150:
        print("✓ within first-chunk budget")
    else:
        print("✗ over budget")


if __name__ == "__main__":
    _run_phase1_demo()
