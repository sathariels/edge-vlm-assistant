"""Latency instrumentation for edge-vlm-assistant.

Every stage of the pipeline is timed with wall-clock perf_counter.
Waterfall data is accumulated per-query and can be serialised to JSON
for the frontend dashboard or benchmark results.

Phase 1: timer context manager + QueryMetrics dataclass.
Phase 2: add the WebSocket emission hook.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def timer(name: str, warn_above_ms: Optional[float] = None) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for *name*.

    Args:
        name: Human-readable label for the stage.
        warn_above_ms: If set and elapsed > this value, log at WARNING level.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        msg = f"[timer] {name}: {elapsed_ms:.1f}ms"
        if warn_above_ms is not None and elapsed_ms > warn_above_ms:
            logger.warning(msg + f" (budget {warn_above_ms:.0f}ms)")
        else:
            logger.info(msg)


@dataclass
class QueryMetrics:
    """Waterfall timestamps for a single voice→vision→speech query.

    All times are wall-clock seconds from time.perf_counter().
    Durations are computed lazily as properties.
    """

    query_id: int = 0

    # Stage timestamps (set by each pipeline stage)
    t_speech_start: float = 0.0      # VAD detects onset of speech
    t_speech_end: float = 0.0        # VAD detects end of speech
    t_asr_start: float = 0.0
    t_asr_end: float = 0.0
    t_frame_grabbed: float = 0.0     # frame pulled from ring buffer
    t_vlm_start: float = 0.0
    t_vlm_first_token: float = 0.0
    t_vlm_end: float = 0.0
    t_tts_start: float = 0.0
    t_tts_first_chunk: float = 0.0
    t_tts_end: float = 0.0

    transcript: str = ""
    response: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived durations (ms)
    # ------------------------------------------------------------------

    @property
    def vad_ms(self) -> float:
        return (self.t_speech_end - self.t_speech_start) * 1000.0

    @property
    def asr_ms(self) -> float:
        return (self.t_asr_end - self.t_asr_start) * 1000.0

    @property
    def vlm_first_token_ms(self) -> float:
        return (self.t_vlm_first_token - self.t_vlm_start) * 1000.0

    @property
    def tts_first_chunk_ms(self) -> float:
        return (self.t_tts_first_chunk - self.t_tts_start) * 1000.0

    @property
    def total_ms(self) -> float:
        """End-of-speech to first audio byte — the headline number."""
        return (self.t_tts_first_chunk - self.t_speech_end) * 1000.0

    def to_dict(self) -> dict[str, object]:
        # Waterfall offsets: ms elapsed from t_speech_end (the "pipeline zero point").
        # These let the frontend draw each stage at its true start position,
        # revealing Phase 3 VLM+TTS concurrency visually.
        base = self.t_speech_end
        def _off(t: float) -> float:
            return round((t - base) * 1000.0, 1) if base > 0 and t > 0 else 0.0

        return {
            "query_id": self.query_id,
            "transcript": self.transcript,
            "response": self.response,
            "stages_ms": {
                "vad": round(self.vad_ms, 1),
                "asr": round(self.asr_ms, 1),
                "vlm_first_token": round(self.vlm_first_token_ms, 1),
                "tts_first_chunk": round(self.tts_first_chunk_ms, 1),
                "total": round(self.total_ms, 1),
            },
            # Absolute start offsets from end-of-speech (for waterfall rendering)
            "waterfall": {
                "asr_start_ms": _off(self.t_asr_start),
                "vlm_start_ms": _off(self.t_vlm_start),
                "tts_start_ms": _off(self.t_tts_start),
            },
            **{f"extra_{k}": round(v, 1) for k, v in self.extra.items()},
        }

    def log_summary(self) -> None:
        logger.info(
            "[query %d] total=%.0fms | vad=%.0f asr=%.0f vlm_tok=%.0f tts=%.0f | '%s'",
            self.query_id,
            self.total_ms,
            self.vad_ms,
            self.asr_ms,
            self.vlm_first_token_ms,
            self.tts_first_chunk_ms,
            self.transcript[:60],
        )
