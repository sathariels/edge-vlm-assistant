"""Silero VAD wrapper.

Provides a thin, typed wrapper around the silero-vad package.
Expects float32 mono audio at 16kHz in 512-sample chunks (= 32ms).

The model is loaded lazily via load() and cached.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VAD:
    """Silero Voice Activity Detector."""

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None

    def load(self) -> None:
        """Download (if needed) and load the Silero VAD model."""
        t0 = time.perf_counter()
        # silero-vad auto-downloads to ~/.cache/torch/hub/
        from silero_vad import load_silero_vad  # type: ignore[import-untyped]

        self._model = load_silero_vad()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Silero VAD loaded in %.0fms", elapsed_ms)

    def speech_prob(self, chunk: np.ndarray, sample_rate: int = 16_000) -> float:
        """Return speech probability in [0, 1] for a single audio chunk.

        Args:
            chunk: float32 array of shape (512,) at 16kHz.
            sample_rate: Must match the rate used to record *chunk*.

        Returns:
            Probability that the chunk contains speech.
        """
        if self._model is None:
            raise RuntimeError("Call VAD.load() before inference.")
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        tensor = torch.from_numpy(chunk)
        with torch.no_grad():
            prob: float = self._model(tensor, sample_rate).item()
        return prob
