"""Orchestrator — wires the three loops together.

Manages the full lifecycle:
  1. Load all models (sequential in Phase 2; can be parallelised in Phase 3)
  2. KV-cache warmup: one dummy VLM inference so the first real query isn't cold
  3. Start CaptureLoop + AudioLoop as daemon threads
  4. Route each utterance through InferencePipeline
  5. Accumulate QueryMetrics; save to file on shutdown
  6. Handle SIGINT/SIGTERM gracefully

Usage (CLI):
    python -m src.orchestrator

Usage (programmatic):
    orch = Orchestrator()
    orch.load()
    orch.start()
    orch.run()        # blocks until Ctrl+C
    orch.stop()
    orch.save_metrics("path/to/results.json")
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.asr import ASR
from src.audio import AudioLoop
from src.capture import CaptureLoop
from src.config import CAPTURE_HEIGHT, CAPTURE_WIDTH, RING_BUFFER_SIZE
from src.inference import InferencePipeline
from src.metrics import QueryMetrics
from src.ocr import OCR
from src.ring_buffer import RingBuffer
from src.translate import Translator
from src.tts import TTS
from src.vad import VAD
from src.vlm import VLM

logger = logging.getLogger(__name__)


class Orchestrator:
    """Wires the three loops and manages the pipeline lifecycle."""

    def __init__(self, enable_ocr: bool = True) -> None:
        # Models
        self._vad = VAD()
        self._asr = ASR()
        self._vlm = VLM()
        self._tts = TTS()
        # OCR + translation are optional; loaded lazily and skipped if deps missing
        self._ocr: Optional[OCR] = OCR() if enable_ocr else None
        self._translator: Optional[Translator] = Translator() if enable_ocr else None

        # Shared ring buffer — written by CaptureLoop, read by InferencePipeline
        self._ring_buffer: RingBuffer[np.ndarray] = RingBuffer(RING_BUFFER_SIZE)

        # Loop objects (created after load())
        self._capture_loop: Optional[CaptureLoop] = None
        self._audio_loop: Optional[AudioLoop] = None
        self._pipeline: Optional[InferencePipeline] = None

        # sounddevice InputStream handle (held open for the lifetime of the session)
        self._sd_stream: Optional[Any] = None

        # Callbacks for metrics
        self._metrics_callbacks: list[Callable[[QueryMetrics], None]] = []

        # Metrics accumulation
        self._metrics: list[QueryMetrics] = []
        self._metrics_lock = threading.Lock()

        # Lifecycle
        self._running = threading.Event()
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all models. Must be called before start()."""
        logger.info("Loading models...")
        t0 = time.perf_counter()

        self._vad.load()
        self._asr.load()
        self._vlm.load()
        self._tts.load()

        # OCR + translation: load if enabled, but skip gracefully if deps missing
        if self._ocr is not None:
            try:
                self._ocr.load()
                assert self._translator is not None
                self._translator.load()
                logger.info("OCR + translation loaded")
            except (ImportError, FileNotFoundError) as exc:
                logger.warning("OCR/translation disabled: %s", exc)
                self._ocr = None
                self._translator = None

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("All models loaded in %.0fms", total_ms)

        self._warmup_vlm()

        self._capture_loop = CaptureLoop(ring_buffer=self._ring_buffer)
        self._audio_loop = AudioLoop(
            vad=self._vad,
            on_utterance=self._on_utterance,
        )
        self._pipeline = InferencePipeline(
            vlm=self._vlm,
            tts=self._tts,
            asr=self._asr,
            ring_buffer=self._ring_buffer,
            on_metrics=self._on_query_complete,
            ocr=self._ocr,
            translator=self._translator,
        )
        self._loaded = True

    def _warmup_vlm(self) -> None:
        """One dummy inference to prime the KV cache and Metal shader compilation.

        Without warmup the first real query is 2-3x slower than subsequent ones.
        This is logged but not counted in latency metrics.
        """
        logger.info("VLM warmup inference (not counted in metrics)...")
        dummy = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        self._vlm.generate(dummy, "warmup", max_tokens=5)
        logger.info("VLM warmup done in %.0fms", (time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start capture and audio loops (non-blocking)."""
        if not self._loaded:
            raise RuntimeError("Call load() before start()")

        assert self._capture_loop is not None
        assert self._audio_loop is not None

        import sounddevice as sd  # type: ignore[import-untyped]

        self._capture_loop.start()
        self._audio_loop.start()

        # Open the microphone InputStream and hold it for the session lifetime.
        # The audio_callback is wired directly to AudioLoop.
        from src.config import CHANNELS, SAMPLE_RATE, VAD_CHUNK_SAMPLES

        self._sd_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=self._audio_loop.audio_callback,
        )
        self._sd_stream.start()
        self._running.set()
        logger.info("Orchestrator started — listening for speech")

    def run(self) -> None:
        """Block until SIGINT or stop() is called. Call after start()."""
        # Register signal handlers so Ctrl+C triggers a clean shutdown
        def _sig_handler(signum: int, frame: object) -> None:
            logger.info("Shutdown signal received")
            self._running.clear()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        print("\nListening — speak into the mic. Press Ctrl+C to stop.\n")
        self._running.wait()  # blocks here

    def stop(self) -> None:
        """Stop all loops and close the audio stream."""
        self._running.clear()

        if self._sd_stream is not None:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None

        if self._audio_loop is not None:
            self._audio_loop.stop()

        if self._capture_loop is not None:
            self._capture_loop.stop()

        logger.info(
            "Orchestrator stopped. %d queries processed.", len(self._metrics)
        )

    def add_metrics_callback(self, cb: Callable[[QueryMetrics], None]) -> None:
        """Register a callback to be fired when a query completes."""
        self._metrics_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def save_metrics(self, path: Path) -> None:
        """Save all accumulated QueryMetrics to a JSON file."""
        with self._metrics_lock:
            records = [m.to_dict() for m in self._metrics]

        if not records:
            logger.warning("No metrics to save")
            return

        totals = [r["stages_ms"]["total"] for r in records]  # type: ignore[index]
        totals_sorted = sorted(totals)
        n = len(totals_sorted)

        summary = {
            "n_queries": n,
            "p50_ms": totals_sorted[n // 2],
            "p95_ms": totals_sorted[int(n * 0.95)],
            "p99_ms": totals_sorted[min(int(n * 0.99), n - 1)],
            "min_ms": totals_sorted[0],
            "max_ms": totals_sorted[-1],
        }

        output = {
            "meta": {
                "phase": "2-baseline",
                "description": "Unoptimized end-to-end latency. Beat these in Phase 3.",
            },
            "summary": summary,
            "queries": records,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2))
        logger.info("Metrics saved to %s (%d queries, p50=%.0fms)", path, n, summary["p50_ms"])

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_utterance(
        self, audio: np.ndarray, t_speech_start: float, t_speech_end: float
    ) -> None:
        """Called by AudioLoop dispatch thread on each completed utterance."""
        assert self._pipeline is not None
        self._pipeline.process_utterance(audio, t_speech_start, t_speech_end)

    def _on_query_complete(self, m: QueryMetrics) -> None:
        with self._metrics_lock:
            self._metrics.append(m)
        # Print a one-line summary to stdout for the CLI demo
        over = "✗ OVER BUDGET" if m.total_ms > 800 else "✓"
        print(
            f"  [{m.query_id}] {over} {m.total_ms:.0f}ms total | "
            f"asr={m.asr_ms:.0f} vlm={m.vlm_first_token_ms:.0f} tts={m.tts_first_chunk_ms:.0f} | "
            f"'{m.transcript[:50]}'"
        )
        for cb in self._metrics_callbacks:
            cb(m)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()

    from src.config import TARGET_TOTAL_MS

    print("=" * 60)
    print("edge-vlm-assistant — Phase 2 integration demo")
    print(f"Target: <{TARGET_TOTAL_MS}ms voice-to-response")
    print("=" * 60)

    orch = Orchestrator()

    try:
        orch.load()
        orch.start()
        orch.run()
    finally:
        orch.stop()

        metrics_path = Path("benchmarks/results/session_metrics.json")
        orch.save_metrics(metrics_path)

        with orch._metrics_lock:
            n = len(orch._metrics)
        if n > 0:
            print(f"\nSession complete. {n} queries. Results: {metrics_path}")
        else:
            print("\nNo queries recorded this session.")


if __name__ == "__main__":
    main()
