"""Loop 2 — Microphone capture, VAD, and ASR trigger.

Architecture:
  sounddevice callback → VAD → speech segment buffer
  On end-of-speech: call on_utterance(audio_array, t_speech_start, t_speech_end)

The loop runs entirely in sounddevice's audio callback thread.
ASR is kicked off from *outside* this loop (by the orchestrator or inline
for Phase 1 testing) to avoid blocking audio capture.

Phase 1 standalone usage:
    python -m src.audio
  Prints "SPEECH START" / "SPEECH END" with timestamps. Press Ctrl+C to quit.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from src.config import (
    CHANNELS,
    SAMPLE_RATE,
    VAD_CHUNK_SAMPLES,
    VAD_SILENCE_FRAMES,
    VAD_SPEECH_THRESHOLD,
)
from src.vad import VAD

logger = logging.getLogger(__name__)

# Callback signature: (audio: np.ndarray, t_start: float, t_end: float) -> None
UtteranceCallback = Callable[[np.ndarray, float, float], None]


class AudioLoop:
    """Mic → Silero VAD → utterance segments, running in a dedicated thread."""

    def __init__(
        self,
        vad: VAD,
        on_utterance: Optional[UtteranceCallback] = None,
    ) -> None:
        self._vad = vad
        self._on_utterance = on_utterance or self._default_utterance_handler

        # State machine
        self._in_speech: bool = False
        self._silence_frames: int = 0
        self._speech_buffer: list[np.ndarray] = []
        self._t_speech_start: float = 0.0

        # Queue for delivering utterances from callback thread to a worker thread
        # so ASR never runs inside the audio callback.
        self._utterance_queue: queue.Queue[
            tuple[np.ndarray, float, float]
        ] = queue.Queue()

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._dispatch_worker, daemon=True, name="audio-dispatch"
        )
        self._worker_thread.start()
        logger.info(
            "AudioLoop starting (rate=%d, chunk=%d samples = %.0fms)",
            SAMPLE_RATE,
            VAD_CHUNK_SAMPLES,
            VAD_CHUNK_SAMPLES / SAMPLE_RATE * 1000,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._utterance_queue.put_nowait(None)  # type: ignore[arg-type]  # sentinel
            self._worker_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # sounddevice integration
    # ------------------------------------------------------------------

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice from the audio thread for each chunk."""
        if status:
            logger.warning("sounddevice status: %s", status)

        chunk = indata[:, 0].copy()  # mono, float32
        t_now = time.perf_counter()

        prob = self._vad.speech_prob(chunk, SAMPLE_RATE)

        if prob >= VAD_SPEECH_THRESHOLD:
            if not self._in_speech:
                self._in_speech = True
                self._speech_buffer = []
                self._t_speech_start = t_now
                logger.debug("VAD: speech start (prob=%.2f)", prob)
            self._silence_frames = 0
            self._speech_buffer.append(chunk)
        else:
            if self._in_speech:
                self._speech_buffer.append(chunk)
                self._silence_frames += 1
                if self._silence_frames >= VAD_SILENCE_FRAMES:
                    # End of speech — hand off the segment
                    t_end = t_now
                    audio = np.concatenate(self._speech_buffer)
                    self._utterance_queue.put_nowait((audio, self._t_speech_start, t_end))
                    self._in_speech = False
                    self._silence_frames = 0
                    self._speech_buffer = []
                    logger.debug(
                        "VAD: speech end (%.0fms)", (t_end - self._t_speech_start) * 1000
                    )

    # ------------------------------------------------------------------
    # Worker thread — dispatches utterances without blocking audio
    # ------------------------------------------------------------------

    def _dispatch_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            audio, t_start, t_end = item
            self._on_utterance(audio, t_start, t_end)

    # ------------------------------------------------------------------
    # Default handler (Phase 1 demo)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_utterance_handler(
        audio: np.ndarray, t_start: float, t_end: float
    ) -> None:
        duration_ms = (t_end - t_start) * 1000.0
        print(
            f"[{time.strftime('%H:%M:%S')}] UTTERANCE: {duration_ms:.0f}ms, "
            f"{len(audio)/SAMPLE_RATE:.2f}s of audio"
        )


# ---------------------------------------------------------------------------
# Phase 1 standalone demo
# ---------------------------------------------------------------------------

def _run_phase1_demo() -> None:
    """Listen on the mic, print speech start/end events with wall-clock timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("Loading Silero VAD...")
    vad = VAD()
    vad.load()

    speech_events: list[tuple[str, float]] = []

    def on_utterance(audio: np.ndarray, t_start: float, t_end: float) -> None:
        duration_ms = (t_end - t_start) * 1000.0
        samples = len(audio)
        print(
            f"\n  SPEECH END  | duration={duration_ms:.0f}ms  samples={samples}  "
            f"audio_len={samples/SAMPLE_RATE:.2f}s"
        )
        speech_events.append(("end", t_end))

    # Monkey-patch to also print start events
    _orig_callback_maker = AudioLoop

    class _VerboseAudioLoop(AudioLoop):
        def audio_callback(
            self,
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            was_in_speech = self._in_speech
            super().audio_callback(indata, frames, time_info, status)
            if self._in_speech and not was_in_speech:
                print(f"\n  SPEECH START| t={time.perf_counter():.3f}", flush=True)
                speech_events.append(("start", time.perf_counter()))

    loop = _VerboseAudioLoop(vad=vad, on_utterance=on_utterance)
    loop.start()

    print(
        f"\nListening on default mic (rate={SAMPLE_RATE}Hz, "
        f"chunk={VAD_CHUNK_SAMPLES} samples = "
        f"{VAD_CHUNK_SAMPLES/SAMPLE_RATE*1000:.0f}ms)."
    )
    print("Speak into the mic. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_CHUNK_SAMPLES,
            callback=loop.audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n\nStopping. Captured {len(speech_events)} events.")
    finally:
        loop.stop()


if __name__ == "__main__":
    _run_phase1_demo()
