"""Loop 3 — Inference pipeline (event-driven).

Called once per utterance from the orchestrator's dispatch thread.
Never runs inside the audio callback — that thread must stay free for capture.

Phase 3 pipeline (concurrent VLM + TTS):
──────────────────────────────────────────────────────────────────────────────
  inference thread:   ASR → grab frame → VLM.stream() → sentence accumulator
                                                              │
                                                    sentence_queue.put()
                                                              │
  tts thread:                                    sentence_queue.get()
                                                     → TTS.synthesize_streaming()
                                                     → sounddevice write
──────────────────────────────────────────────────────────────────────────────

The TTS thread starts as soon as the FIRST complete sentence is ready (~200ms
into VLM generation), while VLM continues generating the rest of the response.
End-to-end first-syllable latency shrinks from ASR+VLM+TTS to ASR+VLM_sent1+TTS.

OCR + translation demo flow (Phase 3 feature):
  If the transcript contains keywords ("translate", "read this", …),
  OCR is run on the frame instead of VLM, and the result is translated es→en.
  The translated text is then spoken via TTS exactly like a VLM response.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from src.asr import ASR
from src.config import CAPTURE_HEIGHT, CAPTURE_WIDTH
from src.metrics import QueryMetrics
from src.ring_buffer import RingBuffer
from src.tts import TTS
from src.vlm import VLM

if TYPE_CHECKING:
    from src.ocr import OCR
    from src.translate import Translator

logger = logging.getLogger(__name__)

MetricsCallback = Callable[[QueryMetrics], None]

# ---------------------------------------------------------------------------
# Sentence-boundary detection
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r'[.!?]["\']?\s*$')

def _is_sentence_boundary(text: str) -> bool:
    return bool(_SENTENCE_END.search(text))

# ---------------------------------------------------------------------------
# OCR trigger keywords
# ---------------------------------------------------------------------------
_OCR_KEYWORDS = frozenset([
    "translate", "read this", "read that",
    "what does this say", "what does it say", "what does that say",
    "what is written", "what's written", "what is it saying",
])

def _wants_ocr(transcript: str) -> bool:
    t = transcript.lower()
    return any(kw in t for kw in _OCR_KEYWORDS)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Runs a single voice→vision→speech query end-to-end.

    Instantiate once; call process_utterance() on every utterance.
    OCR and Translator are optional — pass them to enable the OCR demo flow.
    """

    def __init__(
        self,
        vlm: VLM,
        tts: TTS,
        asr: ASR,
        ring_buffer: RingBuffer[np.ndarray],
        on_metrics: Optional[MetricsCallback] = None,
        ocr: Optional["OCR"] = None,
        translator: Optional["Translator"] = None,
    ) -> None:
        self._vlm = vlm
        self._tts = tts
        self._asr = asr
        self._ring_buffer = ring_buffer
        self._on_metrics = on_metrics
        self._ocr = ocr
        self._translator = translator
        self._query_counter: int = 0

        # Fallback frame when ring buffer hasn't received any frames yet
        self._fallback_frame: np.ndarray = np.zeros(
            (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_utterance(
        self,
        audio: np.ndarray,
        t_speech_start: float,
        t_speech_end: float,
    ) -> Optional[QueryMetrics]:
        """Run the full pipeline for one captured utterance.

        Args:
            audio:          float32 mono audio at 16kHz from the VAD segment.
            t_speech_start: perf_counter() timestamp of speech onset.
            t_speech_end:   perf_counter() timestamp of end-of-speech.

        Returns:
            Populated QueryMetrics, or None if transcript was empty.
        """
        m = QueryMetrics(query_id=self._query_counter)
        self._query_counter += 1
        m.t_speech_start = t_speech_start
        m.t_speech_end = t_speech_end

        # ------------------------------------------------------------------
        # Stage 1: ASR
        # ------------------------------------------------------------------
        m.t_asr_start = time.perf_counter()
        transcript = self._asr.transcribe(audio)
        m.t_asr_end = time.perf_counter()
        m.transcript = transcript

        if not transcript.strip():
            logger.debug("Empty transcript — skipping inference")
            return None

        # ------------------------------------------------------------------
        # Stage 2: grab freshest frame (O(1))
        # Copy immediately so the capture thread can't overwrite the pool slot
        # while we're running inference.
        # ------------------------------------------------------------------
        frame_ref = self._ring_buffer.latest()
        m.t_frame_grabbed = time.perf_counter()
        frame: np.ndarray = (
            frame_ref.copy() if frame_ref is not None else self._fallback_frame
        )
        if frame_ref is None:
            logger.warning("Ring buffer empty — using blank fallback frame")

        # ------------------------------------------------------------------
        # Stage 3+4: VLM/OCR → TTS (concurrent)
        # ------------------------------------------------------------------
        if self._ocr is not None and _wants_ocr(transcript):
            response = self._run_ocr_pipeline(frame, m)
        else:
            response = self._run_vlm_pipeline(frame, transcript, m)

        m.response = response

        if not response.strip():
            logger.warning("Empty response — nothing to speak")
            return m

        m.log_summary()

        if self._on_metrics is not None:
            self._on_metrics(m)

        return m

    # ------------------------------------------------------------------
    # VLM streaming pipeline (Phase 3)
    # ------------------------------------------------------------------

    def _run_vlm_pipeline(
        self,
        frame: np.ndarray,
        transcript: str,
        m: QueryMetrics,
    ) -> str:
        """Stream VLM tokens into TTS concurrently. Returns full response text."""
        sentence_queue: queue.Queue[Optional[str]] = queue.Queue()
        full_response_parts: list[str] = []
        tts_done = threading.Event()

        def tts_worker() -> None:
            first_chunk_captured = False
            try:
                with sd.RawOutputStream(
                    samplerate=self._tts.sample_rate,
                    channels=1,
                    dtype="int16",
                ) as stream:
                    while True:
                        sentence = sentence_queue.get()
                        if sentence is None:  # VLM done sentinel
                            break
                        if not sentence.strip():
                            continue
                        for chunk in self._tts.synthesize_streaming(sentence):
                            if not first_chunk_captured:
                                m.t_tts_first_chunk = time.perf_counter()
                                first_chunk_captured = True
                            stream.write(chunk)
            finally:
                if not first_chunk_captured:
                    m.t_tts_first_chunk = time.perf_counter()
                m.t_tts_end = time.perf_counter()
                tts_done.set()

        tts_thread = threading.Thread(target=tts_worker, daemon=True, name="tts-stream")
        tts_thread.start()
        m.t_tts_start = time.perf_counter()

        # Stream VLM tokens; flush to TTS queue on each sentence boundary
        m.t_vlm_start = time.perf_counter()
        first_token = True
        token_buf = ""

        try:
            for token in self._vlm.stream(frame, transcript):
                if first_token:
                    m.t_vlm_first_token = time.perf_counter()
                    first_token = False
                token_buf += token
                full_response_parts.append(token)

                if _is_sentence_boundary(token_buf):
                    sentence_queue.put(token_buf.strip())
                    token_buf = ""

            # Flush any trailing text that didn't end with punctuation
            if token_buf.strip():
                full_response_parts.append(token_buf)
                sentence_queue.put(token_buf.strip())

        finally:
            sentence_queue.put(None)  # sentinel always sent, even on exception

        m.t_vlm_end = time.perf_counter()
        if first_token:  # stream yielded nothing
            m.t_vlm_first_token = m.t_vlm_end

        tts_done.wait()
        return "".join(full_response_parts)

    # ------------------------------------------------------------------
    # OCR + translation pipeline (Phase 3 feature)
    # ------------------------------------------------------------------

    def _run_ocr_pipeline(
        self,
        frame: np.ndarray,
        m: QueryMetrics,
    ) -> str:
        """OCR the current frame, translate es→en, speak the result."""
        assert self._ocr is not None

        t0 = time.perf_counter()
        detected_text = self._ocr.recognize(frame)
        m.extra["ocr_ms"] = (time.perf_counter() - t0) * 1000.0

        if not detected_text.strip():
            response = "I don't see any readable text in front of you."
        elif self._translator is not None:
            t1 = time.perf_counter()
            translated = self._translator.translate(detected_text)
            m.extra["translate_ms"] = (time.perf_counter() - t1) * 1000.0
            response = f"The text says: {translated}"
        else:
            response = f"I can read: {detected_text}"

        # Speak via the same concurrent TTS path
        m.t_vlm_start = t0
        m.t_vlm_first_token = time.perf_counter()
        m.t_vlm_end = m.t_vlm_first_token

        # Re-use the streaming TTS infrastructure (response is already complete)
        # VLM is not involved here, so we feed the response directly.
        # Wrap it as a single-sentence VLM "response" — sentence queue path handles it.
        sentence_queue: queue.Queue[Optional[str]] = queue.Queue()
        tts_done = threading.Event()

        def tts_worker() -> None:
            first_chunk_captured = False
            try:
                with sd.RawOutputStream(
                    samplerate=self._tts.sample_rate,
                    channels=1,
                    dtype="int16",
                ) as stream:
                    sentence = sentence_queue.get()
                    if sentence:
                        for chunk in self._tts.synthesize_streaming(sentence):
                            if not first_chunk_captured:
                                m.t_tts_first_chunk = time.perf_counter()
                                first_chunk_captured = True
                            stream.write(chunk)
            finally:
                if not first_chunk_captured:
                    m.t_tts_first_chunk = time.perf_counter()
                m.t_tts_end = time.perf_counter()
                tts_done.set()

        m.t_tts_start = time.perf_counter()
        tts_thread = threading.Thread(target=tts_worker, daemon=True, name="tts-ocr")
        tts_thread.start()
        sentence_queue.put(response)
        tts_done.wait()

        return response
