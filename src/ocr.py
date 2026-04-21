"""PaddleOCR wrapper — reads text from webcam frames.

Used for the OCR + translation demo flow (Phase 3).
Triggered when the user says "translate this", "read this", etc.

PaddleOCR is significantly smaller and faster than Tesseract for natural-scene
text (menus, signs, whiteboards) while requiring no model path management —
it auto-downloads its weights on first run.

Phase 3 standalone usage:
    python -m src.ocr
  Opens webcam, runs OCR on each frame, prints detected text. Press 'q' to quit.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class OCR:
    """PaddleOCR wrapper. Thread-safe after load() completes."""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._model = None  # type: ignore[assignment]

    def load(self) -> None:
        """Load (and auto-download if needed) PaddleOCR model weights."""
        t0 = time.perf_counter()
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped, import-not-found]
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Run: pip install paddlepaddle paddleocr"
            ) from exc

        # use_angle_cls=True handles rotated text (common on menus/signs)
        # show_log=False suppresses PaddleOCR's verbose startup output
        self._model = PaddleOCR(
            use_angle_cls=True,
            lang=self._lang,
            show_log=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("PaddleOCR loaded (lang=%s) in %.0fms", self._lang, elapsed_ms)

    def recognize(self, frame: np.ndarray) -> str:
        """Return all text detected in *frame* joined by newlines.

        Args:
            frame: BGR numpy array (any resolution; PaddleOCR resizes internally).

        Returns:
            Detected text, or empty string if nothing found.
        """
        if self._model is None:
            raise RuntimeError("Call OCR.load() before inference.")

        t0 = time.perf_counter()
        result = self._model.ocr(frame, cls=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # result: list of pages, each page is a list of [bbox, (text, confidence)]
        texts: list[str] = []
        for page in result or []:
            for line in page or []:
                if line and len(line) >= 2:
                    text_conf = line[1]
                    if isinstance(text_conf, (list, tuple)) and text_conf:
                        text = text_conf[0]
                        conf = text_conf[1] if len(text_conf) > 1 else 1.0
                        if conf > 0.5:  # filter low-confidence detections
                            texts.append(str(text))

        result_text = "\n".join(texts)
        logger.info(
            "OCR: %.0fms | %d text regions | '%s'",
            elapsed_ms,
            len(texts),
            result_text[:80],
        )
        return result_text


# ---------------------------------------------------------------------------
# Phase 3 standalone demo
# ---------------------------------------------------------------------------

def _run_phase3_demo() -> None:
    """Open webcam, run OCR every 2 seconds, print results."""
    import cv2  # type: ignore[import-untyped]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("Loading PaddleOCR...")
    ocr = OCR()
    ocr.load()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Press 'q' to quit, 'r' to run OCR on current frame.\n")
    last_ocr = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            if now - last_ocr > 2.0:
                text = ocr.recognize(frame)
                print(f"\n--- OCR result ---\n{text or '(no text detected)'}\n")
                last_ocr = now

            cv2.imshow("edge-vlm-assistant [OCR]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                text = ocr.recognize(frame)
                print(f"\n--- OCR result ---\n{text or '(no text detected)'}\n")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_phase3_demo()
