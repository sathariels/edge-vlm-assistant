"""MarianMT es→en translation wrapper.

Used for the OCR + translation demo flow (Phase 3).
Translates Spanish text (extracted by OCR) to English.

Why MarianMT over NLLB-200:
  MarianMT Helsinki-NLP/opus-mt-es-en is a single-pair model (~600MB vs 1.2GB
  for NLLB-200-distilled-600M). For a single-language demo, per-pair models
  are faster and smaller — consistent with the wearable RAM budget thesis.

Phase 3 standalone usage:
    python -m src.translate
  Translates a test sentence from Spanish to English, prints latency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.config import MARIAN_ES_EN_PATH

logger = logging.getLogger(__name__)


class Translator:
    """Helsinki-NLP MarianMT es→en translation wrapper."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or MARIAN_ES_EN_PATH
        self._model = None  # type: ignore[assignment]
        self._tokenizer = None  # type: ignore[assignment]

    def load(self) -> None:
        """Load MarianMT model and tokenizer from local path."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"MarianMT model not found at {self._model_path}. "
                "Run scripts/download_models.sh first."
            )

        from transformers import MarianMTModel, MarianTokenizer  # type: ignore[import-untyped]

        t0 = time.perf_counter()
        self._tokenizer = MarianTokenizer.from_pretrained(str(self._model_path))
        self._model = MarianMTModel.from_pretrained(str(self._model_path))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "MarianMT es→en loaded from '%s' in %.0fms",
            self._model_path,
            elapsed_ms,
        )

    def translate(self, text: str) -> str:
        """Translate *text* from Spanish to English.

        Args:
            text: Spanish text, possibly multi-line (each line translated separately).

        Returns:
            Translated English text. Lines preserved.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call Translator.load() before translating.")
        if not text.strip():
            return ""

        t0 = time.perf_counter()

        # Split into lines so OCR-extracted multi-line text is handled naturally
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        inputs = self._tokenizer(
            lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        translated_ids = self._model.generate(**inputs)
        translated_lines = self._tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )

        result = " ".join(translated_lines)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Translation: %.0fms | '%s' → '%s'",
            elapsed_ms,
            text[:40],
            result[:40],
        )
        return result


# ---------------------------------------------------------------------------
# Phase 3 standalone demo
# ---------------------------------------------------------------------------

def _run_phase3_demo() -> None:
    """Translate test sentences, print latency."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    TEST_SENTENCES = [
        "Hola, ¿cómo estás?",
        "Este es un menú de restaurante.",
        "Por favor, gire a la derecha en la próxima calle.",
        "Precaución: zona de construcción.",
    ]

    print(f"Loading MarianMT from {MARIAN_ES_EN_PATH}...")
    translator = Translator()
    translator.load()
    print()

    for sentence in TEST_SENTENCES:
        t0 = time.perf_counter()
        result = translator.translate(sentence)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"  {ms:5.0f}ms | '{sentence}' → '{result}'")


if __name__ == "__main__":
    _run_phase3_demo()
