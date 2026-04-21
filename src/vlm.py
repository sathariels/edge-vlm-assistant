"""VLM wrapper — Moondream 3 via mlx-vlm on Apple Silicon.

Why Moondream:
  Purpose-built for edge. Fits the ~4GB wearable RAM budget.
  MLX int4 quantization is already baked into moondream3-mlx.

Fallback:
  If mlx-vlm is unavailable (e.g. non-Apple-Silicon CI), falls back to
  transformers with device_map="mps" (Moondream 2).

Phase 1 standalone usage:
    python -m src.vlm
  Loads Moondream, runs one inference on a built-in test image,
  prints first-token latency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional, Any

import numpy as np

from src.config import MOONDREAM_PATH

logger = logging.getLogger(__name__)


class VLM:
    """Moondream wrapper with streaming generation and first-token timing."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or MOONDREAM_PATH
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None
        self._backend: str = "unloaded"

    def load(self) -> None:
        """Load Moondream 3 (MLX) or fall back to Moondream 2 (MPS)."""
        t0 = time.perf_counter()

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Moondream model not found at {self._model_path}. "
                "Run scripts/download_models.sh first."
            )

        try:
            self._load_mlx()
        except (ImportError, OSError, ValueError) as exc:
            # ImportError  → mlx-vlm not installed
            # OSError      → missing required architecture files
            # ValueError   → mlx-vlm doesn't support this model architecture
            #                (e.g. Moondream 3 params unknown to mlx-vlm 0.4.x)
            logger.warning(
                "mlx-vlm load failed (%s: %s), falling back to transformers+MPS",
                type(exc).__name__,
                exc,
            )
            self._load_mps()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "VLM loaded (%s) from '%s' in %.0fms",
            self._backend,
            self._model_path,
            elapsed_ms,
        )

    def _load_mlx(self) -> None:
        from mlx_vlm import load  # type: ignore[import-untyped]
        from mlx_vlm.utils import load_config  # type: ignore[import-untyped]

        self._model, self._processor = load(str(self._model_path), trust_remote_code=True)
        self._config = load_config(str(self._model_path), trust_remote_code=True)
        self._backend = "mlx"

    def _load_mps(self) -> None:
        # Load local Moondream model (any version) via transformers trust_remote_code.
        # The model dir ships its own hf_moondream.py architecture file.
        # Tokenizer is embedded in the model — no separate processor needed.
        import torch
        from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

        self._processor = None  # moondream's query() handles tokenization internally
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_path),
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="mps",
        )
        self._backend = "transformers-mps"

    def generate(
        self, frame: np.ndarray, prompt: str, max_tokens: int = 200
    ) -> tuple[str, float]:
        """Run VLM inference on *frame* with *prompt*.

        Returns:
            (response_text, first_token_latency_ms)

        The first_token_latency is approximated as the time to first token
        in streaming mode, or the total generation time in non-streaming mode
        (Phase 1 baseline — streaming is a Phase 3 optimization).
        """
        if self._model is None:
            raise RuntimeError("Call VLM.load() before inference.")

        if self._backend == "mlx":
            return self._generate_mlx(frame, prompt, max_tokens)
        else:
            return self._generate_mps(frame, prompt, max_tokens)

    def stream(
        self, frame: np.ndarray, prompt: str, max_tokens: int = 200
    ) -> Iterator[str]:
        """Yield token strings as the model generates them (Phase 3).

        The caller accumulates tokens into sentences and starts TTS on each
        complete sentence immediately, achieving VLM+TTS concurrency.

        Falls back to yielding the whole response as one chunk if the mlx-vlm
        version in the venv doesn't expose a streaming API (still better than
        Phase 2 because TTS starts as soon as the first 'chunk' arrives).
        """
        if self._model is None:
            raise RuntimeError("Call VLM.load() before inference.")

        if self._backend == "mlx":
            yield from self._stream_mlx(frame, prompt, max_tokens)
        else:
            # MPS path: no streaming API — yield full response as one chunk.
            # TTS pipeline still starts immediately on receipt of first chunk.
            response, _ = self._generate_mps(frame, prompt, max_tokens)
            yield response

    def _stream_mlx(
        self, frame: np.ndarray, prompt: str, max_tokens: int
    ) -> Iterator[str]:
        """Try mlx-vlm streaming APIs in priority order; fall back to blocking.

        Attempt 1 — mlx_vlm.utils.stream_generate  (mlx-vlm >= 0.1.x)
        Attempt 2 — mlx_vlm.generate(stream=True)  (some builds)
        Fallback  — blocking generate(), yield whole response as one chunk
        """
        import cv2
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-untyped]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        formatted = apply_chat_template(
            self._processor, self._config, prompt, num_images=1
        )

        # --- Attempt 1 ---
        try:
            from mlx_vlm.utils import stream_generate  # type: ignore[import-untyped]

            for result in stream_generate(
                self._model,
                self._processor,
                image_rgb,
                formatted,
                max_tokens=max_tokens,
            ):
                yield result.text if hasattr(result, "text") else str(result)
            return
        except (ImportError, AttributeError):
            pass

        # --- Attempt 2 ---
        try:
            from mlx_vlm import generate  # type: ignore[import-untyped]

            result = generate(
                self._model,
                self._processor,
                image_rgb,
                formatted,
                max_tokens=max_tokens,
                verbose=False,
                stream=True,
            )
            if hasattr(result, "__next__") or (
                hasattr(result, "__iter__") and not isinstance(result, str)
            ):
                for token in result:
                    yield token.text if hasattr(token, "text") else str(token)
                return
        except (ImportError, AttributeError, TypeError):
            pass

        # --- Fallback: blocking ---
        logger.debug("mlx-vlm streaming unavailable, falling back to blocking generate")
        response, _ = self._generate_mlx(frame, prompt, max_tokens)
        yield response

    def _generate_mlx(
        self, frame: np.ndarray, prompt: str, max_tokens: int
    ) -> tuple[str, float]:
        import cv2
        from mlx_vlm import generate  # type: ignore[import-untyped]
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-untyped]

        # mlx_vlm.generate accepts an image path or numpy array (BGR→RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        formatted = apply_chat_template(
            self._processor, self._config, prompt, num_images=1
        )

        t0 = time.perf_counter()
        # Phase 1: non-streaming baseline. verbose=False suppresses internal prints.
        response = generate(
            self._model,
            self._processor,
            image_rgb,
            formatted,
            max_tokens=max_tokens,
            verbose=False,
        )
        first_token_ms = (time.perf_counter() - t0) * 1000.0
        # NOTE: In Phase 1 this measures total generation time, not true first-token.
        # Phase 3 will wire up streaming to get the actual first-token latency.

        logger.info(
            "VLM [mlx]: total=%.0fms | '%s'",
            first_token_ms,
            (response or "")[:80],
        )
        return str(response), first_token_ms

    def _generate_mps(
        self, frame: np.ndarray, prompt: str, max_tokens: int
    ) -> tuple[str, float]:
        from PIL import Image  # type: ignore[import-untyped]

        image = Image.fromarray(frame[..., ::-1])  # BGR→RGB→PIL

        t0 = time.perf_counter()
        # encode_image and query are @property descriptors that return callables.
        # This API works for both Moondream 2 and Moondream 3 local checkpoints.
        enc_image = self._model.encode_image(image)
        result = self._model.query(enc_image, prompt)
        response = result["answer"] if isinstance(result, dict) else str(result)
        first_token_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "VLM [mps]: total=%.0fms | '%s'",
            first_token_ms,
            (response or "")[:80],
        )
        return str(response), first_token_ms


# ---------------------------------------------------------------------------
# Phase 1 standalone demo
# ---------------------------------------------------------------------------

def _run_phase1_demo() -> None:
    """Load Moondream, run on a static test image, print latency."""
    import cv2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print(f"Loading Moondream from {MOONDREAM_PATH}...")
    vlm = VLM()
    vlm.load()
    print(f"  Backend: {vlm._backend}")

    # Create a simple test image if no webcam frame is available
    test_frame = np.zeros((448, 448, 3), dtype=np.uint8)
    cv2.putText(
        test_frame,
        "Test image",
        (100, 224),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        3,
    )

    prompt = "What do you see in this image? Be brief."

    # Warmup inference (KV cache cold-start; not counted)
    print("\nRunning warmup inference (discarded)...")
    vlm.generate(test_frame, prompt, max_tokens=10)

    # Measured inference
    print("Running measured inference...")
    N_RUNS = 3
    latencies: list[float] = []
    for i in range(N_RUNS):
        response, latency_ms = vlm.generate(test_frame, prompt)
        latencies.append(latency_ms)
        print(f"  Run {i+1}: {latency_ms:.0f}ms → '{response[:60]}'")

    import statistics
    print(f"\nVLM latency over {N_RUNS} runs:")
    print(f"  p50 : {statistics.median(latencies):.0f}ms")
    print(f"  min : {min(latencies):.0f}ms")
    print(f"  max : {max(latencies):.0f}ms")
    print(f"  budget: {400}ms")
    if statistics.median(latencies) <= 400:
        print("  ✓ within budget")
    else:
        print("  ✗ over budget — Phase 3 optimizations needed")


if __name__ == "__main__":
    _run_phase1_demo()
