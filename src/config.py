"""Central configuration for edge-vlm-assistant.

All paths, hardware constants, and latency targets live here.
Never hardcode paths or magic numbers elsewhere — import from this module.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
_models_dir_override = os.environ.get("MODELS_DIR")
MODELS_DIR: Path = Path(_models_dir_override) if _models_dir_override else Path.home() / "models"

MOONDREAM_PATH: Path = MODELS_DIR / "moondream3-mlx"
WHISPER_PATH: Path = MODELS_DIR / "whisper"
PIPER_PATH: Path = MODELS_DIR / "piper"
MARIAN_ES_EN_PATH: Path = MODELS_DIR / "marian-es-en"
# Silero VAD auto-downloads to HuggingFace cache; no path needed here.

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
# VLM-native resolution — never send raw 1080p to inference.
CAPTURE_WIDTH: int = 448
CAPTURE_HEIGHT: int = 448
CAPTURE_FPS: int = 30
RING_BUFFER_SIZE: int = 30  # ~1 second of frames at 30fps

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000       # Hz, required by Whisper and Silero VAD
CHANNELS: int = 1               # mono
# Silero VAD requires exactly 512 samples per chunk at 16kHz (= 32ms)
VAD_CHUNK_SAMPLES: int = 512
VAD_SPEECH_THRESHOLD: float = 0.5   # probability above which we consider speech
VAD_SILENCE_FRAMES: int = 20        # consecutive non-speech chunks before end-of-speech

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = "distil-small.en"   # falls back to path in WHISPER_PATH
WHISPER_COMPUTE_TYPE: str = "int8"
WHISPER_BEAM_SIZE: int = 1   # beam=1 for minimum latency

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
# Piper synthesize_stream_raw uses the voice's native sample rate (~22050 Hz)
TTS_SENTENCE_SILENCE: float = 0.0  # no gap between sentences for lower latency

# ---------------------------------------------------------------------------
# Latency targets (milliseconds) — these are THE headline metrics.
# ---------------------------------------------------------------------------
TARGET_TOTAL_MS: int = 800
TARGET_VAD_MS: int = 100
TARGET_ASR_MS: int = 150
TARGET_VLM_FIRST_TOKEN_MS: int = 400
TARGET_TTS_FIRST_CHUNK_MS: int = 150
