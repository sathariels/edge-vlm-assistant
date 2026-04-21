#!/usr/bin/env bash
# download_models.sh — idempotent model fetcher for edge-vlm-assistant
#
# Run once on a new machine before anything else.
# Checks existence before downloading — safe to re-run.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR"

echo "Models directory: $MODELS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
check_and_download() {
    local name="$1"
    local path="$2"
    local cmd="$3"

    if [ -e "$path" ]; then
        echo "✓  $name — already exists at $path"
    else
        echo "↓  $name — downloading..."
        eval "$cmd"
        echo "✓  $name — done"
    fi
}

# ---------------------------------------------------------------------------
# Moondream 2 MLX int4 — ~1.8 GB
#
# Source: mlx-community/moondream2-4bit
#
# Why mlx-community and not vikhyatk/moondream3:
#   mlx-vlm's load() function dynamically imports a Python architecture file
#   (hf_moondream.py) that must live inside the model directory. The
#   mlx-community conversion repos include this file alongside MLX weights.
#   vikhyatk/moondream3 is not yet publicly available as a standalone HF
#   repo; attempting to snapshot_download it produces a 404. When Moondream 3
#   ships an official mlx-community conversion, update the repo_id here and
#   in src/config.py MOONDREAM_PATH.
#
# Do NOT add ignore_patterns here — we need ALL files including *.py and
# *.safetensors. The mlx-community repo ships MLX .safetensors, not PyTorch.
# ---------------------------------------------------------------------------
check_and_download \
    "Moondream 2 MLX int4 (mlx-community)" \
    "$MODELS_DIR/moondream3-mlx" \
    "python -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/moondream2-4bit',
    local_dir='$MODELS_DIR/moondream3-mlx',
)
\""

# ---------------------------------------------------------------------------
# distil-whisper-small.en — ~320 MB
# ---------------------------------------------------------------------------
check_and_download \
    "distil-whisper-small.en" \
    "$MODELS_DIR/whisper" \
    "python -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ctranslate2-4you/distil-whisper-small.en-ct2-int8',
    local_dir='$MODELS_DIR/whisper',
)
\""

# ---------------------------------------------------------------------------
# Piper voice — en_US-lessac-medium (~61 MB)
# ---------------------------------------------------------------------------
check_and_download \
    "Piper en_US-lessac-medium" \
    "$MODELS_DIR/piper/en_US-lessac-medium.onnx" \
    "mkdir -p '$MODELS_DIR/piper' && \
     curl -L -o '$MODELS_DIR/piper/en_US-lessac-medium.onnx' \
       'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx' && \
     curl -L -o '$MODELS_DIR/piper/en_US-lessac-medium.onnx.json' \
       'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json'"

# ---------------------------------------------------------------------------
# MarianMT es→en — ~599 MB (Phase 2)
# ---------------------------------------------------------------------------
check_and_download \
    "MarianMT es→en" \
    "$MODELS_DIR/marian-es-en" \
    "python -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Helsinki-NLP/opus-mt-es-en',
    local_dir='$MODELS_DIR/marian-es-en',
)
\""

echo ""
echo "All models ready. Run 'python scripts/sanity_check.py' to verify."
