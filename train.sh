#!/usr/bin/env bash
set -euo pipefail

# Training script for BEVMap
# Starts Docker container and runs training with specified number of epochs
# Usage: ./train.sh [num_epochs] [work_dir]
# Example: ./train.sh 1 work_dirs/train_smoke

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="bevmap:latest"
DOCKERFILE_PATH="$ROOT_DIR/docker/Dockerfile"
DEFAULT_DATASET_ROOT="/home/bevlog/Documentos/BEVLOG/DATASET/finetuning/record_2025-10-15_10-05-18"
DATASET_ROOT="${BEVLOG_DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

# Parse command-line arguments
NUM_EPOCHS="${1:-1}"
WORK_DIR="${2:-work_dirs/train}"

cd "$ROOT_DIR"

# Build Docker image if it doesn't exist
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "[bevmap-train] Building Docker image '$IMAGE_NAME'..."
  docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" . > /dev/null 2>&1
  echo "[bevmap-train] Docker image built successfully."
fi

# Verify dataset exists
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[bevmap-train] ERROR: Dataset not found at: $DATASET_ROOT"
  echo "[bevmap-train] Set BEVLOG_DATASET_ROOT environment variable to override the location."
  exit 1
fi

echo "[bevmap-train] Starting training..."
echo "  Epochs: $NUM_EPOCHS"
echo "  Dataset: $DATASET_ROOT"
echo "  Work directory: $WORK_DIR"
echo ""

# Run training in Docker container
docker run --gpus all -it --rm \
  -v "$ROOT_DIR":/workspace \
  -v "$DATASET_ROOT":/workspace/data/bevlog_raw:ro \
  "$IMAGE_NAME" \
  bash -c "cd /workspace && python tools/train.py configs/bevmap/bevdet-r50.py --work-dir=$WORK_DIR --cfg-options max_epochs=$NUM_EPOCHS"
