#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="bevmap:latest"
DOCKERFILE_PATH="$ROOT_DIR/docker/Dockerfile"
BUILD_LOG="$ROOT_DIR/docker_build.log"
DEFAULT_DATASET_ROOT="/home/bevlog/Documentos/BEVLOG/DATASET/finetuning/record_2025-10-15_10-05-18"
DATASET_ROOT="${BEVLOG_DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"

cd "$ROOT_DIR"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "[bevmap] Docker image '$IMAGE_NAME' not found. Building..."
  docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" . | tee "$BUILD_LOG"
else
  echo "[bevmap] Docker image '$IMAGE_NAME' already exists."
fi

echo "[bevmap] Starting container..."
DOCKER_RUN_ARGS=(--gpus all -it --rm -v "$ROOT_DIR":/workspace)

if [[ -d "$DATASET_ROOT" ]]; then
  echo "[bevmap] Mounting dataset from: $DATASET_ROOT"
  DOCKER_RUN_ARGS+=(-v "$DATASET_ROOT":/workspace/data/bevlog_raw:ro)
else
  echo "[bevmap] Dataset path not found: $DATASET_ROOT"
  echo "[bevmap] You can export BEVLOG_DATASET_ROOT to override the location."
fi

docker run "${DOCKER_RUN_ARGS[@]}" "$IMAGE_NAME"
