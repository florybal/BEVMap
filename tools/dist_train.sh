#!/usr/bin/env bash
set -euo pipefail

POSARGS=("$@")
CONFIG="./configs/bevmap/bevdet-r50.py"

if [[ ${#POSARGS[@]} -gt 0 && "${POSARGS[0]}" != --* ]]; then
  CONFIG="${POSARGS[0]}"
  POSARGS=("${POSARGS[@]:1}")
fi

CONFIG="${CONFIG:-${CONFIG_FILE:-}}"
if [[ -z "$CONFIG" ]]; then
  echo "ERROR: config file not provided."
  echo "Usage: $0 CONFIG [GPUS] [extra args]"
  echo "Or set CONFIG_FILE environment variable."
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG"
  exit 1
fi

GPUS_ARG=""
if [[ ${#POSARGS[@]} -gt 0 && "${POSARGS[0]}" =~ ^[0-9]+$ ]]; then
  GPUS_ARG="${POSARGS[0]}"
  POSARGS=("${POSARGS[@]:1}")
fi

GPUS="${GPUS_ARG:-${GPUS:-1}}"
PORT=${PORT:-29500}
PYTHONPATH=${PYTHONPATH:-}
ARGS=("${POSARGS[@]}")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--work_dirs" ]]; then
    ARGS[$i]="--work-dir"
  fi
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    "$(dirname "$0")/train.py" "$CONFIG" --launcher pytorch "${ARGS[@]}"
