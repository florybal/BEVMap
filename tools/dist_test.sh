#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT [GPUS] [extra args]"
  exit 1
fi

CONFIG=$1
CHECKPOINT=$2
shift 2

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  GPUS=$1
  shift
else
  GPUS=1
fi

PORT=${PORT:-29600}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    "$(dirname "$0")/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:1}" --eval bbox
