#!/usr/bin/env bash
# Overnight run: medium subset + 20 epochs (~4–5 h on V100).
# Usage: tmux new -s detr && ./scripts/run_overnight.sh
set -euo pipefail

export EPOCHS="${EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_TRAIN="${MAX_TRAIN:-2500}"
export MAX_VAL="${MAX_VAL:-400}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LR="${LR:-1e-5}"
export LR_BACKBONE="${LR_BACKBONE:-1e-6}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Overnight DETR run ==="
echo "  train images: $MAX_TRAIN"
echo "  val images:   $MAX_VAL"
echo "  epochs:       $EPOCHS"
echo "  batch size:   $BATCH_SIZE"
echo "  lr / lr_bb:   $LR / $LR_BACKBONE"
echo "  (~${MAX_TRAIN} train + ~${MAX_VAL} val images to download; already cached files are skipped)"
echo ""

exec "$ROOT/scripts/run_server.sh"
