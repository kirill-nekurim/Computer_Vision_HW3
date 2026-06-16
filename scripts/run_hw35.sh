#!/usr/bin/env bash
# HW3.5 pipeline: crops -> synthetic generation -> classifier ablation
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== 1. Prepare classification crops ==="
python -m src.prepare_cls_crops

echo "=== 2. Generate synthetic images (SD + ControlNet) ==="
python scripts/generate_synthetic.py \
  --num-images "${NUM_SYNTHETIC:-50}" \
  --classes truck bicycle chair

echo "=== 3. Train classifier ablation ==="
python -m src.train_classifier \
  --epochs "${CLS_EPOCHS:-15}" \
  --batch-size "${CLS_BATCH_SIZE:-32}"

echo "Done. See outputs/hw35/"
