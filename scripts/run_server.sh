#!/usr/bin/env bash
# Full pipeline on a Linux server with NVIDIA GPU (e.g. V100).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-5}"
LR_BACKBONE="${LR_BACKBONE:-1e-6}"
MAX_TRAIN="${MAX_TRAIN:-2500}"
MAX_VAL="${MAX_VAL:-400}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "=== HW3 DETR server pipeline ==="
echo "Project: $ROOT"

if ! command -v python3 >/dev/null; then
  echo "python3 not found"
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# V100 (CC 7.0): recent default torch wheels (e.g. 2.12+cu130) lack sm_70 kernels
if ! python - <<'PY' 2>/dev/null
import torch
if not torch.cuda.is_available():
    raise SystemExit(0)
x = torch.randn(1, 3, 8, 8, device="cuda")
torch.nn.Conv2d(3, 4, 3, device="cuda")(x)
PY
then
  echo "Reinstalling PyTorch 2.5.1+cu124 for V100..."
  pip install -q --force-reinstall \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
  pip install -q "transformers>=4.40,<5"
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — check nvidia-smi and torch install"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
PY

echo "=== 1/6 Annotations ==="
if [[ -f data/coco/annotations/instances_train2017.json ]]; then
  echo "Annotations already present — skip download"
else
  python scripts/download_coco.py --annotations-only
fi

echo "=== 2/6 Subset JSON ==="
python -m src.prepare_coco_subset \
  --skip-image-copy \
  --max-train-images "$MAX_TRAIN" \
  --max-val-images "$MAX_VAL"

echo "=== 3/6 Subset images ==="
python scripts/download_subset_images.py

echo "=== 4/6 Training ==="
python -m src.train \
  --data-dir data/coco_subset \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr-backbone "$LR_BACKBONE" \
  --num-workers "$NUM_WORKERS" \
  --max-train-samples "$MAX_TRAIN" \
  --max-val-samples "$MAX_VAL"

echo "=== 5/6 Evaluation (best checkpoint) ==="
CHECKPOINT="outputs/checkpoints/best"
if [[ ! -d "$CHECKPOINT" ]]; then
  CHECKPOINT="outputs/checkpoints/final"
fi
python -m src.evaluate \
  --data-dir data/coco_subset \
  --checkpoint "$CHECKPOINT" \
  --split val

echo "=== 6/6 Error analysis ==="
python -m src.error_analysis \
  --data-dir data/coco_subset \
  --checkpoint "$CHECKPOINT" \
  --max-samples 200 \
  --num-examples 8

echo ""
echo "Done."
echo "  metrics:  outputs/metrics/metrics_val.json"
echo "  plots:    outputs/plots/losses.png"
echo "  tensorboard: outputs/tensorboard/"
echo "  profiler: outputs/profiler/trace.json"
echo "  visuals:  outputs/visualizations/"
