"""Shared configuration for DETR fine-tuning on COCO subset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COCO_DIR = DATA_DIR / "coco"
COCO_SUBSET_DIR = DATA_DIR / "coco_subset"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"
PROFILER_DIR = OUTPUT_DIR / "profiler"
PLOTS_DIR = OUTPUT_DIR / "plots"
VIS_DIR = OUTPUT_DIR / "visualizations"
METRICS_DIR = OUTPUT_DIR / "metrics"

# 10 COCO classes (name -> original COCO category_id)
# NOTE: bottle=44, cup=47, chair=62 (NOT 39/41/56 which are baseball bat/skateboard/broccoli)
SELECTED_CATEGORIES = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "bus": 6,
    "truck": 8,
    "cat": 17,
    "dog": 18,
    "bottle": 44,
    "cup": 47,
    "chair": 62,
}

MODEL_NAME = "facebook/detr-resnet-50"
NUM_CLASSES = len(SELECTED_CATEGORIES)

# Training defaults (override via CLI / notebook)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-5
DEFAULT_LR_BACKBONE = 1e-6
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_GRAD_NORM = 0.1
DEFAULT_NUM_WORKERS = 4
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
