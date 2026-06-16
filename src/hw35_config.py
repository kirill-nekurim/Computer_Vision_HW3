"""Configuration for HW3.5: synthetic data + classifier ablation."""

from pathlib import Path

from src.config import COCO_SUBSET_DIR, OUTPUT_DIR, PROJECT_ROOT

HW35_DIR = OUTPUT_DIR / "hw35"
CLS_CROPS_DIR = PROJECT_ROOT / "data" / "cls_crops"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

# Rare classes selected for SD + ControlNet augmentation.
SYNTHETIC_CLASSES = ("truck", "bicycle", "chair")

# Stable Diffusion + ControlNet defaults.
SD_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
DEFAULT_IMAGES_PER_CLASS = 50
DEFAULT_SD_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROL_SCALE = 1.0
SD_IMAGE_SIZE = 512
MIN_REF_CROP_AREA = 12_000

# Classifier training defaults.
CLASSIFIER_MODEL = "resnet18"
DEFAULT_CLS_EPOCHS = 15
DEFAULT_CLS_BATCH_SIZE = 32
DEFAULT_CLS_LR = 1e-3
DEFAULT_CLS_IMAGE_SIZE = 224
DEFAULT_NUM_WORKERS = 4

CLASSIFIER_BASELINE_DIR = HW35_DIR / "baseline"
CLASSIFIER_SYNTHETIC_DIR = HW35_DIR / "with_synthetic"
ABLATION_PATH = HW35_DIR / "ablation.json"
SYNTHETIC_VIZ_PATH = HW35_DIR / "synthetic_examples.png"

TRAIN_ANN = COCO_SUBSET_DIR / "annotations" / "instances_train_subset.json"
VAL_ANN = COCO_SUBSET_DIR / "annotations" / "instances_val_subset.json"
TRAIN_IMG_DIR = COCO_SUBSET_DIR / "train2017"
VAL_IMG_DIR = COCO_SUBSET_DIR / "val2017"

PROMPTS = {
    "truck": (
        "(large cargo truck:1.3), photorealistic photo, truck as the main subject, "
        "clearly visible vehicle, highway, sharp focus, daylight"
    ),
    "bicycle": (
        "(bicycle:1.2), photorealistic photo, bicycle as the main subject, "
        "outdoor scene, sharp focus, high quality"
    ),
    "chair": (
        "(wooden chair:1.3), photorealistic photo, single chair as the main subject, "
        "centered in frame, indoor scene, sharp focus, high quality"
    ),
}

# Rotated during generation for diversity; first entry matches PROMPTS[class].
PROMPT_VARIANTS = {
    "truck": [
        PROMPTS["truck"],
        (
            "(delivery truck:1.3), white box truck, front three-quarter view, "
            "photorealistic, truck clearly visible, daylight"
        ),
        (
            "(semi truck:1.2), photorealistic photo, large truck as main subject, "
            "on road, sharp focus"
        ),
    ],
    "bicycle": [
        PROMPTS["bicycle"],
        (
            "(road bicycle:1.2), photorealistic photo, bicycle clearly visible, "
            "park or street, sharp focus"
        ),
    ],
    "chair": [
        PROMPTS["chair"],
        (
            "(dining chair:1.3), photorealistic photo, one chair clearly visible, "
            "living room, sharp focus"
        ),
        (
            "(office chair:1.2), photorealistic photo, chair as main subject, "
            "indoor scene, sharp focus"
        ),
    ],
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, cartoon, painting, watermark, text, letters, "
    "empty scene, no object, landscape only, abstract, texture only"
)

CLASS_NEGATIVE_PROMPTS = {
    "truck": "empty road, road only, highway without vehicle, no truck, cars only, street without truck",
    "bicycle": "empty path, no bicycle, person walking only, road without bicycle",
    "chair": "empty room, room without furniture, no chair, table only, sofa, bench",
}

# Per-class overrides when Canny edges over-emphasize background (e.g. road vs truck).
CLASS_GEN_PARAMS = {
    "truck": {"guidance_scale": 9.0, "control_scale": 0.75},
    "chair": {"guidance_scale": 9.5, "control_scale": 0.7},
    "bicycle": {"guidance_scale": 7.5, "control_scale": 1.0},
}
