"""Utility helpers for COCO subset and DETR training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import SELECTED_CATEGORIES


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_id_mappings() -> tuple[dict[int, int], dict[int, str], dict[str, int]]:
    """Map original COCO category ids to contiguous 0..N-1 labels."""
    old_to_new: dict[int, int] = {}
    id_to_label: dict[int, str] = {}
    label_to_id: dict[str, int] = {}

    for new_id, (name, old_id) in enumerate(SELECTED_CATEGORIES.items()):
        old_to_new[old_id] = new_id
        id_to_label[new_id] = name
        label_to_id[name] = new_id

    return old_to_new, id_to_label, label_to_id


def save_label_map(path: Path) -> dict[str, Any]:
    old_to_new, id_to_label, label_to_id = build_id_mappings()
    payload = {
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
        "num_classes": len(id_to_label),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_label_map(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["old_to_new"] = {int(k): v for k, v in data["old_to_new"].items()}
    data["id_to_label"] = {int(k): v for k, v in data["id_to_label"].items()}
    return data


def xyxy_to_xywh(box: list[float]) -> list[float]:
    """Convert absolute (x_min, y_min, x_max, y_max) to COCO [x, y, w, h]."""
    x_min, y_min, x_max, y_max = box
    return [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)]


def load_detr_processor(model_name: str = "facebook/detr-resnet-50"):
    """Load DETR image processor (compat across transformers lazy-import quirks)."""
    try:
        from transformers import DetrImageProcessor
    except ImportError:
        from transformers.models.detr.image_processing_detr import DetrImageProcessor
    return DetrImageProcessor.from_pretrained(model_name)
