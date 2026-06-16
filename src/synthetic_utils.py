"""Shared helpers for SD + ControlNet synthetic generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.hw35_config import MIN_REF_CROP_AREA, SD_IMAGE_SIZE


def crop_area(path: Path) -> int:
    with Image.open(path) as img:
        width, height = img.size
    return width * height


def green_dominance(img: Image.Image) -> float:
    """Share of pixels where green dominates — catches broccoli mislabeled as chair."""
    arr = np.array(img.resize((64, 64), Image.Resampling.BILINEAR))
    red = arr[:, :, 0].astype(int)
    green = arr[:, :, 1].astype(int)
    blue = arr[:, :, 2].astype(int)
    return float(((green > red + 15) & (green > blue + 15)).mean())


def is_plausible_reference(class_name: str, img: Image.Image) -> bool:
    if class_name == "chair" and green_dominance(img) >= 0.18:
        return False
    return True


def prepare_reference_image(ref: Image.Image, size: int = SD_IMAGE_SIZE) -> Image.Image:
    """Center-crop to square and resize — avoids letterbox bars in SD output."""
    ref = ref.copy()
    width, height = ref.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    ref = ref.crop((left, top, left + side, top + side))
    return ref.resize((size, size), Image.Resampling.LANCZOS)


def iter_reference_paths(
    crops_dir: Path,
    class_name: str,
    min_area: int = MIN_REF_CROP_AREA,
) -> list[Path]:
    class_dir = crops_dir / "train" / class_name
    paths = list(class_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No reference crops found in {class_dir}")

    large_paths = [path for path in paths if crop_area(path) >= min_area]
    candidates = large_paths if large_paths else paths
    candidates.sort(key=crop_area, reverse=True)

    valid: list[Path] = []
    for path in candidates:
        with Image.open(path) as img:
            if is_plausible_reference(class_name, img.convert("RGB")):
                valid.append(path)
    return valid if valid else candidates


def best_reference_path(crops_dir: Path, class_name: str, min_area: int = MIN_REF_CROP_AREA) -> Path:
    """Pick the largest plausible reference crop for notebook demos."""
    return iter_reference_paths(crops_dir, class_name, min_area=min_area)[0]
