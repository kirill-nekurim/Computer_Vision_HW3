"""Extract cropped object images from COCO subset for classification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from src.config import COCO_SUBSET_DIR
from src.hw35_config import CLS_CROPS_DIR, SYNTHETIC_CLASSES
from src.utils import ensure_dirs, load_label_map


def crop_with_padding(
    image: Image.Image,
    bbox: list[float],
    pad_ratio: float = 0.1,
    min_size: int = 32,
) -> Image.Image | None:
    x, y, w, h = bbox
    if w < min_size or h < min_size:
        return None

    img_w, img_h = image.size
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    x0 = max(0, int(x - pad_x))
    y0 = max(0, int(y - pad_y))
    x1 = min(img_w, int(x + w + pad_x))
    y1 = min(img_h, int(y + h + pad_y))
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def export_split(
    split: str,
    ann_file: Path,
    image_dir: Path,
    output_dir: Path,
    id_to_label: dict[int, str],
) -> dict[str, int]:
    coco = COCO(str(ann_file))
    split_dir = output_dir / split
    ensure_dirs(split_dir)

    counts: Counter[str] = Counter()
    ann_ids = sorted(coco.getAnnIds())
    anns_all = coco.loadAnns(ann_ids)
    ann_ids = [ann["id"] for ann in anns_all if ann.get("iscrowd", 0) == 0]

    for ann_id in tqdm(ann_ids, desc=f"crops {split}"):
        ann = coco.loadAnns(ann_id)[0]
        class_name = id_to_label[ann["category_id"]]
        img_info = coco.loadImgs(ann["image_id"])[0]
        image_path = image_dir / img_info["file_name"]
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        crop = crop_with_padding(image, ann["bbox"])
        if crop is None:
            continue

        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{ann['image_id']:012d}_{ann_id:06d}.jpg"
        crop.save(class_dir / out_name, quality=95)
        counts[class_name] += 1

    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare classification crops from COCO subset.")
    parser.add_argument("--output-dir", type=Path, default=CLS_CROPS_DIR)
    parser.add_argument("--data-dir", type=Path, default=COCO_SUBSET_DIR)
    args = parser.parse_args()

    label_map = load_label_map(args.data_dir / "label_map.json")
    id_to_label = label_map["id_to_label"]

    splits = {
        "train": (
            args.data_dir / "annotations" / "instances_train_subset.json",
            args.data_dir / "train2017",
        ),
        "val": (
            args.data_dir / "annotations" / "instances_val_subset.json",
            args.data_dir / "val2017",
        ),
    }

    stats: dict[str, dict[str, int]] = {}
    for split, (ann_file, image_dir) in splits.items():
        stats[split] = export_split(split, ann_file, image_dir, args.output_dir, id_to_label)

    summary = {
        "output_dir": str(args.output_dir),
        "synthetic_target_classes": list(SYNTHETIC_CLASSES),
        "counts": stats,
    }
    summary_path = args.output_dir / "stats.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved crop stats to {summary_path}")


if __name__ == "__main__":
    main()
