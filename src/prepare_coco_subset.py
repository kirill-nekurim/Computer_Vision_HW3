"""Build a 10-class COCO subset from full COCO 2017 annotations."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from src.config import COCO_DIR, COCO_SUBSET_DIR, SELECTED_CATEGORIES
from src.utils import build_id_mappings, save_label_map


def filter_coco_split(
    ann_path: Path,
    image_src_dir: Path | None,
    image_dst_dir: Path | None,
    old_to_new: dict[int, int],
    max_images: int | None = None,
    fraction: float | None = None,
    copy_images: bool = True,
) -> dict:
    with ann_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    selected_old_ids = set(old_to_new.keys())
    kept_anns = []
    image_ids_with_anns: set[int] = set()

    for ann in coco["annotations"]:
        old_cat = ann["category_id"]
        if old_cat not in selected_old_ids:
            continue
        kept_anns.append(
            {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": old_to_new[old_cat],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            }
        )
        image_ids_with_anns.add(ann["image_id"])

    kept_images = [img for img in coco["images"] if img["id"] in image_ids_with_anns]
    if fraction is not None:
        limit = max(1, int(len(kept_images) * fraction))
        kept_images = kept_images[:limit]
    if max_images is not None:
        kept_images = kept_images[:max_images]
    kept_image_ids = {img["id"] for img in kept_images}
    kept_anns = [ann for ann in kept_anns if ann["image_id"] in kept_image_ids]

    categories = [
        {"id": new_id, "name": name, "supercategory": "object"}
        for name, old_id in SELECTED_CATEGORIES.items()
        for new_id in [old_to_new[old_id]]
    ]

    if copy_images and image_src_dir and image_dst_dir:
        image_dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img in kept_images:
            src = image_src_dir / img["file_name"]
            dst = image_dst_dir / img["file_name"]
            if not src.exists():
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": kept_anns,
        "categories": categories,
    }
    return subset


def summarize_subset(name: str, subset: dict) -> None:
    per_class = defaultdict(int)
    for ann in subset["annotations"]:
        per_class[ann["category_id"]] += 1
    print(f"\n{name}:")
    print(f"  images: {len(subset['images'])}")
    print(f"  annotations: {len(subset['annotations'])}")
    print(f"  per-class counts: {dict(sorted(per_class.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 10-class COCO subset")
    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=COCO_DIR,
        help="Path to extracted COCO 2017 root (contains annotations/ and train2017/, val2017/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COCO_SUBSET_DIR,
        help="Where to write filtered annotations and copied images",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Optional cap for quick experiments",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Optional cap for quick experiments",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Use a fraction of filtered train images, e.g. 0.25 for 25%%",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Use a fraction of filtered val images, e.g. 0.25 for 25%%",
    )
    parser.add_argument(
        "--skip-image-copy",
        action="store_true",
        help="Only write annotation JSON (download images with scripts/download_subset_images.py)",
    )
    args = parser.parse_args()

    coco_dir = args.coco_dir
    out_dir = args.output_dir
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    old_to_new, _, _ = build_id_mappings()
    save_label_map(out_dir / "label_map.json")

    copy_images = not args.skip_image_copy
    src_train = coco_dir / "train2017" if copy_images else None
    src_val = coco_dir / "val2017" if copy_images else None
    dst_train = out_dir / "train2017" if copy_images else None
    dst_val = out_dir / "val2017" if copy_images else None

    train_subset = filter_coco_split(
        coco_dir / "annotations" / "instances_train2017.json",
        src_train,
        dst_train,
        old_to_new,
        max_images=args.max_train_images,
        fraction=args.train_fraction,
        copy_images=copy_images,
    )
    val_subset = filter_coco_split(
        coco_dir / "annotations" / "instances_val2017.json",
        src_val,
        dst_val,
        old_to_new,
        max_images=args.max_val_images,
        fraction=args.val_fraction,
        copy_images=copy_images,
    )

    train_ann_path = ann_dir / "instances_train_subset.json"
    val_ann_path = ann_dir / "instances_val_subset.json"
    train_ann_path.write_text(json.dumps(train_subset), encoding="utf-8")
    val_ann_path.write_text(json.dumps(val_subset), encoding="utf-8")

    summarize_subset("train", train_subset)
    summarize_subset("val", val_subset)
    print(f"\nSaved subset to {out_dir}")


if __name__ == "__main__":
    main()
