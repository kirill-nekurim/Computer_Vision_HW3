#!/usr/bin/env python3
"""Download only images listed in COCO subset annotation files (~MBs, not ~20GB)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import COCO_SUBSET_DIR
from src.utils import ensure_dirs

IMAGE_BASE = {
    "train2017": "http://images.cocodataset.org/train2017",
    "val2017": "http://images.cocodataset.org/val2017",
}


def download_image(url: str, dest: Path, session: requests.Session, retries: int = 5) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        return True

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
            return True
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"Failed: {dest.name} ({exc})")
                return False
            time.sleep(min(30, 2**attempt))
    return False


def download_split(ann_file: Path, split: str, subset_dir: Path) -> dict:
    with ann_file.open(encoding="utf-8") as f:
        coco = json.load(f)

    image_dir = subset_dir / split
    ensure_dirs(image_dir)
    base_url = IMAGE_BASE[split]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (hw3-detr subset downloader)"})

    images = coco["images"]
    ok, skip, fail = 0, 0, 0

    bar = tqdm(
        images,
        desc=split,
        unit="img",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    for img in bar:
        dest = image_dir / img["file_name"]
        if dest.exists() and dest.stat().st_size > 1024:
            skip += 1
            bar.set_postfix(new=ok, skip=skip, fail=fail, refresh=False)
            continue
        url = f"{base_url}/{img['file_name']}"
        if download_image(url, dest, session):
            ok += 1
        else:
            fail += 1
        bar.set_postfix(new=ok, skip=skip, fail=fail, refresh=False)

    summary = {
        "split": split,
        "total": len(images),
        "downloaded": ok,
        "skipped": skip,
        "failed": fail,
    }
    print(
        f"{split}: {ok + skip}/{len(images)} ready "
        f"(new={ok}, already={skip}, failed={fail})"
    )
    return summary


def download_all(subset_dir: Path = COCO_SUBSET_DIR) -> dict:
    ann_dir = subset_dir / "annotations"
    train_ann = ann_dir / "instances_train_subset.json"
    val_ann = ann_dir / "instances_val_subset.json"

    if not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError(
            "Subset annotations not found. Run prepare_coco_subset first."
        )

    with train_ann.open(encoding="utf-8") as f:
        n_train = len(json.load(f)["images"])
    with val_ann.open(encoding="utf-8") as f:
        n_val = len(json.load(f)["images"])

    print(f"Images to process: train={n_train}, val={n_val}, total={n_train + n_val}")

    train_stats = download_split(train_ann, "train2017", subset_dir)
    val_stats = download_split(val_ann, "val2017", subset_dir)

    total = {
        "total": train_stats["total"] + val_stats["total"],
        "downloaded": train_stats["downloaded"] + val_stats["downloaded"],
        "skipped": train_stats["skipped"] + val_stats["skipped"],
        "failed": train_stats["failed"] + val_stats["failed"],
    }
    ready = total["downloaded"] + total["skipped"]
    print(
        f"\nDone: {ready}/{total['total']} images ready "
        f"(new={total['downloaded']}, already={total['skipped']}, failed={total['failed']})"
    )
    print(f"Folder: {subset_dir}")
    return {"train": train_stats, "val": val_stats, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only images referenced in subset annotations"
    )
    parser.add_argument("--subset-dir", type=Path, default=COCO_SUBSET_DIR)
    args = parser.parse_args()
    download_all(args.subset_dir)


if __name__ == "__main__":
    main()
