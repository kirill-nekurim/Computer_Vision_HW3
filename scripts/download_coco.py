#!/usr/bin/env python3
"""Download COCO 2017 annotations and image archives."""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.config import COCO_DIR
from src.utils import ensure_dirs

URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

MIN_SIZES = {
    "annotations_trainval2017.zip": 200 * 1024 * 1024,
    "train2017.zip": 15 * 1024 * 1024 * 1024,
    "val2017.zip": 500 * 1024 * 1024,
}


def _parse_total_size(response: requests.Response, downloaded: int) -> int:
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[-1])
    content_length = response.headers.get("Content-Length")
    if content_length:
        size = int(content_length)
        return downloaded + size if response.status_code == 206 else size
    return 0


def download(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 50,
    connect_timeout: int = 30,
    read_timeout: int = 300,
) -> None:
    min_size = MIN_SIZES.get(dest.name, 1024)
    ensure_dirs(dest.parent)

    if dest.exists() and dest.stat().st_size >= min_size:
        print(f"Already exists: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"Downloading {url}")
    print(f"  -> {dest}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (hw3-detr downloader)"})

    for attempt in range(1, max_retries + 1):
        downloaded = dest.stat().st_size if dest.exists() else 0
        headers = {}
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"Resume attempt {attempt}/{max_retries} from {downloaded / 1e9:.2f} GB")

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as response:
                if response.status_code == 416:
                    break

                if downloaded > 0 and response.status_code not in (206, 200):
                    response.raise_for_status()

                if downloaded == 0:
                    response.raise_for_status()

                total = _parse_total_size(response, downloaded)
                mode = "ab" if downloaded > 0 and response.status_code == 206 else "wb"
                if mode == "wb" and downloaded > 0:
                    downloaded = 0

                with tqdm(
                    total=total or None,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    with dest.open(mode) as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            bar.update(len(chunk))

            if dest.exists() and dest.stat().st_size >= min_size:
                print(f"Done: {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
                return

        except (requests.RequestException, OSError) as exc:
            wait_s = min(60, 2 ** min(attempt, 6))
            partial = dest.stat().st_size if dest.exists() else 0
            print(f"Attempt {attempt} failed: {exc}")
            print(f"  partial size: {partial / 1e6:.1f} MB, retry in {wait_s}s")
            time.sleep(wait_s)

    final_size = dest.stat().st_size if dest.exists() else 0
    if final_size < min_size:
        raise RuntimeError(
            f"Download failed after {max_retries} attempts: {dest.name} "
            f"is {final_size / 1e6:.1f} MB (expected >= {min_size / 1e9:.1f} GB).\n"
            "Tips:\n"
            "  - retry the same command (resume is supported)\n"
            "  - try: python scripts/download_coco.py --val-only   (~1 GB)\n"
            "  - or download in Colab notebook instead"
        )


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    marker = dest_dir / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"Already extracted: {zip_path.name}")
        return
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO 2017")
    parser.add_argument("--output-dir", type=Path, default=COCO_DIR)
    parser.add_argument("--annotations-only", action="store_true", help="Skip image zips (~20 GB)")
    parser.add_argument("--val-only", action="store_true", help="Download only val2017 (~1 GB)")
    parser.add_argument("--retries", type=int, default=50, help="Max retry attempts per file")
    args = parser.parse_args()

    out = args.output_dir
    ensure_dirs(out)

    ann_zip = out / "annotations_trainval2017.zip"
    download(URLS["annotations"], ann_zip, max_retries=args.retries)
    extract_zip(ann_zip, out)

    if args.annotations_only:
        print("Annotations ready. Download images separately when needed.")
        return

    image_keys = ["val2017"] if args.val_only else ["train2017", "val2017"]
    for key in image_keys:
        zip_path = out / f"{key}.zip"
        download(URLS[key], zip_path, max_retries=args.retries)
        extract_zip(zip_path, out)


if __name__ == "__main__":
    main()
