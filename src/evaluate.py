"""Evaluate fine-tuned DETR: mAP and mAP@0.5 via pycocotools."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.config import (
    CHECKPOINT_DIR,
    COCO_SUBSET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SCORE_THRESHOLD,
    METRICS_DIR,
)
from src.dataset import CocoSubsetDataset, collate_fn
from src.utils import ensure_dirs, load_label_map, xyxy_to_xywh


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def box_cxcywh_to_xywh(box: torch.Tensor, width: int, height: int) -> list[float]:
    cx, cy, w, h = box.tolist()
    x = (cx - w / 2) * width
    y = (cy - h / 2) * height
    return [x, y, w * width, h * height]


@torch.no_grad()
def run_inference(
    model,
    processor,
    dataloader,
    device,
    score_threshold: float,
) -> list[dict]:
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="Inference"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = batch["orig_sizes"].tolist()
        processed = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )

        for image_id, preds in zip(batch["image_ids"], processed):
            for score, label, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(label.item()),
                        "bbox": xyxy_to_xywh([float(v) for v in box.tolist()]),
                        "score": float(score.item()),
                    }
                )
    return results


def compute_coco_metrics(ann_file: Path, predictions: list[dict]) -> dict:
    coco_gt = COCO(str(ann_file))
    if not predictions:
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP": float(stats[0]),
        "mAP50": float(stats[1]),
        "mAP75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DETR on COCO subset")
    parser.add_argument("--data-dir", type=Path, default=COCO_SUBSET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "final")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs(METRICS_DIR)

    label_map = load_label_map(args.data_dir / "label_map.json")
    processor = DetrImageProcessor.from_pretrained(args.checkpoint)
    model = DetrForObjectDetection.from_pretrained(args.checkpoint)
    device = get_device()
    model.to(device)

    if args.split == "val":
        image_dir = args.data_dir / "val2017"
        ann_file = args.data_dir / "annotations" / "instances_val_subset.json"
    else:
        image_dir = args.data_dir / "train2017"
        ann_file = args.data_dir / "annotations" / "instances_train_subset.json"

    dataset = CocoSubsetDataset(image_dir, ann_file, processor)
    if args.max_samples:
        dataset.ids = dataset.ids[: args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, processor=processor),
    )

    predictions = run_inference(model, processor, loader, device, args.score_threshold)
    metrics = compute_coco_metrics(ann_file, predictions)

    pred_path = METRICS_DIR / f"predictions_{args.split}.json"
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    metrics_path = METRICS_DIR / f"metrics_{args.split}.json"
    payload = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "score_threshold": args.score_threshold,
        "num_classes": label_map["num_classes"],
        **metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nMetrics:")
    print(f"  mAP:   {metrics['mAP']:.4f}")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()
