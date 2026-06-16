"""Error analysis: classification vs localization mistakes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.config import (
    CHECKPOINT_DIR,
    COCO_SUBSET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SCORE_THRESHOLD,
    METRICS_DIR,
    VIS_DIR,
)
from src.dataset import CocoSubsetDataset, collate_fn
from src.utils import ensure_dirs, load_label_map, xyxy_to_xywh
from src.visualize import draw_error_examples, save_detection_image


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    return inter / (area_a + area_b - inter + 1e-6)


def analyze_image(gt_anns, pred_anns, iou_threshold: float, id_to_label: dict) -> dict:
    matched_gt = set()
    matched_pred = set()
    classification_errors = []
    localization_errors = []
    true_positives = []

    for pi, pred in enumerate(pred_anns):
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_anns):
            if gi in matched_gt:
                continue
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi < 0:
            continue

        gt = gt_anns[best_gi]
        if best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            if pred["category_id"] == gt["category_id"]:
                true_positives.append({"pred": pred, "gt": gt, "iou": best_iou})
            else:
                classification_errors.append(
                    {
                        "pred": pred,
                        "gt": gt,
                        "iou": best_iou,
                        "pred_label": id_to_label[pred["category_id"]],
                        "gt_label": id_to_label[gt["category_id"]],
                    }
                )
        elif pred["category_id"] == gt["category_id"]:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            localization_errors.append(
                {
                    "pred": pred,
                    "gt": gt,
                    "iou": best_iou,
                    "pred_label": id_to_label[pred["category_id"]],
                    "gt_label": id_to_label[gt["category_id"]],
                }
            )

    false_positives = [pred_anns[i] for i in range(len(pred_anns)) if i not in matched_pred]
    false_negatives = [gt_anns[i] for i in range(len(gt_anns)) if i not in matched_gt]

    return {
        "classification_errors": classification_errors,
        "localization_errors": localization_errors,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


@torch.no_grad()
def collect_predictions(model, processor, dataloader, device, score_threshold):
    model.eval()
    per_image = {}

    for batch in tqdm(dataloader, desc="Collecting predictions"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=batch["orig_sizes"].tolist(),
        )

        for image_id, preds in zip(batch["image_ids"], processed):
            items = []
            for score, label, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
                items.append(
                    {
                        "category_id": int(label.item()),
                        "bbox": xyxy_to_xywh([float(v) for v in box.tolist()]),
                        "score": float(score.item()),
                    }
                )
            per_image[int(image_id)] = items

    return per_image


def main() -> None:
    parser = argparse.ArgumentParser(description="DETR error analysis")
    parser.add_argument("--data-dir", type=Path, default=COCO_SUBSET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "final")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-examples", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs(METRICS_DIR, VIS_DIR)

    label_map = load_label_map(args.data_dir / "label_map.json")
    id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}

    processor = DetrImageProcessor.from_pretrained(args.checkpoint)
    model = DetrForObjectDetection.from_pretrained(args.checkpoint)
    from src.train import get_device

    device = get_device()
    model.to(device)

    ann_file = args.data_dir / "annotations" / "instances_val_subset.json"
    image_dir = args.data_dir / "val2017"
    coco = COCO(str(ann_file))

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

    predictions = collect_predictions(model, processor, loader, device, args.score_threshold)

    totals = Counter()
    per_class_cls = Counter()
    per_class_loc = Counter()
    cls_examples = []
    loc_examples = []

    for img_id in dataset.ids:
        gt_ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        gt_anns = coco.loadAnns(gt_ann_ids)
        pred_anns = predictions.get(img_id, [])

        result = analyze_image(gt_anns, pred_anns, args.iou_threshold, id_to_label)
        totals["classification_errors"] += len(result["classification_errors"])
        totals["localization_errors"] += len(result["localization_errors"])
        totals["true_positives"] += len(result["true_positives"])
        totals["false_positives"] += len(result["false_positives"])
        totals["false_negatives"] += len(result["false_negatives"])

        for err in result["classification_errors"]:
            per_class_cls[err["gt_label"]] += 1
            if len(cls_examples) < args.num_examples:
                cls_examples.append({"image_id": img_id, **err})

        for err in result["localization_errors"]:
            per_class_loc[err["gt_label"]] += 1
            if len(loc_examples) < args.num_examples:
                loc_examples.append({"image_id": img_id, **err})

    summary = {
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "totals": dict(totals),
        "per_class_classification_errors": dict(per_class_cls),
        "per_class_localization_errors": dict(per_class_loc),
    }
    summary_path = METRICS_DIR / "error_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    draw_error_examples(
        cls_examples,
        loc_examples,
        coco,
        image_dir,
        id_to_label,
        VIS_DIR,
    )

    print("\nError analysis summary:")
    for key, value in totals.items():
        print(f"  {key}: {value}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
