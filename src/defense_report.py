"""Helpers for defense-ready reports and notebook visualizations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from pycocotools.coco import COCO
from src.utils import load_detr_processor, load_label_map, xyxy_to_xywh


def dataset_class_distribution(data_dir: Path) -> dict:
    """Count boxes per class in train/val subset annotations."""
    label_map = load_label_map(data_dir / "label_map.json")
    id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}
    result = {"classes": [], "train_total_boxes": 0, "val_total_boxes": 0}

    for split, fname in [("train", "instances_train_subset.json"), ("val", "instances_val_subset.json")]:
        ann_path = data_dir / "annotations" / fname
        if not ann_path.exists():
            continue
        coco = COCO(str(ann_path))
        counts = Counter()
        for ann in coco.dataset["annotations"]:
            counts[ann["category_id"]] += 1
        for cat_id, count in sorted(counts.items()):
            name = id_to_label.get(cat_id, str(cat_id))
            entry = next((c for c in result["classes"] if c["name"] == name), None)
            if entry is None:
                entry = {"name": name, "category_id": cat_id, "train_boxes": 0, "val_boxes": 0}
                result["classes"].append(entry)
            key = f"{split}_boxes"
            entry[key] = count
            result[f"{split}_total_boxes"] += count

    result["classes"].sort(key=lambda c: c["name"])
    return result


def plot_class_distribution(stats: dict, output_path: Path | None = None) -> plt.Figure:
    names = [c["name"] for c in stats["classes"]]
    train_counts = [c.get("train_boxes", 0) for c in stats["classes"]]
    val_counts = [c.get("val_boxes", 0) for c in stats["classes"]]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width / 2 for i in x], train_counts, width, label="train", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], val_counts, width, label="val", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Number of boxes")
    ax.set_title(
        f"COCO-subset class distribution "
        f"(train={stats['train_total_boxes']}, val={stats['val_total_boxes']} boxes)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def verify_bbox_encoding(data_dir: Path, processor=None) -> dict:
    """
    Sanity check: COCO bbox must be passed as [x, y, w, h] top-left to DetrImageProcessor.
    Returns expected vs actual normalized center for one val sample.
    """
    if processor is None:
        processor = load_detr_processor("facebook/detr-resnet-50")

    ann_file = data_dir / "annotations" / "instances_val_subset.json"
    img_dir = data_dir / "val2017"
    coco = COCO(str(ann_file))
    img_id = coco.getImgIds()[0]
    img_info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    image = Image.open(img_dir / img_info["file_name"]).convert("RGB")
    w, h = image.size
    gt_bbox = anns[0]["bbox"]

    target = {
        "image_id": img_id,
        "annotations": [
            {
                "image_id": img_id,
                "category_id": anns[0]["category_id"],
                "bbox": gt_bbox,
                "area": anns[0]["area"],
                "iscrowd": 0,
            }
        ],
    }
    encoding = processor(images=image, annotations=target, return_tensors="pt")
    encoded = encoding["labels"][0]["boxes"].squeeze().tolist()
    if isinstance(encoded[0], list):
        encoded = encoded[0]

    x, y, bw, bh = gt_bbox
    expected = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]

    ok = all(abs(a - b) < 1e-3 for a, b in zip(encoded, expected))
    return {
        "image_id": img_id,
        "file_name": img_info["file_name"],
        "coco_bbox_xywh": gt_bbox,
        "encoded_cxcywh_norm": encoded,
        "expected_cxcywh_norm": expected,
        "bbox_encoding_ok": ok,
    }


def load_training_dashboard(plots_dir: Path) -> plt.Figure | None:
    path = plots_dir / "training_dashboard.png"
    if not path.exists():
        return None
    return plt.imread(str(path))


def metrics_summary_table(metrics_dir: Path) -> list[dict]:
    """Rows for a defense summary table from saved artifacts."""
    rows = []
    hparams_path = metrics_dir / "hparams.json"
    val_metrics_path = metrics_dir / "val_metrics.csv"
    history_path = metrics_dir / "train_history.json"

    if hparams_path.exists():
        hp = json.loads(hparams_path.read_text(encoding="utf-8"))
        rows.append({"metric": "epochs", "value": hp.get("epochs")})
        rows.append({"metric": "batch_size", "value": hp.get("batch_size")})
        rows.append({"metric": "lr (head)", "value": hp.get("lr")})
        rows.append({"metric": "lr (backbone)", "value": hp.get("lr_backbone")})
        rows.append({"metric": "best epoch", "value": hp.get("best_epoch")})
        rows.append({"metric": "best mAP@50", "value": f"{hp.get('best_map50', 0):.4f}"})

    if val_metrics_path.exists():
        lines = val_metrics_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) > 1:
            last = lines[-1].split(",")
            header = lines[0].split(",")
            last_row = dict(zip(header, last))
            rows.append({"metric": "final mAP", "value": f"{float(last_row.get('mAP', 0)):.4f}"})
            rows.append({"metric": "final mAP@50", "value": f"{float(last_row.get('mAP50', 0)):.4f}"})

    if history_path.exists() and not any(r["metric"] == "final mAP@50" for r in rows):
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history and history[-1].get("metrics"):
            m = history[-1]["metrics"]
            rows.append({"metric": "final mAP@50", "value": f"{m.get('mAP50', 0):.4f}"})

    return rows


@torch.no_grad()
def draw_prediction_gallery(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    num_images: int = 6,
    score_threshold: float = 0.7,
) -> list[Path]:
    """Save GT (green) vs predictions (red) for random val images."""
    from src.dataset import CocoSubsetDataset, collate_fn
    from src.train import get_device
    from torch.utils.data import DataLoader
    from functools import partial
    from transformers import DetrForObjectDetection

    label_map = load_label_map(data_dir / "label_map.json")
    id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}

    processor = load_detr_processor(checkpoint)
    model = DetrForObjectDetection.from_pretrained(checkpoint)
    device = get_device()
    model.to(device)
    model.eval()

    ann_file = data_dir / "annotations" / "instances_val_subset.json"
    image_dir = data_dir / "val2017"
    coco = COCO(str(ann_file))
    dataset = CocoSubsetDataset(image_dir, ann_file, processor)
    dataset.ids = dataset.ids[:num_images]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for batch in loader:
        image_id = batch["image_ids"][0]
        img_info = coco.loadImgs(image_id)[0]
        image_path = image_dir / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=batch["orig_sizes"].tolist(),
        )[0]

        gt_ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
        gt_anns = coco.loadAnns(gt_ann_ids)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"image_id={image_id}  (green=GT, red=pred, thr={score_threshold})")

        for gt in gt_anns:
            x, y, w, h = gt["bbox"]
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"))
            label = id_to_label.get(gt["category_id"], "?")
            ax.text(x, max(y - 4, 0), f"GT:{label}", color="lime", fontsize=8, backgroundcolor="black")

        for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
            x, y, w, h = xyxy_to_xywh([float(v) for v in box.tolist()])
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"))
            name = id_to_label.get(int(label.item()), "?")
            ax.text(x, y + h + 10, f"{name} {score:.2f}", color="red", fontsize=8, backgroundcolor="black")

        out_path = output_dir / f"gallery_{image_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved
