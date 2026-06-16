"""Visualization helpers for DETR predictions and error analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def save_detection_image(
    image_path: Path,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    id_to_label: dict,
    output_path: Path,
    title: str = "",
) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    for gt in gt_boxes:
        x, y, w, h = gt["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        label = id_to_label.get(gt["category_id"], str(gt["category_id"]))
        ax.text(x, y - 4, f"GT: {label}", color="lime", fontsize=9, backgroundcolor="black")

    for pred in pred_boxes:
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        label = id_to_label.get(pred["category_id"], str(pred["category_id"]))
        score = pred.get("score", 1.0)
        ax.text(x, y + h + 12, f"Pred: {label} ({score:.2f})", color="red", fontsize=9, backgroundcolor="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def draw_error_examples(
    cls_examples: list[dict],
    loc_examples: list[dict],
    coco,
    image_dir: Path,
    id_to_label: dict,
    output_dir: Path,
) -> None:
    for idx, err in enumerate(cls_examples):
        img_info = coco.loadImgs(err["image_id"])[0]
        image_path = image_dir / img_info["file_name"]
        title = (
            f"Classification error: GT={err['gt_label']}, Pred={err['pred_label']}, "
            f"IoU={err['iou']:.2f}"
        )
        save_detection_image(
            image_path,
            [err["gt"]],
            [err["pred"]],
            id_to_label,
            output_dir / f"classification_error_{idx:02d}.png",
            title=title,
        )

    for idx, err in enumerate(loc_examples):
        img_info = coco.loadImgs(err["image_id"])[0]
        image_path = image_dir / img_info["file_name"]
        label = id_to_label[err["gt"]["category_id"]]
        title = f"Localization error: class={label}, IoU={err['iou']:.2f}"
        save_detection_image(
            image_path,
            [err["gt"]],
            [err["pred"]],
            id_to_label,
            output_dir / f"localization_error_{idx:02d}.png",
            title=title,
        )
