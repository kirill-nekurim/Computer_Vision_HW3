"""PyTorch dataset for COCO subset compatible with HuggingFace DETR."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import DetrImageProcessor

class CocoSubsetDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        ann_file: Path,
        processor: DetrImageProcessor,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(ann_file))
        self.processor = processor
        self.ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")
        width, height = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        class_labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            # DetrImageProcessor expects COCO [x_min, y_min, width, height] (top-left corner).
            boxes.append(ann["bbox"])
            class_labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "image_id": img_id,
                    "category_id": class_labels[i],
                    "bbox": boxes[i],
                    "area": areas[i],
                    "iscrowd": iscrowd[i],
                }
                for i in range(len(class_labels))
            ],
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_id": img_id,
            "orig_size": torch.tensor([height, width]),
        }


def collate_fn(batch: list[dict], processor: DetrImageProcessor) -> dict:
    del processor  # padding is done manually for transformers 4.x / 5.x compatibility
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    orig_sizes = torch.stack([item["orig_size"] for item in batch])

    max_h = max(pv.shape[-2] for pv in pixel_values)
    max_w = max(pv.shape[-1] for pv in pixel_values)

    padded_images: list[torch.Tensor] = []
    pixel_masks: list[torch.Tensor] = []
    for pv in pixel_values:
        _, h, w = pv.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h or pad_w:
            pv = torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h))
        padded_images.append(pv)
        mask = torch.zeros(max_h, max_w, dtype=torch.int64)
        mask[:h, :w] = 1
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_images),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
    }
