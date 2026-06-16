"""Train ResNet classifier for HW3.5 ablation (baseline vs synthetic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from src.cls_dataset import MixedClassificationDataset, build_transforms, CropClassificationDataset
from src.config import COCO_SUBSET_DIR
from src.hw35_config import (
    ABLATION_PATH,
    CLASSIFIER_BASELINE_DIR,
    CLASSIFIER_MODEL,
    CLASSIFIER_SYNTHETIC_DIR,
    CLS_CROPS_DIR,
    DEFAULT_CLS_BATCH_SIZE,
    DEFAULT_CLS_EPOCHS,
    DEFAULT_CLS_IMAGE_SIZE,
    DEFAULT_CLS_LR,
    DEFAULT_NUM_WORKERS,
    HW35_DIR,
    SYNTHETIC_CLASSES,
    SYNTHETIC_DIR,
    SYNTHETIC_VIZ_PATH,
)
from src.utils import ensure_dirs, load_label_map


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, model_name: str = CLASSIFIER_MODEL) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "classification_report": report,
    }
    return metrics


def train_one_run(
    use_synthetic: bool,
    crops_dir: Path,
    synthetic_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    label_map = load_label_map(COCO_SUBSET_DIR / "label_map.json")
    class_to_idx = label_map["label_to_id"]
    class_names = [label_map["id_to_label"][i] for i in range(label_map["num_classes"])]

    train_transform = build_transforms(image_size, train=True)
    val_transform = build_transforms(image_size, train=False)

    if use_synthetic:
        train_ds = MixedClassificationDataset(
            real_root=crops_dir / "train",
            synthetic_root=synthetic_dir,
            synthetic_classes=SYNTHETIC_CLASSES,
            class_to_idx=class_to_idx,
            transform=train_transform,
        )
        run_name = "with_synthetic"
    else:
        train_ds = CropClassificationDataset(
            crops_dir / "train",
            class_to_idx,
            transform=train_transform,
        )
        run_name = "baseline"

    val_ds = CropClassificationDataset(
        crops_dir / "val",
        class_to_idx,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    ensure_dirs(output_dir)
    history: list[dict] = []
    best_val_acc = -1.0
    best_metrics: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_metrics = evaluate(model, val_loader, device, class_names)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_info)
        print(
            f"[{run_name}] epoch {epoch}: "
            f"train_acc={train_acc:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "use_synthetic": use_synthetic,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )

    assert best_metrics is not None
    result = {
        "run_name": run_name,
        "use_synthetic": use_synthetic,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "synthetic_classes": list(SYNTHETIC_CLASSES) if use_synthetic else [],
        "best_val_accuracy": best_val_acc,
        "best_metrics": {
            "accuracy": best_metrics["accuracy"],
            "macro_f1": best_metrics["macro_f1"],
            "per_class": best_metrics["per_class"],
        },
        "history": history,
    }

    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def plot_synthetic_examples(synthetic_dir: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(len(SYNTHETIC_CLASSES), 4, figsize=(12, 3 * len(SYNTHETIC_CLASSES)))
    if len(SYNTHETIC_CLASSES) == 1:
        axes = np.array([axes])

    for row, class_name in enumerate(SYNTHETIC_CLASSES):
        paths = sorted((synthetic_dir / class_name).glob("*.jpg"))[:4]
        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths):
                img = plt.imread(paths[col])
                ax.imshow(img)
            if col == 0:
                ax.set_title(class_name, loc="left", fontsize=11)

    plt.suptitle("Synthetic examples (Stable Diffusion + ControlNet Canny)", fontsize=13)
    plt.tight_layout()
    ensure_dirs(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_ablation_table(baseline: dict, with_synthetic: dict) -> dict:
    rows = []
    for result in (baseline, with_synthetic):
        rare_recall = {
            cls: result["best_metrics"]["per_class"][cls]["recall"]
            for cls in SYNTHETIC_CLASSES
        }
        rows.append(
            {
                "experiment": result["run_name"],
                "use_synthetic": result["use_synthetic"],
                "train_samples": result["train_samples"],
                "accuracy": result["best_metrics"]["accuracy"],
                "macro_f1": result["best_metrics"]["macro_f1"],
                "recall_truck": rare_recall["truck"],
                "recall_bicycle": rare_recall["bicycle"],
                "recall_chair": rare_recall["chair"],
                "mean_recall_rare": float(np.mean(list(rare_recall.values()))),
            }
        )

    delta = {
        "accuracy": rows[1]["accuracy"] - rows[0]["accuracy"],
        "macro_f1": rows[1]["macro_f1"] - rows[0]["macro_f1"],
        "mean_recall_rare": rows[1]["mean_recall_rare"] - rows[0]["mean_recall_rare"],
    }

    return {"rows": rows, "delta_with_synthetic_minus_baseline": delta}


def main() -> None:
    parser = argparse.ArgumentParser(description="HW3.5 classifier ablation.")
    parser.add_argument("--crops-dir", type=Path, default=CLS_CROPS_DIR)
    parser.add_argument("--synthetic-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=HW35_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CLS_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CLS_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_CLS_LR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_CLS_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    ensure_dirs(args.output_dir)

    results: dict[str, dict] = {}

    if not args.synthetic_only:
        results["baseline"] = train_one_run(
            use_synthetic=False,
            crops_dir=args.crops_dir,
            synthetic_dir=args.synthetic_dir,
            output_dir=args.output_dir / "baseline",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            num_workers=args.num_workers,
            device=device,
        )

    if not args.baseline_only:
        synthetic_exists = all(
            (args.synthetic_dir / cls).exists()
            and any((args.synthetic_dir / cls).glob("*.jpg"))
            for cls in SYNTHETIC_CLASSES
        )
        if not synthetic_exists:
            raise FileNotFoundError(
                f"Synthetic images not found in {args.synthetic_dir}. "
                "Run: python scripts/generate_synthetic.py"
            )
        results["with_synthetic"] = train_one_run(
            use_synthetic=True,
            crops_dir=args.crops_dir,
            synthetic_dir=args.synthetic_dir,
            output_dir=args.output_dir / "with_synthetic",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            num_workers=args.num_workers,
            device=device,
        )

    if "baseline" in results and "with_synthetic" in results:
        ablation = build_ablation_table(results["baseline"], results["with_synthetic"])
        args.output_dir.joinpath("ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
        print("\n=== Ablation ===")
        for row in ablation["rows"]:
            print(row)
        print("Delta:", ablation["delta_with_synthetic_minus_baseline"])

    if args.synthetic_dir.exists():
        plot_synthetic_examples(args.synthetic_dir, SYNTHETIC_VIZ_PATH)
        print(f"Synthetic gallery saved to {SYNTHETIC_VIZ_PATH}")


if __name__ == "__main__":
    main()
