"""Fine-tune DETR on COCO subset with TensorBoard logging and profiler trace."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.config import (
    CHECKPOINT_DIR,
    COCO_SUBSET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_LR_BACKBONE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_WEIGHT_DECAY,
    METRICS_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    PLOTS_DIR,
    PROFILER_DIR,
    TENSORBOARD_DIR,
)
from src.dataset import CocoSubsetDataset, collate_fn
from src.evaluate import compute_coco_metrics, run_inference
from src.utils import ensure_dirs, load_label_map, save_label_map


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(processor: DetrImageProcessor, label_map: dict) -> DetrForObjectDetection:
    id_to_label = label_map["id_to_label"]
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id_to_label,
        label2id={v: k for k, v in id_to_label.items()},
        ignore_mismatched_sizes=True,
    )
    return model


def build_optimizer(model: DetrForObjectDetection, lr: float, lr_backbone: float, weight_decay: float):
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    return torch.optim.AdamW(param_dicts, weight_decay=weight_decay)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    writer,
    global_step,
    max_grad_norm: float,
    run_profiler=False,
):
    model.train()
    epoch_losses = {"loss": [], "loss_ce": [], "loss_bbox": [], "loss_giou": []}

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        def forward_backward():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            return outputs

        if run_profiler and batch_idx == 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            ensure_dirs(PROFILER_DIR)
            trace_path = PROFILER_DIR / "trace.json"
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                with torch.profiler.record_function("detr_train_step"):
                    outputs = forward_backward()
            prof.export_chrome_trace(str(trace_path))
            print(f"Profiler trace saved to {trace_path}")
        else:
            outputs = forward_backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        loss_dict = {
            "loss": float(outputs.loss.detach().cpu()),
            "loss_ce": float(outputs.loss_dict["loss_ce"].detach().cpu()),
            "loss_bbox": float(outputs.loss_dict["loss_bbox"].detach().cpu()),
            "loss_giou": float(outputs.loss_dict["loss_giou"].detach().cpu()),
        }

        for key, value in loss_dict.items():
            epoch_losses[key].append(value)
            writer.add_scalar(f"train/{key}", value, global_step)

        progress.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        global_step += 1

    avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}
    return avg, global_step


@torch.no_grad()
def validate_losses(model, dataloader, device):
    model.eval()
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    count = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        totals["loss"] += float(outputs.loss.cpu())
        totals["loss_ce"] += float(outputs.loss_dict["loss_ce"].cpu())
        totals["loss_bbox"] += float(outputs.loss_dict["loss_bbox"].cpu())
        totals["loss_giou"] += float(outputs.loss_dict["loss_giou"].cpu())
        count += 1

    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_map(
    model,
    processor,
    dataloader,
    device,
    ann_file: Path,
    score_threshold: float,
) -> dict:
    predictions = run_inference(model, processor, dataloader, device, score_threshold)
    return compute_coco_metrics(ann_file, predictions)


def append_val_metrics_csv(path: Path, epoch: int, losses: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "epoch": epoch,
        "val_loss": losses["loss"],
        "val_loss_ce": losses["loss_ce"],
        "val_loss_bbox": losses["loss_bbox"],
        "val_loss_giou": losses["loss_giou"],
        "mAP": metrics["mAP"],
        "mAP50": metrics["mAP50"],
        "mAP75": metrics["mAP75"],
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_training_dashboard(history: list[dict], output_path: Path) -> None:
    ensure_dirs(output_path.parent)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, [h["train"]["loss"] for h in history], marker="o", label="train")
    axes[0, 0].plot(epochs, [h["val"]["loss"] for h in history], marker="s", label="val")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h["train"]["loss_ce"] for h in history], marker="o", label="train")
    axes[0, 1].plot(epochs, [h["val"]["loss_ce"] for h in history], marker="s", label="val")
    axes[0, 1].set_title("Classification loss (loss_ce)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h["train"]["loss_bbox"] for h in history], marker="o", label="train bbox")
    axes[1, 0].plot(epochs, [h["train"]["loss_giou"] for h in history], marker="^", label="train giou")
    axes[1, 0].plot(epochs, [h["val"]["loss_bbox"] for h in history], marker="s", label="val bbox")
    axes[1, 0].plot(epochs, [h["val"]["loss_giou"] for h in history], marker="v", label="val giou")
    axes[1, 0].set_title("Bbox regression losses")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if history[0].get("metrics"):
        axes[1, 1].plot(epochs, [h["metrics"]["mAP"] for h in history], marker="o", label="mAP")
        axes[1, 1].plot(epochs, [h["metrics"]["mAP50"] for h in history], marker="s", label="mAP@50")
        axes[1, 1].plot(epochs, [h["metrics"]["mAP75"] for h in history], marker="^", label="mAP@75")
        best_idx = max(range(len(history)), key=lambda i: history[i]["metrics"]["mAP50"])
        best_epoch = history[best_idx]["epoch"]
        best_map50 = history[best_idx]["metrics"]["mAP50"]
        axes[1, 1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].annotate(
            f"best mAP50={best_map50:.3f}\n(epoch {best_epoch})",
            xy=(best_epoch, best_map50),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
        )
        axes[1, 1].set_title("Validation mAP (COCO)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "mAP not computed\n(use without --skip-map-eval)", ha="center", va="center")

    fig.suptitle("DETR fine-tuning — training dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss_history(history: list[dict], output_path: Path) -> None:
    """Backward-compatible alias: saves dashboard to losses.png."""
    plot_training_dashboard(history, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DETR on COCO subset")
    parser.add_argument("--data-dir", type=Path, default=COCO_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr-backbone", type=float, default=DEFAULT_LR_BACKBONE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument(
        "--skip-map-eval",
        action="store_true",
        help="Skip per-epoch mAP (faster smoke test)",
    )
    args = parser.parse_args()

    ensure_dirs(args.output_dir, TENSORBOARD_DIR, PROFILER_DIR, PLOTS_DIR, METRICS_DIR)

    label_map_path = args.data_dir / "label_map.json"
    if not label_map_path.exists():
        save_label_map(label_map_path)
    label_map = load_label_map(label_map_path)

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    model = build_model(processor, label_map)
    device = get_device()
    model.to(device)
    print(f"Using device: {device}")

    train_dataset = CocoSubsetDataset(
        args.data_dir / "train2017",
        args.data_dir / "annotations" / "instances_train_subset.json",
        processor,
    )
    val_dataset = CocoSubsetDataset(
        args.data_dir / "val2017",
        args.data_dir / "annotations" / "instances_val_subset.json",
        processor,
    )

    if args.max_train_samples:
        train_dataset.ids = train_dataset.ids[: args.max_train_samples]
    if args.max_val_samples:
        val_dataset.ids = val_dataset.ids[: args.max_val_samples]

    collate = partial(collate_fn, processor=processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    val_ann_file = args.data_dir / "annotations" / "instances_val_subset.json"
    val_metrics_csv = METRICS_DIR / "val_metrics.csv"
    if val_metrics_csv.exists():
        val_metrics_csv.unlink()

    optimizer = build_optimizer(model, args.lr, args.lr_backbone, args.weight_decay)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

    history = []
    global_step = 0
    best_map50 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_avg, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            writer,
            global_step,
            max_grad_norm=args.max_grad_norm,
            run_profiler=(epoch == 1),
        )
        val_avg = validate_losses(model, val_loader, device)

        for key, value in val_avg.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        if args.skip_map_eval:
            metrics = {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}
        else:
            print(f"Computing mAP on val set (epoch {epoch})...")
            metrics = validate_map(
                model,
                processor,
                val_loader,
                device,
                val_ann_file,
                args.score_threshold,
            )
            for key, value in metrics.items():
                if key.startswith("mAP"):
                    writer.add_scalar(f"val/{key}", value, epoch)

        append_val_metrics_csv(val_metrics_csv, epoch, val_avg, metrics)

        record = {"epoch": epoch, "train": train_avg, "val": val_avg, "metrics": metrics}
        history.append(record)
        print(
            f"Epoch {epoch}: train_loss={train_avg['loss']:.4f}, "
            f"val_loss={val_avg['loss']:.4f}, "
            f"mAP={metrics['mAP']:.4f}, mAP50={metrics['mAP50']:.4f}"
        )

        epoch_dir = args.output_dir / f"epoch_{epoch:03d}"
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        if not args.skip_map_eval and metrics["mAP50"] > best_map50:
            best_map50 = metrics["mAP50"]
            best_epoch = epoch
            best_dir = args.output_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  -> New best checkpoint (mAP50={best_map50:.4f}) saved to {best_dir}")

    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    plot_training_dashboard(history, PLOTS_DIR / "training_dashboard.png")
    plot_loss_history(history, PLOTS_DIR / "losses.png")
    (METRICS_DIR / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    hparams = vars(args)
    hparams["device"] = str(device)
    hparams["best_epoch"] = best_epoch
    hparams["best_map50"] = best_map50
    (METRICS_DIR / "hparams.json").write_text(json.dumps(hparams, indent=2, default=str), encoding="utf-8")

    writer.close()
    print(f"Training complete. Checkpoints: {args.output_dir}")
    if not args.skip_map_eval:
        print(f"Best epoch: {best_epoch} (mAP50={best_map50:.4f}) -> {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
