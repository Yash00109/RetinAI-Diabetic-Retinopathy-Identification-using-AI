from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from dr_detection.config import load_config
from dr_detection.dataset import AptosDataset, filter_frame_by_quality, make_splits
from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model
from dr_detection.quality import QualityThresholds
from dr_detection.transforms import build_transforms


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights_from_labels(labels, num_classes: int):
    import torch

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    import torch
    from tqdm import tqdm

    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        losses.append(float(loss.detach().cpu()))
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.detach().cpu().tolist())

    return float(np.mean(losses)), classification_metrics(y_true, y_pred, num_classes=model.num_classes)


def attach_num_classes(model, num_classes: int):
    model.num_classes = num_classes
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet_b0.json")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, val_frame, test_frame = make_splits(cfg.data, cfg.seed)
    thresholds = QualityThresholds(**cfg.quality.__dict__)
    rejected_frames = []
    if cfg.data.quality_filter:
        train_frame, rejected_train = filter_frame_by_quality(
            train_frame,
            cfg.data.image_dir,
            cfg.data.image_col,
            cfg.data.image_ext,
            thresholds,
        )
        val_frame, rejected_val = filter_frame_by_quality(
            val_frame,
            cfg.data.image_dir,
            cfg.data.image_col,
            cfg.data.image_ext,
            thresholds,
        )
        rejected_train["split"] = "train"
        rejected_val["split"] = "val"
        rejected_frames.extend([rejected_train, rejected_val])

    train_frame.to_csv(output_dir / "train_split.csv", index=False)
    val_frame.to_csv(output_dir / "val_split.csv", index=False)
    test_frame.to_csv(output_dir / "test_split.csv", index=False)
    if rejected_frames:
        import pandas as pd

        pd.concat(rejected_frames, ignore_index=True).to_csv(output_dir / "rejected_quality.csv", index=False)

    train_ds = AptosDataset(
        train_frame,
        cfg.data.image_dir,
        cfg.data.image_col,
        cfg.data.label_col,
        cfg.data.image_ext,
        transform=build_transforms(cfg.train.image_size, train=True),
        quality_filter=False,
        quality_thresholds=thresholds,
    )
    val_ds = AptosDataset(
        val_frame,
        cfg.data.image_dir,
        cfg.data.image_col,
        cfg.data.label_col,
        cfg.data.image_ext,
        transform=build_transforms(cfg.train.image_size, train=False),
        quality_filter=False,
        quality_thresholds=thresholds,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attach_num_classes(
        create_model(cfg.model.name, cfg.data.num_classes, cfg.model.pretrained, cfg.model.drop_rate),
        cfg.data.num_classes,
    ).to(device)

    weights = None
    if cfg.train.class_weighting:
        weights = class_weights_from_labels(train_frame[cfg.data.label_col], cfg.data.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.mixed_precision and device.type == "cuda")
    scaler = scaler if scaler.is_enabled() else None

    wandb = None
    if cfg.logging.use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(project=cfg.logging.wandb_project, config=json.loads(Path(args.config).read_text()))

    best_kappa = -1.0
    patience = 0
    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["quadratic_weighted_kappa"],
            "learning_rate": current_lr,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if wandb:
            wandb.log(row)

        score = val_metrics["quadratic_weighted_kappa"]
        if score > best_kappa:
            best_kappa = score
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": json.loads(Path(args.config).read_text()),
                    "metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )
            (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
