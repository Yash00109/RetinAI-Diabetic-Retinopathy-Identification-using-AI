from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dr_detection.dataset import ManifestImageDataset
from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model
from dr_detection.train import class_weights_from_labels, run_epoch, set_seed
from dr_detection.transforms import build_transforms


def evaluate_predictions(model, loader, device, criterion):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, num_classes=5)
    return float(np.mean(losses)), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DR classifier from a unified image manifest.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/manifest_efficientnet_b0")
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-rate", type=float, default=0.25)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.get("exists", True).astype(bool)] if "exists" in manifest.columns else manifest
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("Manifest must contain non-empty train/val/test splits.")

    train_ds = ManifestImageDataset(
        train_frame,
        transform=build_transforms(args.image_size, train=True, preprocess=not args.no_preprocess),
    )
    val_ds = ManifestImageDataset(
        val_frame,
        transform=build_transforms(args.image_size, train=False, preprocess=not args.no_preprocess),
    )
    test_ds = ManifestImageDataset(
        test_frame,
        transform=build_transforms(args.image_size, train=False, preprocess=not args.no_preprocess),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, 5, pretrained=not args.no_pretrained, drop_rate=args.drop_rate)
    model.num_classes = 5
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.num_classes = 5
    model = model.to(device)

    weights = class_weights_from_labels(train_frame["label"], 5).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler = scaler if scaler.is_enabled() else None

    start_epoch = 1
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(resume_ckpt["model_state"])
        print(f"Resumed from checkpoint: {args.resume}")

    best_qwk = -1.0
    stale = 0
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = evaluate_predictions(model, val_loader, device, criterion)
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
        print(json.dumps(row))

        if row["val_qwk"] > best_qwk:
            best_qwk = row["val_qwk"]
            stale = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"model_state": state, "args": vars(args), "val_metrics": val_metrics}, output_dir / "best_model.pt")
            (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    state = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)["model_state"]
    eval_model = create_model(args.model, 5, pretrained=False, drop_rate=args.drop_rate).to(device)
    eval_model.load_state_dict(state)
    eval_model.eval()

    # Test with TTA (horizontal flip)
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = eval_model(images)
            logits_flip = eval_model(torch.flip(images, dims=[3]))
            avg_logits = (logits + logits_flip) / 2.0
            all_labels.extend(labels.tolist())
            all_logits.append(avg_logits.cpu())
    all_logits_cat = torch.cat(all_logits, dim=0)
    y_pred = all_logits_cat.argmax(dim=1).tolist()
    test_metrics = classification_metrics(all_labels, y_pred, num_classes=5)
    test_loss_val, _ = evaluate_predictions(eval_model, test_loader, device, criterion)
    test_metrics["test_loss"] = test_loss_val
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("TEST_METRICS", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
