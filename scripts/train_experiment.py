"""Reproducible experiment training script for RetinAI.

Phase 4: Fully configurable, reproducible training with experiment tracking.
Every experiment gets a unique ID (EXP-XXX) and saves all artifacts needed
for full reproducibility and auditing.

Usage:
    python scripts/train_experiment.py --config experiments/EXP-000/config.json
    
    OR with CLI overrides:
    
    python scripts/train_experiment.py \
        --manifest data/manifests/aptos_fixed_manifest.csv \
        --experiment-id EXP-000 \
        --model efficientnet_b0 \
        --image-size 384 \
        --epochs 25 \
        --batch-size 32 \
        --learning-rate 0.0002 \
        --weight-decay 0.0001 \
        --loss weighted_ce \
        --label-smoothing 0.05 \
        --sampler normal \
        --augmentation standard \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dr_detection.dataset import ManifestImageDataset
from dr_detection.loss import FocalLoss, OrdinalDistanceLoss
from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model
from dr_detection.sampler import BalancedBatchSampler, ClassAwareRandomSampler
from dr_detection.train import class_weights_from_labels, run_epoch, set_seed
from dr_detection.transforms import build_transforms


def build_loss(loss_type: str, weights: torch.Tensor, label_smoothing: float, device, **kwargs):
    """Build loss function from configuration."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=label_smoothing)
    elif loss_type == "focal":
        gamma = kwargs.get("focal_gamma", 2.0)
        return FocalLoss(weight=weights.to(device), gamma=gamma, label_smoothing=label_smoothing)
    elif loss_type == "ordinal":
        distance_weight = kwargs.get("ordinal_distance_weight", 0.35)
        return OrdinalDistanceLoss(
            num_classes=5, weight=weights.to(device),
            label_smoothing=label_smoothing, distance_weight=distance_weight,
        )
    elif loss_type == "hybrid":
        # Hybrid = alpha * CE + (1-alpha) * Ordinal, implemented as OrdinalDistanceLoss
        distance_weight = kwargs.get("ordinal_distance_weight", 0.35)
        return OrdinalDistanceLoss(
            num_classes=5, weight=weights.to(device),
            label_smoothing=label_smoothing, distance_weight=distance_weight,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_sampler(sampler_type: str, labels, batch_size: int):
    """Build sampler from configuration. Returns (sampler, use_shuffle, batch_sampler)."""
    if sampler_type == "normal":
        return None, True, None
    elif sampler_type == "class_aware":
        sampler = ClassAwareRandomSampler(labels)
        return sampler, False, None
    elif sampler_type == "balanced_batch":
        samples_per_class = max(1, batch_size // 5)
        batch_sampler = BalancedBatchSampler(labels, samples_per_class=samples_per_class)
        return None, False, batch_sampler
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def evaluate_split(model, loader, device, criterion):
    """Evaluate model on a data split."""
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    y_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
            y_probs.extend(probs.detach().cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, num_classes=5)
    metrics["loss"] = float(np.mean(losses))
    return metrics, y_true, y_pred, y_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinAI Experiment Training Script")
    
    # Experiment metadata
    parser.add_argument("--experiment-id", default="EXP-000", help="Unique experiment identifier")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: experiments/{experiment-id})")
    
    # Data
    parser.add_argument("--manifest", default="data/manifests/aptos_fixed_manifest.csv")
    
    # Model
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--drop-rate", type=float, default=0.25)
    parser.add_argument("--no-pretrained", action="store_true")
    
    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    
    # Loss
    parser.add_argument("--loss", default="weighted_ce",
                       choices=["ce", "weighted_ce", "focal", "ordinal", "hybrid"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ordinal-distance-weight", type=float, default=0.35)
    
    # Sampler
    parser.add_argument("--sampler", default="normal",
                       choices=["normal", "class_aware", "balanced_batch"])
    
    # Augmentation
    parser.add_argument("--augmentation", default="standard",
                       choices=["standard", "targeted", "strong"])
    parser.add_argument("--no-preprocess", action="store_true")
    
    # Resume
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = vars(args)
    config["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["torch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config["cuda_device"] = torch.cuda.get_device_name(0)
    
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Experiment: {args.experiment_id}")
    print(f"Output: {output_dir}")
    print(json.dumps(config, indent=2))
    
    # Load manifest
    manifest = pd.read_csv(args.manifest)
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"].astype(bool)]
    
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    
    if len(train_frame) == 0 or len(val_frame) == 0:
        raise ValueError("Manifest must contain non-empty train and val splits.")
    
    print(f"\nTrain: {len(train_frame)} | Val: {len(val_frame)}")
    print(f"Train class distribution: {dict(train_frame['label'].value_counts().sort_index())}")
    
    # Datasets
    train_ds = ManifestImageDataset(
        train_frame,
        transform=build_transforms(args.image_size, train=True, preprocess=not args.no_preprocess, augmentation=args.augmentation),
    )
    val_ds = ManifestImageDataset(
        val_frame,
        transform=build_transforms(args.image_size, train=False, preprocess=not args.no_preprocess),
    )
    
    # Sampler
    sampler, use_shuffle, batch_sampler = build_sampler(
        args.sampler, train_frame["label"].values, args.batch_size
    )
    
    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=use_shuffle,
            sampler=sampler, num_workers=args.workers, pin_memory=True,
        )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, 5, pretrained=not args.no_pretrained, drop_rate=args.drop_rate)
    model.num_classes = 5
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.num_classes = 5
    model = model.to(device)
    
    # Loss
    weights = class_weights_from_labels(train_frame["label"], 5)
    criterion = build_loss(
        args.loss, weights, args.label_smoothing, device,
        focal_gamma=args.focal_gamma,
        ordinal_distance_weight=args.ordinal_distance_weight,
    )
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler = scaler if scaler.is_enabled() else None
    
    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(ckpt["model_state"])
        print(f"Resumed from: {args.resume}")
    
    # Training loop
    best_qwk = -1.0
    stale = 0
    history = []
    
    print(f"\n{'='*80}")
    print(f"Training {args.model} | {args.image_size}px | {args.loss} | {args.sampler} | {args.augmentation}")
    print(f"{'='*80}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics, _, _, _ = evaluate_split(model, val_loader, device, criterion)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["quadratic_weighted_kappa"],
            "learning_rate": current_lr,
        }
        history.append(row)
        
        improved = " *BEST*" if row["val_qwk"] > best_qwk else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val QWK: {val_metrics['quadratic_weighted_kappa']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | "
              f"LR: {current_lr:.6f}{improved}")
        
        if row["val_qwk"] > best_qwk:
            best_qwk = row["val_qwk"]
            stale = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {"model_state": state, "config": config, "val_metrics": val_metrics, "epoch": epoch},
                output_dir / "best_model.pt",
            )
            (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs (patience={args.patience}).")
                break
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "train_history.csv", index=False)
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["best_val_qwk"] = best_qwk
    config["total_epochs_run"] = len(history)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    print(f"\nTraining complete. Best val QWK: {best_qwk:.4f}")
    print(f"Artifacts saved to: {output_dir}")
    print(f"Run evaluation with:")
    print(f"  python scripts/evaluate_experiment.py --experiment-dir {output_dir} --manifest {args.manifest}")


if __name__ == "__main__":
    main()
