"""Evaluate a trained experiment checkpoint on the test split.

Produces:
- metrics.json (comprehensive evaluation metrics)
- test_predictions.csv (per-sample predictions for analysis)
- confusion_matrix.png (visual confusion matrix)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from dr_detection.dataset import ManifestImageDataset
from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model
from dr_detection.transforms import build_transforms


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def evaluate(args):
    exp_dir = Path(args.experiment_dir)
    config_path = exp_dir / "config.json"
    ckpt_path = exp_dir / "best_model.pt"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest or config.get("manifest", "data/manifests/aptos_fixed_manifest.csv"))
    
    test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if len(test_frame) == 0:
        raise ValueError("No test samples found in manifest.")
    print(f"Test samples: {len(test_frame)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get("model", "efficientnet_b0")
    image_size = config.get("image_size", 384)
    drop_rate = config.get("drop_rate", 0.25)
    
    model = create_model(model_name, 5, pretrained=False, drop_rate=drop_rate)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    
    test_ds = ManifestImageDataset(
        test_frame,
        transform=build_transforms(image_size, train=False, preprocess=True),
        root_dir=getattr(args, "data_dir", None),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.get("batch_size", 32),
        shuffle=False, num_workers=min(config.get("workers", 4), 4), pin_memory=True,
    )
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_probs.extend(probs.cpu().tolist())
    
    metrics = classification_metrics(y_true, y_pred, num_classes=5)
    
    print(f"\n{'='*60}")
    print(f"Test Results for {config.get('experiment_id', 'unknown')}")
    print(f"{'='*60}")
    print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"QWK:               {metrics['quadratic_weighted_kappa']:.4f}")
    
    # Per-class results
    report = metrics.get("classification_report", {})
    print(f"\nPer-class results:")
    print(f"{'Class':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for i in range(5):
        cls = report.get(str(i), {})
        print(f"{i} - {CLASS_NAMES[i]:<14} {cls.get('precision', 0):>8.4f} {cls.get('recall', 0):>8.4f} "
              f"{cls.get('f1-score', 0):>8.4f} {int(cls.get('support', 0)):>8}")
    
    # Save metrics
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        "image_path": test_frame["image_path"].tolist(),
        "true_label": y_true,
        "pred_label": y_pred,
        "true_class": [CLASS_NAMES[y] for y in y_true],
        "pred_class": [CLASS_NAMES[y] for y in y_pred],
        "correct": [int(t == p) for t, p in zip(y_true, y_pred)],
    })
    for i in range(5):
        predictions_df[f"prob_{CLASS_NAMES[i]}"] = [p[i] for p in y_probs]
    predictions_df.to_csv(exp_dir / "test_predictions.csv", index=False)
    
    # Confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (Counts)\n{config.get('experiment_id', '')}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    # Normalized by row (recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Row-Normalized / Recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    fig.savefig(exp_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nArtifacts saved to: {exp_dir}")
    print(f"  - metrics.json")
    print(f"  - test_predictions.csv")
    print(f"  - confusion_matrix.png")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RetinAI experiment checkpoint.")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory")
    parser.add_argument("--manifest", default=None, help="Override manifest path")
    parser.add_argument("--data-dir", default=None, help="Root directory for dataset images (default: auto-detected)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
