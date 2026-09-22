"""Quick diagnostic script for model debugging.

Loads a checkpoint, runs a forward pass on sample images, and prints
per-class confidence distributions and the most-confused class pairs.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dr_detection.dataset import ManifestImageDataset
from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model
from dr_detection.transforms import build_transforms

CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick model diagnostic tool.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--manifest", default="data/processed/manifest_quality_accepted.csv")
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=100, help="Max test samples for quick check")
    args = parser.parse_args()

    print("=" * 60)
    print(" RetinAI DR - Model Debug & Diagnostic Report")
    print("=" * 60)

    # 1. Load and inspect checkpoint
    print("\n[1/5] Loading checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    if "args" in checkpoint:
        train_args = checkpoint["args"]
        print("  Training config:")
        for k in ["model", "epochs", "batch_size", "image_size", "learning_rate", "label_smoothing"]:
            if k in train_args:
                print(f"    {k}: {train_args[k]}")
    elif "config" in checkpoint:
        print("  Config present (legacy format)")

    if "val_metrics" in checkpoint:
        vm = checkpoint["val_metrics"]
        print(f"  Best val accuracy: {vm.get('accuracy', 'N/A')}")
        print(f"  Best val QWK: {vm.get('quadratic_weighted_kappa', 'N/A')}")
    if "metrics" in checkpoint:
        vm = checkpoint["metrics"]
        print(f"  Best val accuracy: {vm.get('accuracy', 'N/A')}")
        print(f"  Best val QWK: {vm.get('quadratic_weighted_kappa', 'N/A')}")

    # 2. Load model
    print("\n[2/5] Loading model...")
    model = create_model(args.model, 5, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model} ({param_count:,} parameters)")

    # 3. Load data
    print("\n[3/5] Loading test data...")
    manifest = pd.read_csv(args.manifest)
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"].astype(bool)]
    test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if len(test_frame) > args.max_samples:
        test_frame = test_frame.sample(args.max_samples, random_state=42).reset_index(drop=True)
    print(f"  Test samples: {len(test_frame)}")
    print(f"  Label distribution: {dict(test_frame['label'].value_counts().sort_index())}")

    dataset = ManifestImageDataset(test_frame, transform=build_transforms(args.image_size, train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 4. Run inference
    print("\n[4/5] Running inference...")
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.tolist())
            all_probs.append(probs)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = all_probs.argmax(axis=1)

    # 5. Print diagnostics
    print("\n[5/5] Diagnostic Results")
    print("-" * 60)

    metrics = classification_metrics(all_labels.tolist(), all_preds.tolist(), num_classes=5)
    print(f"\n  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  QWK:               {metrics['quadratic_weighted_kappa']:.4f}")

    print("\n  Per-Class Recall:")
    for i in range(5):
        recall = metrics["per_class_recall"][str(i)]
        bar = "#" * int(recall * 30)
        print(f"    Class {i} ({CLASS_NAMES[i]:>17s}): {recall:.3f} |{bar}")

    # Confidence analysis
    print("\n  Per-Class Mean Confidence (on correct predictions):")
    for i in range(5):
        mask_correct = (all_labels == i) & (all_preds == i)
        if mask_correct.sum() > 0:
            mean_conf = all_probs[mask_correct, i].mean()
            print(f"    Class {i} ({CLASS_NAMES[i]:>17s}): {mean_conf:.3f}")
        else:
            print(f"    Class {i} ({CLASS_NAMES[i]:>17s}): No correct predictions!")

    # Most confused pairs
    cm = np.array(metrics["confusion_matrix"])
    np.fill_diagonal(cm, 0)
    print("\n  Top 5 Most Confused Pairs (true -> predicted):")
    flat_indices = np.argsort(cm.ravel())[::-1][:5]
    for idx in flat_indices:
        true_cls = idx // 5
        pred_cls = idx % 5
        count = cm[true_cls, pred_cls]
        if count > 0:
            print(f"    {CLASS_NAMES[true_cls]} -> {CLASS_NAMES[pred_cls]}: {count} samples")

    # Sample predictions
    print("\n  Sample Predictions (first 5):")
    for i in range(min(5, len(all_labels))):
        true_lbl = int(all_labels[i])
        pred_lbl = int(all_preds[i])
        conf = all_probs[i, pred_lbl]
        status = "OK" if true_lbl == pred_lbl else "WRONG"
        print(f"    [{status:>5s}] True: {CLASS_NAMES[true_lbl]:>17s} | "
              f"Pred: {CLASS_NAMES[pred_lbl]:>17s} (conf={conf:.3f})")

    print("\n" + "=" * 60)
    print(" Diagnostic complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
