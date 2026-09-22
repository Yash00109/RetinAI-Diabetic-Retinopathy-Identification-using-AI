import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dr_detection.dataset import ManifestImageDataset
from dr_detection.metrics import (
    binary_referable_metrics,
    calibration_metrics,
    classification_metrics,
)
from dr_detection.models import create_model
from dr_detection.transforms import build_robustness_transforms, build_transforms


# For pretty printing
def print_header(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_table(data_dict, title=None):
    if title:
        print(f"--- {title} ---")
    for k, v in data_dict.items():
        if isinstance(v, float):
            print(f"{k.ljust(30)}: {v:.4f}")
        elif not isinstance(v, (list, dict)):
            print(f"{k.ljust(30)}: {v}")
    print()

def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    logits_out = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            # Default tta = hflip
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
            labels.extend(batch_labels.detach().cpu().tolist())
            logits_out.append(logits.detach().cpu().numpy())
    return np.asarray(labels, dtype=int), np.concatenate(logits_out, axis=0)

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def expected_scores(logits: np.ndarray) -> np.ndarray:
    probs = softmax(logits)
    return probs @ np.arange(probs.shape[1], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--subset", type=int, default=None, help="Subset size for robustness tests")
    parser.add_argument("--output-dir", default=None, help="Directory to save evaluation results JSON")
    args = parser.parse_args()

    print_header("PHASE 1: DATA & SPLIT VALIDATION")
    
    manifest = pd.read_csv(args.manifest)
    print(f"Loaded manifest with {len(manifest)} rows.")
    
    # 1. Label-range validation
    invalid_labels = manifest[~manifest["label"].isin([0, 1, 2, 3, 4])]
    print(f"Label-range validation: {'PASS' if len(invalid_labels) == 0 else 'FAIL'} (Invalid: {len(invalid_labels)})")
    
    # 2. Missing-image test
    missing = []
    for p in manifest["image_path"]:
        if not os.path.exists(p):
            missing.append(p)
    print(f"Missing-image test: {'PASS' if len(missing) == 0 else 'FAIL'} (Missing: {len(missing)})")
    
    # Filter only existing
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"].astype(bool)].reset_index(drop=True)
    elif len(missing) > 0:
        manifest = manifest[~manifest["image_path"].isin(missing)].reset_index(drop=True)
        
    # 3. Duplicate-image test
    duplicates = manifest["image_path"].duplicated().sum()
    print(f"Duplicate-image test: {'PASS' if duplicates == 0 else 'FAIL'} (Duplicates: {duplicates})")

    # 4. Cross-split leakage test
    train_images = set(manifest[manifest["split"] == "train"]["image_path"])
    test_images = set(manifest[manifest["split"] == "test"]["image_path"])
    leakage = train_images.intersection(test_images)
    print(f"Cross-split leakage test: {'PASS' if len(leakage) == 0 else 'FAIL'} (Leaks: {len(leakage)})")
    
    # 5. Class-imbalance report
    print("\nClass Distribution (All splits):")
    print(manifest.groupby("split")["label"].value_counts().unstack().fillna(0).astype(int))
    
    print_header("PHASE 1: CORE 5-CLASS METRICS (Test Set)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model(args.model, 5, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model checkpoint: {args.checkpoint}")

    test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if args.subset and len(test_frame) > args.subset:
        test_frame = test_frame.sample(args.subset, random_state=42).reset_index(drop=True)
        print(f"Using subset of {args.subset} test images.")
        
    test_dataset = ManifestImageDataset(test_frame, transform=build_transforms(args.image_size, train=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    test_y, test_logits = predict_logits(model, test_loader, device)
    
    # Expected round predictions
    test_scores = expected_scores(test_logits)
    test_preds_expected = np.rint(test_scores).clip(0, 4).astype(int)
    
    metrics_5class = classification_metrics(test_y.tolist(), test_preds_expected.tolist(), 5)
    
    # Balanced Accuracy manually computed
    cm = np.array(metrics_5class["confusion_matrix"])
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    balanced_acc = np.mean(recalls)
    metrics_5class["balanced_accuracy"] = float(balanced_acc)
    
    print_table({
        "Accuracy": metrics_5class["accuracy"],
        "Balanced Accuracy": metrics_5class["balanced_accuracy"],
        "Macro F1": metrics_5class["macro_f1"],
        "Quadratic Weighted Kappa": metrics_5class["quadratic_weighted_kappa"],
    }, title="Primary Ordinal Metrics")
    
    print("Confusion Matrix:")
    print(np.array(metrics_5class["confusion_matrix"]))
    
    print("\nClassification Report:")
    report = metrics_5class["classification_report"]
    for i in range(5):
        s = str(i)
        if s in report:
            print(f"Class {i}: Precision: {report[s]['precision']:.4f} | Recall: {report[s]['recall']:.4f} | F1: {report[s]['f1-score']:.4f} | Support: {report[s]['support']}")
            
    print_header("PHASE 2: CLINICAL-SAFETY & CALIBRATION (Referable DR)")
    
    test_probs = softmax(test_logits)
    bin_metrics = binary_referable_metrics(test_y, test_probs, threshold=0.5)
    
    print_table(bin_metrics, title="Binary Referable-DR (Threshold=0.5)")
    
    cal_metrics = calibration_metrics(test_y, test_probs)
    print_table(cal_metrics, title="Probability Calibration")
    
    print_header("PHASE 2: ROBUSTNESS TESTS")
    
    perturbations = ["blur", "brightness", "noise"]
    robust_results = {}
    
    for pert in perturbations:
        print(f"Running robustness test: {pert}...")
        pert_dataset = ManifestImageDataset(test_frame, transform=build_robustness_transforms(args.image_size, pert))
        pert_loader = DataLoader(pert_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
        pert_y, pert_logits = predict_logits(model, pert_loader, device)
        pert_scores = expected_scores(pert_logits)
        pert_preds = np.rint(pert_scores).clip(0, 4).astype(int)
        
        m = classification_metrics(pert_y.tolist(), pert_preds.tolist(), 5)
        robust_results[pert] = {
            "QWK": m["quadratic_weighted_kappa"],
            "Accuracy": m["accuracy"],
            "Macro_F1": m["macro_f1"]
        }
        
    for pert, res in robust_results.items():
        print(f"--- Robustness: {pert.capitalize()} ---")
        print(f"QWK Drop: {metrics_5class['quadratic_weighted_kappa'] - res['QWK']:.4f} (Base: {metrics_5class['quadratic_weighted_kappa']:.4f} -> {res['QWK']:.4f})")
        print(f"Acc Drop: {metrics_5class['accuracy'] - res['Accuracy']:.4f}")
        print()

    print_header("EVALUATION COMPLETE")

    # Save all results to JSON
    all_results = {
        "core_metrics": {
            "accuracy": metrics_5class["accuracy"],
            "balanced_accuracy": metrics_5class.get("balanced_accuracy"),
            "macro_f1": metrics_5class["macro_f1"],
            "quadratic_weighted_kappa": metrics_5class["quadratic_weighted_kappa"],
            "confusion_matrix": metrics_5class["confusion_matrix"],
            "classification_report": metrics_5class["classification_report"],
        },
        "referable_dr": bin_metrics,
        "calibration": cal_metrics,
        "robustness": robust_results,
    }

    output_dir = args.output_dir or str(Path(args.checkpoint).parent)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "comprehensive_eval_results.json"
    results_file.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
