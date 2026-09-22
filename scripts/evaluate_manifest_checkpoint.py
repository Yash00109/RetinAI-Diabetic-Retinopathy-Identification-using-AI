from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dr_detection.dataset import ManifestImageDataset
from dr_detection.metrics import classification_metrics, quadratic_weighted_kappa
from dr_detection.models import create_model
from dr_detection.transforms import build_transforms


def predict_logits(model, loader, device, tta: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    logits_out: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if tta == "hflip":
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


def threshold_predict(scores: np.ndarray, thresholds: tuple[float, float, float, float]) -> np.ndarray:
    return np.digitize(scores, np.asarray(thresholds, dtype=np.float32)).astype(int)


def threshold_score(y_true: np.ndarray, preds: np.ndarray, metric: str) -> float:
    if metric == "qwk":
        return quadratic_weighted_kappa(y_true, preds)
    if metric == "macro_f1":
        return float(f1_score(y_true, preds, labels=list(range(5)), average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, preds))
    if metric == "balanced":
        qwk = quadratic_weighted_kappa(y_true, preds)
        macro_f1 = float(f1_score(y_true, preds, labels=list(range(5)), average="macro", zero_division=0))
        return 0.5 * qwk + 0.5 * macro_f1
    raise ValueError(f"Unsupported threshold metric: {metric}")


def optimize_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    step: float,
    metric: str = "qwk",
) -> tuple[tuple[float, float, float, float], float]:
    values = np.arange(0.35, 3.66, step)
    best_thresholds = (0.5, 1.5, 2.5, 3.5)
    best_score = -1.0
    for thresholds in itertools.combinations(values, 4):
        preds = threshold_predict(scores, thresholds)
        score = threshold_score(y_true, preds, metric)
        if score > best_score:
            best_score = score
            best_thresholds = tuple(float(round(x, 4)) for x in thresholds)
    return best_thresholds, float(best_score)


def split_loader(manifest: pd.DataFrame, split: str, image_size: int, batch_size: int, workers: int, preprocess: bool):
    frame = manifest[manifest["split"] == split].reset_index(drop=True)
    dataset = ManifestImageDataset(
        frame,
        transform=build_transforms(image_size, train=False, preprocess=preprocess),
    )
    return frame, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a manifest checkpoint with ordinal threshold optimization.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--drop-rate", type=float, default=0.25)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--threshold-metric", choices=["qwk", "macro_f1", "accuracy", "balanced"], default="qwk")
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"].astype(bool)].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model, 5, pretrained=False, drop_rate=args.drop_rate).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, val_loader = split_loader(
        manifest,
        "val",
        args.image_size,
        args.batch_size,
        args.workers,
        preprocess=not args.no_preprocess,
    )
    test_frame, test_loader = split_loader(
        manifest,
        "test",
        args.image_size,
        args.batch_size,
        args.workers,
        preprocess=not args.no_preprocess,
    )

    val_y, val_logits = predict_logits(model, val_loader, device, args.tta)
    test_y, test_logits = predict_logits(model, test_loader, device, args.tta)

    val_scores = expected_scores(val_logits)
    test_scores = expected_scores(test_logits)
    thresholds, val_threshold_score = optimize_thresholds(
        val_y,
        val_scores,
        args.threshold_step,
        args.threshold_metric,
    )

    evaluations = {
        "config": vars(args),
        "argmax": {
            "val": classification_metrics(val_y.tolist(), val_logits.argmax(axis=1).tolist(), 5),
            "test": classification_metrics(test_y.tolist(), test_logits.argmax(axis=1).tolist(), 5),
        },
        "expected_round": {
            "val": classification_metrics(val_y.tolist(), np.rint(val_scores).clip(0, 4).astype(int).tolist(), 5),
            "test": classification_metrics(test_y.tolist(), np.rint(test_scores).clip(0, 4).astype(int).tolist(), 5),
        },
        "optimized_thresholds": {
            "thresholds": thresholds,
            "threshold_metric": args.threshold_metric,
            "val_optimized_score": val_threshold_score,
            "val": classification_metrics(val_y.tolist(), threshold_predict(val_scores, thresholds).tolist(), 5),
            "test": classification_metrics(test_y.tolist(), threshold_predict(test_scores, thresholds).tolist(), 5),
        },
    }

    (output_dir / "posthoc_metrics.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    predictions = test_frame[["source", "image_path", "label"]].copy()
    predictions["argmax_pred"] = test_logits.argmax(axis=1)
    predictions["expected_score"] = test_scores
    predictions["threshold_pred"] = threshold_predict(test_scores, thresholds)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print(json.dumps(evaluations["optimized_thresholds"], indent=2))


if __name__ == "__main__":
    main()
