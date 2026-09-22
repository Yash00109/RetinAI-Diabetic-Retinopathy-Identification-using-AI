from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_manifest_checkpoint import (
    expected_scores,
    optimize_thresholds,
    predict_logits,
    softmax,
    split_loader,
    threshold_predict,
)

from dr_detection.metrics import classification_metrics
from dr_detection.models import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an averaged checkpoint ensemble on a manifest split.")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--drop-rate", type=float, default=0.25)
    parser.add_argument("--threshold-step", type=float, default=0.2)
    parser.add_argument("--threshold-metric", choices=["qwk", "macro_f1", "accuracy", "balanced"], default="qwk")
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"].astype(bool)].reset_index(drop=True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_probs_sum = None
    test_probs_sum = None
    val_y = None
    test_y = None

    for checkpoint_path in args.checkpoints:
        model = create_model(args.model, 5, pretrained=False, drop_rate=args.drop_rate).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        current_val_y, val_logits = predict_logits(model, val_loader, device, args.tta)
        current_test_y, test_logits = predict_logits(model, test_loader, device, args.tta)

        if val_y is None:
            val_y = current_val_y
            test_y = current_test_y
        elif not (np.array_equal(val_y, current_val_y) and np.array_equal(test_y, current_test_y)):
            raise ValueError("Checkpoint predictions were generated against mismatched labels.")

        val_probs = softmax(val_logits)
        test_probs = softmax(test_logits)
        val_probs_sum = val_probs if val_probs_sum is None else val_probs_sum + val_probs
        test_probs_sum = test_probs if test_probs_sum is None else test_probs_sum + test_probs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert val_y is not None and test_y is not None
    val_probs_avg = val_probs_sum / len(args.checkpoints)
    test_probs_avg = test_probs_sum / len(args.checkpoints)
    val_scores = expected_scores(np.log(np.clip(val_probs_avg, 1e-12, 1.0)))
    test_scores = expected_scores(np.log(np.clip(test_probs_avg, 1e-12, 1.0)))
    thresholds, val_threshold_score = optimize_thresholds(
        val_y,
        val_scores,
        args.threshold_step,
        args.threshold_metric,
    )

    metrics = {
        "config": vars(args),
        "argmax": {
            "val": classification_metrics(val_y.tolist(), val_probs_avg.argmax(axis=1).tolist(), 5),
            "test": classification_metrics(test_y.tolist(), test_probs_avg.argmax(axis=1).tolist(), 5),
        },
        "optimized_thresholds": {
            "thresholds": thresholds,
            "threshold_metric": args.threshold_metric,
            "val_optimized_score": val_threshold_score,
            "val": classification_metrics(val_y.tolist(), threshold_predict(val_scores, thresholds).tolist(), 5),
            "test": classification_metrics(test_y.tolist(), threshold_predict(test_scores, thresholds).tolist(), 5),
        },
    }

    (output_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = test_frame[["source", "image_path", "label"]].copy()
    predictions["ensemble_pred"] = test_probs_avg.argmax(axis=1)
    predictions["ensemble_expected_score"] = test_scores
    predictions["ensemble_threshold_pred"] = threshold_predict(test_scores, thresholds)
    predictions.to_csv(output_dir / "ensemble_test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
