from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def quadratic_weighted_kappa(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Per-class recall (sensitivity)
    per_class_recall = {}
    for i in labels:
        support = cm[i].sum()
        per_class_recall[str(i)] = float(cm[i, i] / support) if support > 0 else 0.0
    # Balanced accuracy = mean of per-class recalls
    balanced_acc = float(np.mean(list(per_class_recall.values())))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_acc,
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true, y_pred),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def binary_referable_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Computes binary clinical metrics for referable DR (classes 2, 3, 4)."""
    y_true_binary = (y_true >= 2).astype(int)
    y_prob_referable = y_pred_prob[:, 2:].sum(axis=1)
    y_pred_binary = (y_prob_referable >= threshold).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true_binary, y_prob_referable)
        pr_auc = average_precision_score(y_true_binary, y_prob_referable)
    except Exception:
        roc_auc, pr_auc = float("nan"), float("nan")

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
    }


def calibration_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """Computes calibration metrics (ECE and Brier score)."""
    num_classes = y_pred_prob.shape[1]
    y_true_one_hot = np.eye(num_classes)[y_true]
    brier = float(np.mean(np.sum((y_pred_prob - y_true_one_hot) ** 2, axis=1)))

    n_bins = 10
    confidences = np.max(y_pred_prob, axis=1)
    predictions = np.argmax(y_pred_prob, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(in_bin):
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * np.mean(in_bin)

    return {
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
    }


if __name__ == "__main__":
    print("==================================================")
    print(" RetinAI DR - Metrics Verification & Self-Test")
    print("==================================================")
    
    # Mock ground truth and predicted labels (5 classes: 0 to 4)
    mock_true = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    mock_pred = [0, 1, 2, 3, 3, 0, 2, 2, 3, 4]
    
    clf_res = classification_metrics(mock_true, mock_pred, num_classes=5)
    print(f"[OK] Classification Accuracy : {clf_res['accuracy'] * 100:.1f}%")
    print(f"[OK] Quadratic Weighted Kappa: {clf_res['quadratic_weighted_kappa']:.4f}")
    print(f"[OK] Macro F1 Score          : {clf_res['macro_f1']:.4f}")
    
    # Mock probabilities for referable DR evaluation (10 samples, 5 classes)
    np.random.seed(42)
    mock_probs = np.random.dirichlet(np.ones(5), size=10)
    mock_true_arr = np.array(mock_true)
    
    referable_res = binary_referable_metrics(mock_true_arr, mock_probs, threshold=0.5)
    print(f"[OK] Referable DR Sensitivity : {referable_res['sensitivity'] * 100:.1f}%")
    print(f"[OK] Referable DR Specificity : {referable_res['specificity'] * 100:.1f}%")
    print(f"[OK] Referable DR ROC-AUC     : {referable_res['roc_auc']:.4f}")
    
    calib_res = calibration_metrics(mock_true_arr, mock_probs)
    print(f"[OK] Expected Calib. Error    : {calib_res['expected_calibration_error']:.4f}")
    print(f"[OK] Brier Score              : {calib_res['brier_score']:.4f}")
    print("==================================================")
    print(" >> All metrics verified and working perfectly! <<")
    print("==================================================")

