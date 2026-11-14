"""
VoxGuard - Evaluation Metrics Engine
====================================
Fulfills TASK 2 Section 6: Model Evaluation Metrics.
Calculates Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Equal Error Rate (EER).
Consistently treats Class 1 (Fake/Spoof) as the positive class.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def calculate_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    positive_label: int = 1,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate Equal Error Rate (EER) where False Acceptance Rate (FAR) == False Rejection Rate (FRR).

    In deepfake speech forensics:
    - Positive class (1) = Fake/Spoof
    - Negative class (0) = Real/Bonafide
    - False Acceptance: Negative (Real) misclassified as Positive (Fake) -> FAR = FP / N_real
    - False Rejection: Positive (Fake) misclassified as Negative (Real) -> FRR = FN / N_fake

    Returns:
        eer: Equal Error Rate expressed as a float in [0.0, 1.0]
        eer_threshold: Operating decision threshold where |FAR - FRR| is minimized
        fpr: Array of False Acceptance Rates
        fnr: Array of False Rejection Rates
    """
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5, np.array([]), np.array([])

    # Compute ROC curve using sklearn
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=positive_label)
    fnr = 1.0 - tpr

    # Point where FAR and FRR intersect
    diff = np.abs(fpr - fnr)
    min_idx = np.argmin(diff)

    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    eer_threshold = float(thresholds[min_idx]) if min_idx < len(thresholds) else 0.5

    return eer, eer_threshold, fpr, fnr


def compute_forensic_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full suite of binary classification metrics for forensic evaluation.

    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake)
        y_scores: Continuous model output probabilities P(fake) in [0.0, 1.0]
        threshold: Decision threshold for discrete classification

    Returns:
        Dictionary of formatted metrics.
    """
    total_samples = len(y_true)
    if total_samples == 0:
        return {"error": "Empty dataset"}

    unique_classes = np.unique(y_true)
    num_real = int(np.sum(y_true == 0))
    num_fake = int(np.sum(y_true == 1))

    y_pred = (y_scores >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))

    # Guard against single-class evaluation splits
    if len(unique_classes) < 2:
        return {
            "total_samples": total_samples,
            "real_samples": num_real,
            "fake_samples": num_fake,
            "accuracy": acc,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "auc_roc": None,
            "eer_percent": None,
            "eer_threshold": None,
            "note": "Insufficient class variety for Precision/Recall/EER (single class present)",
        }

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    eer_val, eer_thresh, _, _ = calculate_eer(y_true, y_scores)

    return {
        "total_samples": total_samples,
        "real_samples": num_real,
        "fake_samples": num_fake,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "eer_percent": round(eer_val * 100.0, 2) if not np.isnan(eer_val) else None,
        "eer_threshold": round(eer_thresh, 4),
    }

