"""
VoxGuard - Evaluation & Cross-Dataset Benchmarking Suite
========================================================
Comprehensive evaluation of trained VoxGuard models against held-out test data.

Calculates:
1. Overall Classification Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
2. Biometric Forensics Standards: Equal Error Rate (EER) per SRS Equation (15)
3. Confusion Matrix: TP, FP, TN, FN
4. Cross-Dataset Generalization Analysis:
   - ASVspoof 2021 LA Performance
   - In-The-Wild Performance
   - Generalization Gap (Delta EER = EER_inthewild - EER_asvspoof)
5. Generates reports/evaluation_report.json and ROC curve plots.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

# Enable project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset import VoxGuardDataset
from src.models.voxguard_cnn import VoxGuardCNN, load_cnn_model
from src.models.voxguard_transformer import VoxGuardTransformer, load_transformer_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VoxGuard.Evaluate")


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute Equal Error Rate (EER) and operating threshold theta* per SRS Eq (15):
    theta* = arg min_theta |P_fa(theta) - P_miss(theta)|,
    EER = (P_fa(theta*) + P_miss(theta*)) / 2.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # Find intersection point where FPR ~= FNR
    abs_diffs = np.abs(fpr - fnr)
    idx = int(np.argmin(abs_diffs))

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    best_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, best_threshold, fpr, tpr


@torch.no_grad()
def run_evaluation(
    checkpoint_path: Union[str, Path] = "models/voxguard_model.pt",
    arch: str = "cnn",
    max_test_samples: int = 2000,
    batch_size: int = 32,
    device_str: str = "cpu",
    output_dir: Union[str, Path] = "reports",
) -> Dict[str, any]:
    """
    Execute full empirical evaluation and produce audit artifacts.
    """
    ckpt_file = PROJECT_ROOT / checkpoint_path
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_file.resolve()}")

    device = torch.device(device_str)

    logger.info("=" * 70)
    logger.info("VOXGUARD MODEL EVALUATION & CROSS-DATASET AUDIT")
    logger.info(f"Model Checkpoint: {ckpt_file.name} | Architecture: {arch.upper()}")
    logger.info("=" * 70)

    # 1. Load Model
    if arch.lower() == "transformer":
        model = load_transformer_model(ckpt_file, device=device)
    else:
        model = load_cnn_model(ckpt_file, device=device)

    # 2. Load Held-Out Test Set
    test_ds = VoxGuardDataset(
        split="test",
        max_samples=max_test_samples,
        random_seed=42,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logger.info(f"Loaded {len(test_ds)} evaluation samples from held-out test split.")

    # 3. Model Inference Loop
    all_targets = []
    all_probs = []
    all_datasets = []

    for x, y, _, ds_name in test_loader:
        x = x.to(device)
        probs = model(x).cpu().numpy().flatten()
        targets = y.numpy().flatten()

        all_probs.extend(probs)
        all_targets.extend(targets)
        all_datasets.extend(ds_name)

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    datasets = np.array(all_datasets)

    # 4. Global Metrics Computation
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    eer, eer_threshold, fpr, tpr = compute_eer(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 5. Dataset-Specific Breakdown (Cross-Dataset Generalization)
    dataset_metrics = {}
    for ds in ["asvspoof", "inthewild"]:
        mask = (datasets == ds)
        if np.sum(mask) > 10:
            ds_true = y_true[mask]
            ds_prob = y_prob[mask]
            ds_pred = (ds_prob >= 0.5).astype(int)

            ds_acc = float(accuracy_score(ds_true, ds_pred))
            try:
                ds_auc = float(roc_auc_score(ds_true, ds_prob))
            except Exception:
                ds_auc = 0.5
            ds_eer, _, _, _ = compute_eer(ds_true, ds_prob)

            dataset_metrics[ds] = {
                "samples": int(np.sum(mask)),
                "accuracy": round(ds_acc, 4),
                "auc": round(ds_auc, 4),
                "eer": round(ds_eer, 4),
            }

    # Cross-dataset generalization gap
    if "asvspoof" in dataset_metrics and "inthewild" in dataset_metrics:
        delta_eer = dataset_metrics["inthewild"]["eer"] - dataset_metrics["asvspoof"]["eer"]
    else:
        delta_eer = 0.0

    # 6. Print Report Summary
    logger.info("-" * 70)
    logger.info("OVERALL TEST METRICS:")
    logger.info(f"  * Accuracy         : {acc*100:.2f}%")
    logger.info(f"  * Precision        : {prec:.4f}")
    logger.info(f"  * Recall           : {rec:.4f}")
    logger.info(f"  * F1-Score         : {f1:.4f}")
    logger.info(f"  * AUC-ROC          : {auc:.4f}")
    logger.info(f"  * Equal Error Rate : {eer*100:.2f}% (Threshold: {eer_threshold:.4f})")
    logger.info(f"  * Confusion Matrix : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logger.info("-" * 70)
    logger.info("CROSS-DATASET GENERALIZATION BREAKDOWN:")
    for ds, m in dataset_metrics.items():
        logger.info(f"  * [{ds.upper():<9}] Acc: {m['accuracy']*100:.2f}% | AUC: {m['auc']:.4f} | EER: {m['eer']*100:.2f}% ({m['samples']} samples)")
    logger.info(f"  * Generalization Gap (Delta EER): {delta_eer*100:.2f}%")
    logger.info("=" * 70)

    # 7. Generate ROC Curve Plot
    roc_plot_path = out_dir / f"roc_curve_{arch.lower()}.png"
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"VoxGuard {arch.upper()} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Chance Baseline")
    plt.plot([eer], [1 - eer], marker="o", markersize=7, color="#dc2626", label=f"EER = {eer*100:.2f}%")
    plt.title(f"Receiver Operating Characteristic (ROC) - {arch.upper()}", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    logger.info(f"ROC Curve Saved: {roc_plot_path}")

    # 8. Save structured JSON report
    report_file = out_dir / "evaluation_report.json"
    report_dict = {
        "model_architecture": arch,
        "checkpoint": str(ckpt_file.name),
        "total_test_samples": len(test_ds),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "equal_error_rate": round(eer, 4),
            "optimal_threshold": round(eer_threshold, 4),
        },
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "cross_dataset_metrics": dataset_metrics,
        "generalization_gap_delta_eer": round(delta_eer, 4),
        "roc_curve_plot": str(roc_plot_path.name),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Evaluation Report JSON Saved: {report_file}")

    return report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Model Evaluation Script")
    parser.add_argument("--checkpoint", type=str, default="models/voxguard_model.pt", help="Path to checkpoint")
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "transformer"], help="Model backbone")
    parser.add_argument("--max-test-samples", type=int, default=2000, help="Cap test samples for evaluation")
    args = parser.parse_args()

    max_test = args.max_test_samples if args.max_test_samples > 0 else None

    run_evaluation(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        max_test_samples=max_test,
    )

