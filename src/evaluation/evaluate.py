"""
VoxGuard - Model Evaluation Engine
==================================
Fulfills TASK 2 Section 6: Model Evaluation.
Evaluates classifiers on held-out test corpora, computing overall and dataset-wise metrics
(ASVspoof vs In-The-Wild) to diagnose the cross-dataset generalization gap.
Saves structured JSON and CSV reports to results/metrics/.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset import VoxGuardDataset
from src.models.model_factory import get_model
from src.evaluation.metrics import compute_forensic_metrics
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("VoxGuard.Evaluate")


def run_evaluation(
    checkpoint_path: Union[str, Path] = "models/voxguard_model.pt",
    arch: str = "cnn",
    metadata_path: Union[str, Path] = "data/processed/metadata.csv",
    batch_size: int = 64,
    max_test_samples: Optional[int] = None,
    results_dir: Union[str, Path] = "results/metrics",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full forensic evaluation suite.
    """
    results_path = PROJECT_ROOT / results_dir
    results_path.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("=" * 70)
    logger.info(f"VOXGUARD FORENSIC EVALUATION [{arch.upper()}] ON {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Metadata:   {metadata_path}")
    logger.info("=" * 70)

    # 1. Load Model
    model = get_model(arch=arch, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # 2. Load Test Partition
    test_ds = VoxGuardDataset(
        metadata_path=metadata_path,
        split="test",
        max_samples=max_test_samples,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"Loaded {len(test_ds)} test samples across {len(test_loader)} batches")

    # 3. Model Inference
    all_targets = []
    all_probs = []
    all_datasets = []
    all_paths = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (x, y, paths, dsets) in enumerate(test_loader):
            x = x.to(device)
            probs = model(x).cpu().numpy().flatten()
            targets = y.numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets)
            all_datasets.extend(dsets)
            all_paths.extend(paths)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                pct = ((batch_idx + 1) / len(test_loader)) * 100.0
                logger.info(f"  Evaluated [{batch_idx+1:04d}/{len(test_loader):04d}] ({pct:5.1f}%)")

    inf_time = time.time() - t0
    logger.info(f"Inference completed in {inf_time:.2f}s ({len(test_ds) / max(0.001, inf_time):.1f} samples/s)")

    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)
    datasets = np.array(all_datasets)

    # 4. Overall Metrics
    overall_metrics = compute_forensic_metrics(y_true, y_scores)

    # 5. Dataset-Wise Metrics (ASVspoof vs In-The-Wild)
    dataset_metrics = {}
    for dname in np.unique(datasets):
        mask = (datasets == dname)
        d_true = y_true[mask]
        d_scores = y_scores[mask]
        dataset_metrics[dname] = compute_forensic_metrics(d_true, d_scores)

    # 6. Generalization Gap Diagnosis
    gap_info = {}
    if "asvspoof" in dataset_metrics and "inthewild" in dataset_metrics:
        asv_eer = dataset_metrics["asvspoof"].get("eer_percent")
        itw_eer = dataset_metrics["inthewild"].get("eer_percent")
        asv_auc = dataset_metrics["asvspoof"].get("auc_roc")
        itw_auc = dataset_metrics["inthewild"].get("auc_roc")

        if asv_eer is not None and itw_eer is not None:
            gap_info["eer_gap_percent"] = round(abs(itw_eer - asv_eer), 2)
        if asv_auc is not None and itw_auc is not None:
            gap_info["auc_gap"] = round(abs(itw_auc - asv_auc), 4)

    # 7. Compile Final Report
    report = {
        "architecture": arch,
        "checkpoint": str(checkpoint_path),
        "total_test_samples": len(y_true),
        "overall_metrics": overall_metrics,
        "dataset_wise_metrics": dataset_metrics,
        "generalization_gap": gap_info,
        "inference_duration_seconds": round(inf_time, 2),
    }

    # Save JSON Report
    json_path = results_path / f"evaluation_{arch}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation JSON report to: {json_path}")

    # Also save structured CSV summary
    summary_rows = []
    summary_rows.append({"partition": "OVERALL", **overall_metrics})
    for dname, dm in dataset_metrics.items():
        summary_rows.append({"partition": dname.upper(), **dm})
    summary_df = pd.DataFrame(summary_rows)
    csv_path = results_path / f"evaluation_{arch}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved evaluation summary CSV to: {csv_path}")

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"VOXGUARD EVALUATION SUMMARY [{arch.upper()}]")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    if gap_info:
        print("\n[Cross-Dataset Generalization Gap]")
        for k, v in gap_info.items():
            print(f"  {k}: {v}")
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    parser.add_argument("--checkpoint", type=str, default="models/voxguard_model.pt", help="Checkpoint path")
    parser.add_argument("--arch", type=str, default=None, choices=["cnn", "transformer"], help="Model architecture")
    parser.add_argument("--data-source", type=str, default=None, choices=["real", "dummy"], help="Data source toggle")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Cap test samples (0 = full)")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_source is not None:
        cfg["data_source"] = args.data_source
        if args.data_source == "dummy":
            cfg["metadata_path"] = "data/dummy/metadata.csv"
        else:
            cfg["metadata_path"] = "data/processed/metadata.csv"

    arch = args.arch or cfg.get("model", {}).get("arch", "cnn")
    max_test = args.max_test_samples if (args.max_test_samples is not None and args.max_test_samples > 0) else None

    run_evaluation(
        checkpoint_path=args.checkpoint,
        arch=arch,
        metadata_path=cfg["metadata_path"],
        batch_size=args.batch_size,
        max_test_samples=max_test,
    )

