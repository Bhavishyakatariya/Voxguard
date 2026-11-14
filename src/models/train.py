"""
VoxGuard - Neural Classifier Training Engine
============================================
Trains deepfake voice detection backbones (VoxGuardCNN / VoxGuardTransformer)
against preprocessed Mel-spectrogram features.

Features:
- Binary Cross-Entropy with Logits loss (BCEWithLogitsLoss)
- Dynamic class weighting to compensate for class imbalance (~18% real, ~82% fake)
- Real-time tracking of Loss, Accuracy, F1-Score, and AUC-ROC
- ReduceLROnPlateau learning rate adaptation
- Best checkpoint serialization to models/voxguard_model.pt (canonical) and models/{arch}_best.pt
- Fully configurable CLI (batch size, epochs, arch, learning rate, sample cap)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Enable project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset import VoxGuardDataset
from src.models.voxguard_cnn import VoxGuardCNN
from src.models.voxguard_transformer import VoxGuardTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VoxGuard.Train")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics including Accuracy, F1, and AUC-ROC."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for batch_idx, (x, y, _, _) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # Compute raw logits for numerically stable BCEWithLogitsLoss
        logits = model(x, return_logits=True)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()

        all_probs.extend(probs.flatten())
        all_targets.extend(targets.flatten())

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            avg_loss = total_loss / ((batch_idx + 1) * x.size(0))
            pct = ((batch_idx + 1) / len(loader)) * 100.0
            logger.info(f"  Step [{batch_idx+1:04d}/{len(loader):04d}] ({pct:5.1f}%) - Batch Loss: {loss.item():.4f} (Avg: {avg_loss:.4f})")
            sys.stdout.flush()

        if (batch_idx + 1) % 250 == 0:
            ckpt_save_path = PROJECT_ROOT / "models" / "voxguard_model.pt"
            torch.save(model.state_dict(), ckpt_save_path)
            logger.info(f"  [Checkpoint] Periodic snapshot saved to {ckpt_save_path.name} at step {batch_idx+1}")
            sys.stdout.flush()

    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    return epoch_loss, metrics["accuracy"], metrics["auc"]


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluate model on validation partition."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, return_logits=True)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = y.cpu().numpy()

        all_probs.extend(probs.flatten())
        all_targets.extend(targets.flatten())

    val_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    return val_loss, metrics["accuracy"], metrics["f1"], metrics["auc"]


def train_model(
    arch: str = "cnn",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_train_samples: Optional[int] = 5000,
    max_val_samples: Optional[int] = 1000,
    random_seed: int = 42,
    device_str: Optional[str] = None,
    checkpoint_dir: Union[str, Path] = "models",
    reports_dir: Union[str, Path] = "reports",
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> Dict[str, any]:
    """
    Main training routine for VoxGuard classifiers.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 1. Device configuration
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 4))

    ckpt_path = PROJECT_ROOT / checkpoint_dir
    rpt_path = PROJECT_ROOT / reports_dir
    ckpt_path.mkdir(parents=True, exist_ok=True)
    rpt_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"VOXGUARD NEURAL CLASSIFIER TRAINING INITIATED [{arch.upper()}]")
    logger.info(f"Target Device: {device} | Max Epochs: {epochs} | Batch Size: {batch_size}")
    logger.info("=" * 70)

    # 2. Datasets & Loaders
    train_ds = VoxGuardDataset(
        split="train",
        max_samples=max_train_samples,
        random_seed=random_seed,
    )
    val_ds = VoxGuardDataset(
        split="val",
        max_samples=max_val_samples,
        random_seed=random_seed,
    )

    train_counts = train_ds.get_class_counts()
    val_counts = val_ds.get_class_counts()

    logger.info(f"Training Dataset:   {len(train_ds):>6} samples (Real: {train_counts[0]}, Fake: {train_counts[1]})")
    logger.info(f"Validation Dataset: {len(val_ds):>6} samples (Real: {val_counts[0]}, Fake: {val_counts[1]})")

    num_workers = 0  # 0 is safest across Windows multiprocessing
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model Architecture Instantiation
    if arch.lower() == "transformer":
        model = VoxGuardTransformer().to(device)
    else:
        model = VoxGuardCNN().to(device)

    if resume_checkpoint is not None:
        p_res = Path(resume_checkpoint)
        if not p_res.is_absolute():
            p_res = PROJECT_ROOT / p_res
        if p_res.is_file():
            logger.info(f"Resuming model weights from existing checkpoint: {p_res}")
            state_dict = torch.load(p_res, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

    # 4. Loss Function with Class Weighting
    pos_weight = train_ds.compute_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Optimizer & LR Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 6. Training Loop
    best_val_loss = float("inf")
    best_val_auc = 0.0
    history = []

    canonical_checkpoint = ckpt_path / "voxguard_model.pt"
    arch_checkpoint = ckpt_path / f"voxguard_{arch.lower()}_best.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc, train_auc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc, val_f1, val_auc = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        ep_duration = time.time() - ep_start

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.1f}s) - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train AUC: {train_auc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_auc": round(train_auc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_auc": round(val_auc, 4),
        }
        history.append(epoch_record)

        # Checkpoint Saving: Save when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            logger.info(f"  --> New best model checkpoint saved (Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f})")
            torch.save(model.state_dict(), arch_checkpoint)
            # If CNN, also save to canonical voxguard_model.pt for dashboard / XAI compatibility
            if arch.lower() == "cnn":
                torch.save(model.state_dict(), canonical_checkpoint)

    total_training_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETE IN {total_training_time/60:.2f} MINUTES")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f} | Best AUC: {best_val_auc:.4f}")
    logger.info(f"Primary Checkpoint Saved: {canonical_checkpoint}")
    logger.info(f"Arch Checkpoint Saved   : {arch_checkpoint}")
    logger.info("=" * 70)

    # Save training history JSON
    history_file = rpt_path / f"training_history_{arch.lower()}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": arch,
                "epochs": epochs,
                "batch_size": batch_size,
                "total_train_samples": len(train_ds),
                "total_val_samples": len(val_ds),
                "best_val_loss": round(best_val_loss, 4),
                "best_val_auc": round(best_val_auc, 4),
                "history": history,
            },
            f,
            indent=2,
        )
    logger.info(f"Training History Saved: {history_file}")

    return {
        "status": "Success",
        "architecture": arch,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "checkpoint_path": str(canonical_checkpoint if arch.lower() == "cnn" else arch_checkpoint),
        "history_path": str(history_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Neural Classifier Training Script")
    parser.add_argument("--arch", type=str, default="cnn", choices=["cnn", "transformer"], help="Model backbone")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-train-samples", type=int, default=4000, help="Cap training samples (balanced)")
    parser.add_argument("--max-val-samples", type=int, default=1000, help="Cap validation samples (balanced)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    max_train = args.max_train_samples if args.max_train_samples > 0 else None
    max_val = args.max_val_samples if args.max_val_samples > 0 else None

    train_model(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=max_train,
        max_val_samples=max_val,
        resume_checkpoint=args.resume,
    )
