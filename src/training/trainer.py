"""
VoxGuard - Neural Classifier Trainer Engine
===========================================
Fulfills TASK 2 Section 4: Training Pipeline & Section 5: Checkpointing.
Coordinating model training, dynamic class weighting, validation, early stopping,
and structured artifact generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.models.dataset import VoxGuardDataset
from src.models.model_factory import get_model
from src.training.early_stopping import EarlyStopping
from src.utils.logger import get_logger
from src.utils.checkpoint import save_checkpoint

logger = get_logger("VoxGuard.Trainer")


class SpecAugment:
    """
    SpecAugment: Frequency and Time masking data augmentation for Mel-spectrograms.
    Prevents deepfake classifiers from overfitting to narrow vocoder artifact bands.
    """
    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 16,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, 128, 128)
        augmented = x.clone()
        _, f, t = augmented.shape
        for _ in range(self.num_freq_masks):
            f_len = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, f - f_len))
            augmented[:, f0:f0+f_len, :] = 0.0
        for _ in range(self.num_time_masks):
            t_len = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, t - t_len))
            augmented[:, :, t0:t0+t_len] = 0.0
        return augmented


def compute_epoch_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Helper to compute standard epoch evaluation metrics."""
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

    return {"accuracy": acc, "f1": f1, "auc": auc}


class VoxGuardTrainer:
    """
    Standardized trainer for VoxGuard CNN and Transformer classifiers.
    """

    def __init__(
        self,
        arch: str = "cnn",
        metadata_path: Union[str, Path] = "data/processed/metadata.csv",
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 5,
        patience: int = 3,
        random_seed: int = 42,
        device: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = 2000,
        checkpoint_dir: Union[str, Path] = "models",
        history_dir: Union[str, Path] = "results/training_history",
        resume_checkpoint: Optional[Union[str, Path]] = None,
        use_augmentation: bool = False,
    ):
        self.arch = arch.lower().strip()
        self.metadata_path = Path(metadata_path)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.random_seed = random_seed
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.checkpoint_dir = PROJECT_ROOT / checkpoint_dir
        self.history_dir = PROJECT_ROOT / history_dir
        self.resume_checkpoint = resume_checkpoint
        self.use_augmentation = use_augmentation

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Device detection
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cpu":
            torch.set_num_threads(min(6, os.cpu_count() or 4))

        logger.info(f"Initializing VoxGuardTrainer [{self.arch.upper()}] on {self.device}")
        if self.use_augmentation:
            logger.info("Applying SpecAugment frequency & time masking to training set.")

        # Datasets
        train_transform = SpecAugment() if self.use_augmentation else None
        self.train_ds = VoxGuardDataset(
            metadata_path=self.metadata_path,
            split="train",
            max_samples=self.max_train_samples,
            random_seed=random_seed,
            transform=train_transform,
        )
        self.val_ds = VoxGuardDataset(
            metadata_path=self.metadata_path,
            split="val",
            max_samples=self.max_val_samples,
            random_seed=random_seed,
        )

        train_counts = self.train_ds.get_class_counts()
        val_counts = self.val_ds.get_class_counts()
        logger.info(f"Train samples: {len(self.train_ds)} (Real: {train_counts[0]}, Fake: {train_counts[1]})")
        logger.info(f"Val samples:   {len(self.val_ds)} (Real: {val_counts[0]}, Fake: {val_counts[1]})")

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        # Model instantiation
        self.model = get_model(arch=self.arch, device=self.device)

        if self.resume_checkpoint is not None:
            p_res = Path(self.resume_checkpoint)
            if not p_res.is_absolute():
                p_res = PROJECT_ROOT / p_res
            if p_res.is_file():
                logger.info(f"Resuming weights from checkpoint: {p_res}")
                state = torch.load(p_res, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)

        # Dynamic class-weighted loss (BCEWithLogitsLoss)
        pos_weight = self.train_ds.compute_pos_weight().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer & LR adaptation
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.001, mode="min")

    def train_epoch(self) -> Tuple[float, float, float]:
        """Train one full epoch across train_loader."""
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for batch_idx, (x, y, _, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, return_logits=True)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = y.detach().cpu().numpy()

            all_probs.extend(probs.flatten())
            all_targets.extend(targets.flatten())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_l = total_loss / ((batch_idx + 1) * x.size(0))
                pct = ((batch_idx + 1) / len(self.train_loader)) * 100.0
                logger.info(f"  Step [{batch_idx+1:04d}/{len(self.train_loader):04d}] ({pct:5.1f}%) - Loss: {loss.item():.4f} (Avg: {avg_l:.4f})")
                sys.stdout.flush()

            if (batch_idx + 1) % 250 == 0:
                snap_path = self.checkpoint_dir / "voxguard_model.pt"
                save_checkpoint(self.model, snap_path)
                logger.info(f"  [Snapshot] Periodic weights saved to {snap_path.name}")
                sys.stdout.flush()

        epoch_loss = total_loss / len(self.train_loader.dataset)
        m = compute_epoch_metrics(np.array(all_targets), np.array(all_probs))
        return epoch_loss, m["accuracy"], m["auc"]

    @torch.no_grad()
    def evaluate_val(self) -> Tuple[float, float, float, float]:
        """Evaluate model on validation loader."""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for x, y, _, _ in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x, return_logits=True)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = y.cpu().numpy()

            all_probs.extend(probs.flatten())
            all_targets.extend(targets.flatten())

        val_loss = total_loss / len(self.val_loader.dataset)
        m = compute_epoch_metrics(np.array(all_targets), np.array(all_probs))
        return val_loss, m["accuracy"], m["f1"], m["auc"]

    def fit(self) -> Dict[str, Any]:
        """Run complete multi-epoch training pipeline with checkpointing and history tracking."""
        history = []
        best_val_loss = float("inf")
        best_val_auc = 0.0

        canonical_path = self.checkpoint_dir / "voxguard_model.pt"
        arch_best_path = self.checkpoint_dir / f"voxguard_{self.arch}_best.pt"

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc, train_auc = self.train_epoch()
            val_loss, val_acc, val_f1, val_auc = self.evaluate_val()
            self.scheduler.step(val_loss)
            dur = time.time() - t0

            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({dur:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
            )

            ep_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "duration_seconds": dur,
            }
            history.append(ep_record)

            # Checkpoint updates
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc = val_auc
                save_checkpoint(self.model, canonical_path)
                save_checkpoint(self.model, arch_best_path)
                logger.info(f"  --> Best model saved: Val Loss {val_loss:.4f}, AUC {val_auc:.4f}")

            # Early stopping check
            if self.early_stopping(val_loss, model=self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

        # Save training history JSON
        history_file = self.history_dir / f"{self.arch}_training_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({
                "arch": self.arch,
                "epochs": len(history),
                "best_val_loss": best_val_loss,
                "best_val_auc": best_val_auc,
                "total_duration_seconds": total_time,
                "history": history,
            }, f, indent=2)

        return {"history": history, "best_val_loss": best_val_loss, "best_val_auc": best_val_auc}

