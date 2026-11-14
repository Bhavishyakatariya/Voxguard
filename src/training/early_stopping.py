"""
VoxGuard - Early Stopping Utility
=================================
Fulfills TASK 2 Section 4: Early Stopping to prevent overfitting.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Args:
            patience: How many epochs to wait after last time validation metric improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            mode: 'min' for loss (lower is better), 'max' for AUC/accuracy (higher is better).
            verbose: If True, prints status messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower().strip()
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[dict] = None

    def __call__(self, val_metric: float, model: Optional[nn.Module] = None) -> bool:
        """
        Update early stopping tracker with new validation metric.
        Returns True if early stopping criteria met.
        """
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop

