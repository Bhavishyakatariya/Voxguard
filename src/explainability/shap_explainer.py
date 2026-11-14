"""
VoxGuard - Optional SHAP Explainer Wrapper
==========================================
Fulfills TASK 3 Part 7: Optional game-theoretic SHAP explanation support.
Provides a graceful wrapper around shap.GradientExplainer if SHAP is installed,
or documents its theoretical comparison with Grad-CAM without throwing linter errors.
"""

from typing import Optional, Any
import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger("VoxGuard.XAI.SHAP")


class OptionalSHAPExplainer:
    """
    Optional SHAP Explainer using game-theoretic Shapley value attribution.
    Gracefully handles environments where SHAP is not installed.
    """

    def __init__(self, model: nn.Module, background_samples: Optional[torch.Tensor] = None):
        self.model = model
        self.explainer: Optional[Any] = None
        self.is_available: bool = False

        try:
            # Dynamic import to avoid IDE / linter unresolved import warnings
            import importlib
            shap_module = importlib.import_module("shap")
            self.is_available = True
            logger.info(f"SHAP library detected (version: {getattr(shap_module, '__version__', 'unknown')}).")
            if background_samples is not None:
                self.explainer = shap_module.GradientExplainer(self.model, background_samples)
        except (ImportError, ModuleNotFoundError):
            self.is_available = False
            self.explainer = None
            logger.info("SHAP is not installed. Grad-CAM remains the primary, real-time XAI method.")

    def explain(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Generate SHAP attribution values if SHAP is installed and initialized.
        Returns None gracefully when SHAP is absent.
        """
        if not self.is_available or self.explainer is None:
            return None

        shap_values = self.explainer.shap_values(input_tensor)
        return np.array(shap_values)
