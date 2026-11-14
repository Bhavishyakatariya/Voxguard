"""
VoxGuard - Backwards-Compatible Model Wrapper (voxguard_net.py)
==============================================================
Provides drop-in compatibility for Amrita's explainability module (explain.py)
and Hiral's Streamlit dashboard (src/dashboard/app.py).

Aliases VoxGuardCNN as VoxGuardNet and exposes load_model() so downstream
components function without modifying any import statements.
"""

from pathlib import Path
from typing import Optional, Union
import torch

from src.models.voxguard_cnn import VoxGuardCNN, load_cnn_model


# Backwards compatibility class alias
VoxGuardNet = VoxGuardCNN


def load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> VoxGuardCNN:
    """
    Backwards-compatible model loader for explainability and dashboard modules.
    """
    return load_cnn_model(checkpoint_path=checkpoint_path, device=device)

