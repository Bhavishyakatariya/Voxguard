"""
VoxGuard - Model Factory
========================
Provides factory loader functions to instantiate, load, and configure
deepfake detection models (CNN and Transformer) conforming to the VoxGuard contract.
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from src.models.cnn_model import CNNClassifier
from src.models.transformer_model import TransformerClassifier


def get_model(
    arch: str = "cnn",
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    dropout_rate: Optional[float] = None,
) -> nn.Module:
    """
    Factory function to instantiate VoxGuard models.

    Args:
        arch: 'cnn' or 'transformer'
        checkpoint_path: Optional path to serialized weights (.pt file)
        device: Device to place the model on ('cpu' or 'cuda')
        dropout_rate: Optional override for model dropout

    Returns:
        Instantiated PyTorch model ready for training or evaluation.
    """
    arch_norm = arch.lower().strip()
    if arch_norm in ("cnn", "voxguardcnn"):
        kwargs = {}
        if dropout_rate is not None:
            kwargs["dropout_rate"] = dropout_rate
        model = CNNClassifier(**kwargs)
    elif arch_norm in ("transformer", "voxguardtransformer", "ast"):
        kwargs = {}
        if dropout_rate is not None:
            kwargs["dropout_rate"] = dropout_rate
        model = TransformerClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Supported options: 'cnn', 'transformer'")

    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p.resolve()}")
        state_dict = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)

    return model


__all__ = ["get_model"]

