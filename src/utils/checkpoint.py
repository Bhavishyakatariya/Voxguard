"""
VoxGuard - Model Checkpoint Management
======================================
Fulfills TASK 2 Section 5: Model Checkpointing.
Saves and loads PyTorch model weights conforming strictly to the Shared Interface Contract.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    save_full_state: bool = False,
) -> Path:
    """
    Save model weights to a .pt file.

    By default, saves the exact model state_dict conforming strictly to the
    Shared Interface Contract #2 (enabling zero-breakage downstream integration).
    If save_full_state=True, saves a dictionary containing model_state, optimizer_state, epoch, and metrics.
    """
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    if save_full_state:
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
    else:
        # Canonical weight serialization
        torch.save(model.state_dict(), path)

    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load weights into a model from a .pt file.
    Handles both direct state_dicts and encapsulated full checkpoint dictionaries.
    """
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")

    checkpoint_data = torch.load(path, map_location=device, weights_only=False)

    metadata = {}
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        metadata["epoch"] = checkpoint_data.get("epoch", None)
        metadata["metrics"] = checkpoint_data.get("metrics", {})
    elif isinstance(checkpoint_data, dict):
        model.load_state_dict(checkpoint_data)
    else:
        raise ValueError(f"Unrecognized checkpoint format in {path}")

    model.to(device)
    return metadata

