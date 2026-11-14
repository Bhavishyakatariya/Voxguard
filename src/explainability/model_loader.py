"""
VoxGuard - XAI Model Loader
===========================
Safely restores the actual trained VoxGuard classifier weights from existing
checkpoints into evaluation mode. Verifies parameter integrity, compatibility,
and the target convolutional layer hook for Grad-CAM.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.models.cnn_model import VoxGuardCNN
from src.utils.logger import get_logger

logger = get_logger("VoxGuard.XAI.ModelLoader")

DEFAULT_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "models" / "voxguard_cnn_best.pt",
    PROJECT_ROOT / "models" / "voxguard_model.pt",
]


def load_trained_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[VoxGuardCNN, torch.device]:
    """
    Load the actual trained VoxGuard CNN classifier.

    Args:
        checkpoint_path: Explicit path to .pt checkpoint. If None, resolves
                         to models/voxguard_cnn_best.pt or models/voxguard_model.pt.
        device: 'cpu', 'cuda', or None (auto-detect).

    Returns:
        Tuple of (model in eval mode, torch.device).

    Raises:
        FileNotFoundError: If the specified checkpoint does not exist.
        RuntimeError: If checkpoint cannot be loaded or parameters mismatch.
    """
    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)

    resolved_ckpt: Optional[Path] = None
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(
                f"Trained VoxGuard checkpoint not found at: {p.resolve()}. "
                "Never falling back to a dummy model."
            )
        resolved_ckpt = p
    else:
        for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
            if candidate.is_file():
                resolved_ckpt = candidate
                break
        if resolved_ckpt is None:
            raise FileNotFoundError(
                f"No trained VoxGuard checkpoint found in {[str(c) for c in DEFAULT_CHECKPOINT_CANDIDATES]}. "
                "Ensure Task 2 model training has completed."
            )

    logger.info(f"Loading trained VoxGuard model from: {resolved_ckpt}")
    logger.info(f"Target execution device: {selected_device}")

    # Instantiate model
    model = VoxGuardCNN()

    try:
        state_dict = torch.load(resolved_ckpt, map_location=selected_device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialize checkpoint {resolved_ckpt}: {exc}") from exc

    # Handle dictionary wrapping if saved under 'state_dict' or 'model_state_dict'
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Verify all expected keys are present
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        raise RuntimeError(
            f"Checkpoint is incompatible with VoxGuardCNN. Missing keys ({len(missing)}): {list(missing)[:5]}"
        )

    model.load_state_dict(state_dict, strict=False)
    model.to(selected_device)
    model.eval()

    # Ensure target_layer is correctly configured for Grad-CAM
    if not hasattr(model, "target_layer") or model.target_layer is None:
        model.target_layer = model.conv4
        logger.warning("target_layer attribute was missing; attached reference to model.conv4.")

    logger.info(f"Model loaded successfully. Target layer: {model.target_layer}")
    return model, selected_device

