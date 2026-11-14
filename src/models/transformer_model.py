"""
VoxGuard - Audio Spectrogram Transformer Classifier (Model B)
=============================================================
Fulfills TASK 2 Section 3: Model B: Transformer-Based Classifier.
- Accepts input features of shape (B, 1, 128, 128) Mel-spectrogram
- 16x16 Patch Projection into 64 patch tokens
- Learnable [CLS] token prepended to token sequence
- 1D learnable sinusoidal/normal positional embeddings
- 4-layer Transformer Encoder (Multi-head self-attention with GELU)
- Produces two-class output: Real (0) and Fake (1) via P(fake) in [0.0, 1.0]
- Exposes `self.target_layer` for explainability / attention extraction
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from src.models.voxguard_transformer import VoxGuardTransformer


class TransformerClassifier(VoxGuardTransformer):
    """
    Modular Transformer-Based Classifier for deepfake audio classification.
    Directly subclasses and aliases VoxGuardTransformer to guarantee 100% contract compliance.
    """
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
        )


__all__ = ["TransformerClassifier", "VoxGuardTransformer"]

