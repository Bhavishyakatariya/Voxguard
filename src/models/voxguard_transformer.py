"""
VoxGuard - Audio Spectrogram Transformer Classifier (VoxGuardTransformer)
========================================================================
Implementation of the Audio Spectrogram Transformer (AST) specified in the
VoxGuard SRS (Appendix B & Section 2.7.3) and Section 4.3 of the Project Report.

Key Architectural Specifications:
- Input: (B, 1, 128, 128) Mel-spectrogram
- Patch Size: 16x16 non-overlapping patches -> (128/16) x (128/16) = 64 patches
- Patch Projection: Conv2D(1, 128, kernel=16, stride=16) -> (B, 128, 8, 8) -> (B, 64, 128)
- Learnable [CLS] Token: prepended to sequence -> length N = 65
- 1D Positional Embeddings: learnable parameter (1, 65, 128)
- Transformer Encoder: 4 layers, 4 attention heads, dim_feedforward=512, GELU activations
- Head: LayerNorm(128) -> Linear(128, 1) -> Sigmoid -> P(fake) in [0.0, 1.0]
"""

import math
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn


class VoxGuardTransformer(nn.Module):
    """
    Audio Spectrogram Transformer for Deepfake Speech Classification.
    Conforms strictly to the VoxGuard Shared Interface Contract #2.
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
        super(VoxGuardTransformer, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)  # 8 * 8 = 64
        self.embed_dim = embed_dim

        # 1. Patch Embedding via Conv2D
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Learnable [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional Embeddings (N = 64 patches + 1 [CLS] = 65 tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        # 4. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # Target layer explicitly exposed for XAI / attention visualization
        self.target_layer = self.transformer_encoder.layers[-1]

        # 5. Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal distributions."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract token representations through transformer blocks.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Patch projection: (B, 1, 128, 128) -> (B, embed_dim, 8, 8)
        x_patches = self.patch_embed(x)
        # Flatten spatial dimensions: (B, embed_dim, 64) -> transpose to (B, 64, embed_dim)
        x_tokens = x_patches.flatten(2).transpose(1, 2)

        # Prepend [CLS] token: (B, 1, embed_dim) -> (B, 65, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_tokens), dim=1)

        # Add positional embedding
        x_seq = self.pos_drop(x_seq + self.pos_embed)

        # Pass through Transformer Encoder
        out_seq = self.transformer_encoder(x_seq)
        return out_seq

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass returning P(fake) in [0.0, 1.0], or raw logits if return_logits=True.
        """
        features = self.forward_features(x)
        # Extract [CLS] token at index 0: (B, embed_dim)
        cls_out = features[:, 0]
        normed = self.norm(cls_out)
        logits = self.head(normed)

        if return_logits:
            return logits
        return self.sigmoid(logits)


def load_transformer_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> VoxGuardTransformer:
    """
    Factory loader to instantiate and initialize VoxGuardTransformer.

    Args:
        checkpoint_path: Path to serialized PyTorch state_dict (.pt file).
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded VoxGuardTransformer model in eval mode.
    """
    model = VoxGuardTransformer()
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {p.resolve()}")
        state_dict = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

