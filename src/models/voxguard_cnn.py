"""
VoxGuard - 2D Convolutional Neural Network Classifier (VoxGuardCNN)
===================================================================
Implementation of the 4-block deep 2D Convolutional Neural Network
specified in the VoxGuard SRS (Appendix B & Section 2.7.3).

Key Architectural Specifications:
- Input: (B, 1, 128, 128) float32 Mel-spectrogram in [0.0, 1.0]
- Conv1: 1 -> 32 channels (3x3, pad 1) + BN + ReLU + MaxPool(2x2) -> (32, 64, 64)
- Conv2: 32 -> 64 channels (3x3, pad 1) + BN + ReLU + MaxPool(2x2) -> (64, 32, 32)
- Conv3: 64 -> 128 channels (3x3, pad 1) + BN + ReLU + MaxPool(2x2) -> (128, 16, 16)
- Conv4: 128 -> 256 channels (3x3, pad 1) + BN + ReLU + MaxPool(2x2) -> (256, 8, 8)
- Target Layer: self.target_layer points to self.conv4 for Grad-CAM hooking
- Head: AdaptiveAvgPool(1,1) -> Flatten -> Dropout(0.3) -> FC(256, 64) -> ReLU -> Dropout(0.2) -> FC(64, 1) -> Sigmoid
- Output: (B, 1) scalar P(fake) in [0.0, 1.0]
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn


class VoxGuardCNN(nn.Module):
    """
    4-Block Deep 2D CNN with Grad-CAM hooking capability.
    Conforms strictly to the VoxGuard Shared Interface Contract #2.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super(VoxGuardCNN, self).__init__()

        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 128 -> 256 (Exposes target_layer for Grad-CAM)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Target layer explicitly exposed for Amrita's Grad-CAM hooking
        self.target_layer = self.conv4

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 64)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass through convolutional feature extractor blocks.
        """
        if x.dim() == 3:
            # Add batch dimension if single sample (1, 128, 128) -> (1, 1, 128, 128)
            x = x.unsqueeze(0)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        return x

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass returning P(fake) in [0.0, 1.0], or raw logits if return_logits=True.
        """
        feats = self.forward_features(x)
        pooled = self.global_pool(feats)
        flattened = torch.flatten(pooled, 1)

        z = self.dropout1(flattened)
        z = self.relu_fc(self.fc1(z))
        z = self.dropout2(z)
        logits = self.fc2(z)

        if return_logits:
            return logits
        return self.sigmoid(logits)


def load_cnn_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> VoxGuardCNN:
    """
    Factory loader to instantiate and initialize VoxGuardCNN.

    Args:
        checkpoint_path: Path to serialized PyTorch state_dict (.pt file).
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded VoxGuardCNN model in eval mode.
    """
    model = VoxGuardCNN()
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {p.resolve()}")
        state_dict = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

