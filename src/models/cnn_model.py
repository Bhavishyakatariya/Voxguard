"""
VoxGuard - 2D Convolutional Neural Network Classifier (Model A)
===============================================================
Fulfills TASK 2 Section 3: Model A: CNN Classifier.
- Accepts input features of shape (B, 1, 128, 128) Mel-spectrogram
- 4 Convolutional blocks with BatchNorm, ReLU, MaxPool(2x2)
- Dropout to reduce overfitting
- Adaptive average pooling and fully connected classification head
- Outputs scalar probability P(fake) in [0.0, 1.0] (0 = Real, 1 = Fake)
- Exposes `self.target_layer = self.conv4` for Grad-CAM explainability
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from src.models.voxguard_cnn import VoxGuardCNN


class CNNClassifier(VoxGuardCNN):
    """
    Modular CNN Classifier for deepfake audio classification.
    Directly subclasses and aliases VoxGuardCNN to guarantee 100% contract compliance
    and weight compatibility with existing checkpoints.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__(dropout_rate=dropout_rate)


__all__ = ["CNNClassifier", "VoxGuardCNN"]

