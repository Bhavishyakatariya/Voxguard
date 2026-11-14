"""
VoxGuard - Gradient-Weighted Class Activation Mapping (Grad-CAM)
================================================================
Fulfills TASK 3 Part 1: Scientific Grad-CAM implementation.
Captures convolutional feature activations and backward gradients at
`model.target_layer` (conv4) to generate 2D attribution heatmaps.

Supports both FAKE and REAL class explanations symmetrically.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("VoxGuard.XAI.GradCAM")


class GradCAM:
    """
    Grad-CAM engine for VoxGuard 2D Convolutional Classifiers.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: Trained VoxGuard PyTorch classifier.
            target_layer: Convolutional layer to hook. Defaults to model.target_layer.
        """
        self.model = model
        self.model.eval()

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "target_layer") and model.target_layer is not None:
            self.target_layer = model.target_layer
        elif hasattr(model, "conv4"):
            self.target_layer = model.conv4
        else:
            raise AttributeError("Target layer not found on model. Ensure model has self.target_layer or self.conv4.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input_tensor, output_tensor):
            self.activations = output_tensor.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h_f = self.target_layer.register_forward_hook(forward_hook)
        h_b = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h_f, h_b])

    def remove_hooks(self):
        """Remove PyTorch hooks to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, float]:
        """
        Generate normalized Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Tensor of shape (1, 1, 128, 128) or (1, 128, 128).
            target_class: Optional class to explain (0 = Real, 1 = Fake).
                          If None, explains the predicted class.

        Returns:
            Tuple of:
                heatmap: 2D numpy array of shape (128, 128) with values in [0.0, 1.0].
                predicted_class: 0 (Real) or 1 (Fake).
                confidence: Prediction confidence for the predicted class in [0.0, 1.0].
                prob_fake: Raw P(fake) output in [0.0, 1.0].
        """
        # Ensure correct batch dimensions (1, 1, 128, 128)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        device = next(self.model.parameters()).device
        x = input_tensor.to(device).clone().requires_grad_(True)

        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Forward pass: obtain raw logits for gradient backprop
        logits = self.model(x, return_logits=True)
        prob_fake = torch.sigmoid(logits).item()

        # Class determination: 0 = Real, 1 = Fake
        predicted_class = 1 if prob_fake >= 0.5 else 0
        confidence = prob_fake if predicted_class == 1 else (1.0 - prob_fake)

        # Decide which score to backpropagate
        explain_class = predicted_class if target_class is None else target_class

        if explain_class == 1:
            # Positive logit corresponds to Fake class score
            score = logits[0, 0]
        else:
            # Negative logit corresponds to Real class score
            score = -logits[0, 0]

        # Backward pass
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to intercept activations or gradients.")

        # Global average pooling of gradients: alpha_k = (1 / Z) * sum(gradients)
        # activations shape: (1, Channels, H, W), gradients shape: (1, Channels, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, Channels, 1, 1)

        # Weighted combination of feature maps: L = ReLU(sum(alpha_k * A_k))
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to match original Mel-spectrogram shape (128, 128)
        target_h, target_w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam_resized = F.interpolate(
            cam, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        cam_np = cam_resized.squeeze().cpu().numpy()

        # Min-max normalization into [0.0, 1.0] with numerical stability guard
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())

        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
        return heatmap, predicted_class, confidence, prob_fake

