"""
VoxGuard - Deep Neural Classification Engine
============================================
Exposes dual classification architectures:
1. VoxGuardCNN / CNNClassifier: 4-block 2D CNN with Grad-CAM hooking capability (self.target_layer)
2. VoxGuardTransformer / TransformerClassifier: 16x16 patch Audio Spectrogram Transformer (AST)
3. get_model: Factory loader for instantiating and loading model weights

Also provides backwards-compatible aliases for explainability and dashboard integration.
"""

from src.models.voxguard_cnn import VoxGuardCNN, load_cnn_model
from src.models.voxguard_transformer import VoxGuardTransformer, load_transformer_model
from src.models.voxguard_net import VoxGuardNet
from src.models.cnn_model import CNNClassifier
from src.models.transformer_model import TransformerClassifier
from src.models.model_factory import get_model
from src.models.dataset import VoxGuardDataset

__all__ = [
    "VoxGuardCNN",
    "CNNClassifier",
    "load_cnn_model",
    "VoxGuardTransformer",
    "TransformerClassifier",
    "load_transformer_model",
    "VoxGuardNet",
    "get_model",
    "VoxGuardDataset",
]
