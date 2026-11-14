"""
VoxGuard - Explainability Module (Task 3: Amrita)
=================================================
Final Explainable AI (XAI) implementation for VoxGuard.
Delivers genuine, scientifically grounded visual (Grad-CAM) and natural language
explanations for deepfake audio predictions using the actual trained classifier.
"""

from .model_loader import load_trained_model
from .gradcam import GradCAM
from .caption_generator import generate_explanation_caption
from .visualization import plot_gradcam_spectrogram, save_visualization
from .explain import explain

__all__ = [
    "load_trained_model",
    "GradCAM",
    "generate_explanation_caption",
    "plot_gradcam_spectrogram",
    "save_visualization",
    "explain",
]

