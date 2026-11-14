"""
VoxGuard - XAI Visualization Engine
===================================
Fulfills TASK 3 Part 3: High-resolution visual explanation generating side-by-side
and overlaid Grad-CAM attribution heatmaps over the original Mel-spectrogram.

Calibrated to physical time (seconds) and frequency (kHz) coordinates for
major project presentation and courtroom-style forensic reporting.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.preprocessing.config import SAMPLE_RATE, HOP_LENGTH, TARGET_TIME_STEPS, FMAX


def plot_gradcam_spectrogram(
    mel_spectrogram: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: int,
    confidence: float,
    output_path: Optional[Union[str, Path]] = None,
    title_suffix: str = "",
    alpha: float = 0.45,
) -> plt.Figure:
    """
    Generate professional forensic Grad-CAM visualization over Mel-spectrogram.

    Args:
        mel_spectrogram: 2D array of shape (128, 128) representing log-Mel dB.
        heatmap: 2D array of shape (128, 128) representing Grad-CAM attribution in [0, 1].
        predicted_class: 0 (Real) or 1 (Fake).
        confidence: Float in [0.0, 1.0].
        output_path: Optional path to save high-resolution PNG.
        title_suffix: Additional text for the figure title.
        alpha: Transparency factor for heatmap overlay (0.0 to 1.0).

    Returns:
        matplotlib.figure.Figure object.
    """
    # Time axis: total duration = (128 frames * 512 hop) / 16000 = 4.096 seconds
    total_duration = (TARGET_TIME_STEPS * HOP_LENGTH) / SAMPLE_RATE
    extent = [0.0, total_duration, 0.0, FMAX / 1000.0]  # Frequency in kHz (0 to 8 kHz)

    label_str = "FAKE (AI Clone)" if predicted_class == 1 else "REAL (Authentic Voice)"
    conf_pct = confidence * 100.0
    theme_color = "#E63946" if predicted_class == 1 else "#2A9D8F"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#0F141C")  # Professional dark background

    # Panel 1: Original Log Mel-Spectrogram
    ax1 = axes[0]
    ax1.set_facecolor("#1A202C")
    im1 = ax1.imshow(
        mel_spectrogram,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    ax1.set_title("Input Mel-Spectrogram (Acoustic Feature Map)", color="#FFFFFF", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Time (seconds)", color="#A0AEC0", fontsize=10)
    ax1.set_ylabel("Frequency (kHz)", color="#A0AEC0", fontsize=10)
    ax1.tick_params(colors="#CBD5E0", labelsize=9)
    cb1 = fig.colorbar(im1, ax=ax1, format="%+2.0f dB", fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#CBD5E0", labelsize=8)
    cb1.set_label("Log Energy (dB)", color="#A0AEC0", fontsize=9)

    # Panel 2: Grad-CAM Attribution Overlay
    ax2 = axes[1]
    ax2.set_facecolor("#1A202C")
    # Base grayscale/muted spectrogram for contrast
    ax2.imshow(
        mel_spectrogram,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
    )
    # Overlay heatmap with jet / turbo colormap
    im2 = ax2.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="jet",
        alpha=alpha,
        vmin=0.0,
        vmax=1.0,
    )
    ax2.set_title("Grad-CAM Attribution (Decision Influence Heatmap)", color="#FFFFFF", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Time (seconds)", color="#A0AEC0", fontsize=10)
    ax2.set_ylabel("Frequency (kHz)", color="#A0AEC0", fontsize=10)
    ax2.tick_params(colors="#CBD5E0", labelsize=9)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#CBD5E0", labelsize=8)
    cb2.set_label("Attribution Weight [0.0 - 1.0]", color="#A0AEC0", fontsize=9)

    # Global Title Banner
    full_title = (
        f"VoxGuard Forensic Attribution | Verdict: {label_str} | Confidence: {conf_pct:.1f}%"
    )
    if title_suffix:
        full_title += f" | {title_suffix}"
    fig.suptitle(full_title, color="#FFFFFF", fontsize=14, fontweight="heavy", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path is not None:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)

    return fig


def save_visualization(
    mel_spectrogram: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: int,
    confidence: float,
    output_path: Union[str, Path],
    title_suffix: str = "",
) -> str:
    """Convenience wrapper to save plot directly to file."""
    fig = plot_gradcam_spectrogram(
        mel_spectrogram=mel_spectrogram,
        heatmap=heatmap,
        predicted_class=predicted_class,
        confidence=confidence,
        output_path=output_path,
        title_suffix=title_suffix,
    )
    plt.close(fig)
    return str(Path(output_path).resolve())

