"""
VoxGuard - Spectrogram Visualization Module
===========================================
Generates diagnostic visual plots comparing extracted Mel-spectrograms:
- Loads processed .npy feature matrices (128, 128)
- Displays side-by-side or grid comparisons of Real (0) and Fake (1) samples
- Annotates physical acoustic scales:
    * Y-axis: Mel Frequency Bands (0 Hz - 8,000 Hz)
    * X-axis: Time Frames (0 - 128)
    * Colorbar: Normalized Log-Decibel Energy (dB)
- Saves publication-quality figures without altering feature files.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .config import (
    PROJECT_ROOT,
    METADATA_FILE,
    VISUALIZATION_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    FMIN,
    FMAX,
    LABEL_REAL,
    LABEL_FAKE,
    TARGET_SHAPE,
)

logger = logging.getLogger(__name__)


def plot_mel_spectrograms(
    samples_to_plot: List[Tuple[Path, int, str, str]],
    output_path: Optional[Path] = None,
    figure_title: str = "VoxGuard - Mel-Spectrogram Inspection (Real vs Fake)",
) -> Optional[Path]:
    """
    Render and save a diagnostic comparison figure of Mel-spectrograms.

    Args:
        samples_to_plot: List of tuples containing:
            (npy_filepath, label, dataset_name, split_name)
        output_path: Destination path for PNG plot. Defaults to
            data/processed/visualizations/spectrogram_inspection.png.
        figure_title: Main title for the generated figure.

    Returns:
        Path to the saved figure, or None if no samples could be loaded.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend safe for all environments
    import matplotlib.pyplot as plt

    if not samples_to_plot:
        logger.warning("No samples provided to plot_mel_spectrograms.")
        return None

    # Load and validate matrices
    valid_data = []
    for fpath, label, dataset, split in samples_to_plot:
        p = Path(fpath)
        if not p.is_file():
            p = PROJECT_ROOT / fpath
        if p.is_file():
            matrix = np.load(p)
            valid_data.append((matrix, p.name, label, dataset, split))

    if not valid_data:
        logger.warning("No valid .npy feature files found for visualization.")
        return None

    n_samples = len(valid_data)
    cols = min(n_samples, 4)
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle(figure_title, fontsize=15, fontweight="bold", y=0.98)

    for idx, (matrix, fname, label, dataset, split) in enumerate(valid_data):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        label_text = "REAL (Bonafide)" if label == LABEL_REAL else "FAKE (Spoof)"
        border_color = "#2ca02c" if label == LABEL_REAL else "#d62728"

        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            extent=[0, matrix.shape[1], FMIN, FMAX],
        )

        ax.set_title(
            f"[{label_text}]\n{fname}\n({dataset.upper()} - {split})",
            fontsize=10,
            fontweight="bold",
            color=border_color,
            pad=8,
        )
        ax.set_xlabel("Time Frames (0-128)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)

        # Highlight box around subplot
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0)

        cbar = fig.colorbar(im, ax=ax, format="%+2.0f dB", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Hide unused subplot slots
    for idx in range(n_samples, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path is None:
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        output_path = VISUALIZATION_DIR / "mel_spectrogram_inspection.png"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Spectrogram visualization saved successfully to: {output_path}")
    return output_path


def visualize_from_metadata(
    metadata_csv: Optional[Path] = None,
    num_samples_per_class: int = 2,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Select balanced Real and Fake samples from metadata.csv and plot them.

    Args:
        metadata_csv: Path to metadata.csv. Defaults to config.METADATA_FILE.
        num_samples_per_class: Number of real and fake samples to visualize.
        output_path: Destination path for figure PNG.

    Returns:
        Path to generated figure.
    """
    csv_file = Path(metadata_csv if metadata_csv is not None else METADATA_FILE).resolve()
    if not csv_file.is_file():
        logger.error(f"Cannot visualize: metadata file not found at {csv_file}")
        return None

    real_samples = []
    fake_samples = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = int(row["label"])
            sample_tuple = (
                Path(row["filepath"]),
                lbl,
                row["dataset"],
                row["split"],
            )
            if lbl == LABEL_REAL and len(real_samples) < num_samples_per_class:
                real_samples.append(sample_tuple)
            elif lbl == LABEL_FAKE and len(fake_samples) < num_samples_per_class:
                fake_samples.append(sample_tuple)

            if len(real_samples) >= num_samples_per_class and len(fake_samples) >= num_samples_per_class:
                break

    selected = real_samples + fake_samples
    if not selected:
        logger.warning("No samples found in metadata.csv to visualize.")
        return None

    return plot_mel_spectrograms(selected, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(description="VoxGuard - Visualize Processed Mel-Spectrograms")
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(METADATA_FILE),
        help=f"Path to metadata.csv (default: {METADATA_FILE})",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of Real and Fake samples to display (default: 2 each)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output PNG path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = visualize_from_metadata(
        metadata_csv=Path(args.metadata),
        num_samples_per_class=args.samples_per_class,
        output_path=Path(args.output) if args.output else None,
    )
    if out:
        print(f"[VoxGuard] Visualization created at: {out}")
    else:
        print("[VoxGuard] Visualization could not be completed.")


if __name__ == "__main__":
    main()

