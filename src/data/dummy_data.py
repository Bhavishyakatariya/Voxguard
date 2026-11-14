"""
VoxGuard - Synthetic Dummy Data Generator
=========================================
Fulfills TASK 2 Section 2: Create Dummy Data For Initial Development.

Generates isolated dummy feature files and metadata adhering strictly to
the Shared Interface Contract:
- Format: .npy
- Shape: (128, 128) float32 Mel-spectrogram
- Metadata columns: filepath,label,dataset,split
- Labels: 0 (Real), 1 (Fake)
- Datasets: 'asvspoof', 'inthewild'
- Splits: 'train', 'val', 'test'

IMPORTANT:
This data is stored isolated in `data/dummy/` and is strictly used to verify
pipeline execution and dry runs without modifying or polluting real preprocessed corpora.
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def generate_dummy_dataset(
    output_dir: Optional[Path] = None,
    samples_per_split: int = 24,
    random_seed: int = 42,
) -> Path:
    """
    Generate synthetic dummy dataset with (128, 128) Mel-spectrograms.

    Args:
        output_dir: Root destination directory (defaults to PROJECT_ROOT / 'data' / 'dummy')
        samples_per_split: Number of dummy samples per partition
        random_seed: Reproducibility seed

    Returns:
        Path to generated metadata.csv
    """
    np.random.seed(random_seed)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "dummy"
    else:
        output_dir = Path(output_dir)

    splits = ["train", "val", "test"]
    datasets = ["asvspoof", "inthewild"]
    metadata_rows = []

    print(f"[DummyData] Generating dummy dataset in: {output_dir}")

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_split):
            label = 0 if i % 2 == 0 else 1
            dataset = datasets[i % len(datasets)]
            filename = f"dummy_{split}_{dataset}_{i:04d}_{label}.npy"
            file_abs_path = split_dir / filename

            # Generate synthetic float32 spectrogram of exact shape (128, 128) in [0.0, 1.0]
            # Real speech has distinct spectral peaks, fake has slight artifacts
            base_noise = np.random.uniform(0.0, 0.15, size=(128, 128)).astype(np.float32)
            if label == 0:
                # Simulated harmonic formant bands
                for freq_band in [20, 45, 75]:
                    base_noise[freq_band:freq_band + 4, :] += 0.5
            else:
                # Simulated high-frequency phase artifact
                base_noise[90:110, :] += 0.4
                base_noise[:, 50:70] += 0.3

            dummy_spec = np.clip(base_noise, 0.0, 1.0).astype(np.float32)
            np.save(file_abs_path, dummy_spec)

            # Store relative path in metadata conforming to contract
            rel_path = f"data/dummy/{split}/{filename}"
            metadata_rows.append({
                "filepath": rel_path,
                "label": int(label),
                "dataset": dataset,
                "split": split,
            })

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"[DummyData] Generated {len(metadata_df)} dummy samples across {splits}")
    print(f"[DummyData] Metadata saved to: {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dummy dataset for VoxGuard")
    parser.add_argument("--samples-per-split", type=int, default=24, help="Samples per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dummy_dataset(samples_per_split=args.samples_per_split, random_seed=args.seed)

