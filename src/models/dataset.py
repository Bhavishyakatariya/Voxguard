"""
VoxGuard - PyTorch Dataset Module (VoxGuardDataset)
===================================================
Implementation of the VoxGuardDataset class conforming to the Shared Interface
Contract and Section 2.7.3 of the VoxGuard SRS.

Features:
- Ingests data/processed/metadata.csv and filters by split ('train', 'val', 'test').
- Efficiently loads (128, 128) float32 Mel-spectrogram .npy feature matrices.
- Returns tensors formatted as (1, 128, 128) with binary target labels.
- Provides balanced sub-sampling option (--max-samples) for rapid CPU training.
- Calculates inverse class weights for balanced BCE loss optimization.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class VoxGuardDataset(Dataset):
    """
    PyTorch Dataset consuming preprocessed VoxGuard Mel-spectrograms.
    """

    def __init__(
        self,
        metadata_path: Union[str, Path] = "data/processed/metadata.csv",
        split: str = "train",
        max_samples: Optional[int] = None,
        random_seed: int = 42,
        transform: Optional[Callable] = None,
        cache_in_memory: bool = False,
    ):
        """
        Args:
            metadata_path: Path to master metadata CSV.
            split: Target partition ('train', 'val', or 'test').
            max_samples: If set, cap total samples (balanced between Real and Fake).
            random_seed: Seed for balanced sampling reproducibility.
            transform: Optional transform callable applied to the spectrogram tensor.
            cache_in_memory: Whether to cache loaded arrays in memory.
        """
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.is_absolute():
            self.metadata_path = PROJECT_ROOT / self.metadata_path

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_path.resolve()}")

        self.split = split.lower().strip()
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, torch.Tensor] = {}

        # 1. Load metadata
        df = pd.read_csv(self.metadata_path)
        if "split" not in df.columns or "filepath" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Malformed metadata CSV. Required columns missing: {df.columns.tolist()}")

        # 2. Filter by split
        split_df = df[df["split"].str.lower() == self.split].copy()
        if len(split_df) == 0:
            raise ValueError(f"No records found in metadata for split '{self.split}'")

        # 3. Optional balanced sub-sampling for fast experimentation / CPU training
        if max_samples is not None and max_samples > 0 and max_samples < len(split_df):
            real_df = split_df[split_df["label"] == 0]
            fake_df = split_df[split_df["label"] == 1]

            n_per_class = max_samples // 2
            n_real = min(len(real_df), n_per_class)
            n_fake = min(len(fake_df), n_per_class)

            sampled_real = real_df.sample(n=n_real, random_state=random_seed)
            sampled_fake = fake_df.sample(n=n_fake, random_state=random_seed)

            split_df = pd.concat([sampled_real, sampled_fake]).sample(
                frac=1.0, random_state=random_seed
            ).reset_index(drop=True)

        self.df = split_df.reset_index(drop=True)
        self.samples: List[Tuple[str, int, str]] = [
            (row["filepath"], int(row["label"]), str(row.get("dataset", "unknown")))
            for _, row in self.df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            x_tensor: FloatTensor of shape (1, 128, 128)
            y_tensor: FloatTensor of shape (1,) with label 0.0 or 1.0
            filepath: Relative path string to .npy file
            dataset: Source dataset ('asvspoof' or 'inthewild')
        """
        if self.cache_in_memory and idx in self._cache:
            x_tensor, y_tensor, rel_path, dataset = self._cache[idx]
            return x_tensor, y_tensor, rel_path, dataset

        rel_path, label, dataset = self.samples[idx]
        abs_path = PROJECT_ROOT / rel_path

        if not abs_path.is_file():
            # Fallback check inside data/processed directly
            if not abs_path.exists():
                raise FileNotFoundError(f"Feature file does not exist: {abs_path}")

        try:
            arr = np.load(abs_path, mmap_mode="r")
        except Exception as e:
            raise IOError(f"Failed to read .npy array from {abs_path}: {e}")

        # Ensure correct shape (128, 128)
        if arr.shape != (128, 128):
            raise ValueError(f"Spectrogram tensor shape {arr.shape} does not match contract (128, 128) for {rel_path}")

        # Convert to float32 PyTorch tensor (1, 128, 128) with fast memory copy
        x_tensor = torch.from_numpy(np.array(arr, copy=True)).float().unsqueeze(0)
        y_tensor = torch.tensor([float(label)], dtype=torch.float32)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        if self.cache_in_memory:
            self._cache[idx] = (x_tensor, y_tensor, rel_path, dataset)

        return x_tensor, y_tensor, rel_path, dataset

    def get_class_counts(self) -> Dict[int, int]:
        """Return counts of Real (0) and Fake (1) samples in this partition."""
        real_count = sum(1 for _, lbl, _ in self.samples if lbl == 0)
        fake_count = sum(1 for _, lbl, _ in self.samples if lbl == 1)
        return {0: real_count, 1: fake_count}

    def compute_pos_weight(self) -> torch.Tensor:
        """
        Compute positive class weight for BCEWithLogitsLoss to handle class imbalance:
        pos_weight = N_negative / N_positive = N_real / N_fake.
        """
        counts = self.get_class_counts()
        n_real = max(1, counts[0])
        n_fake = max(1, counts[1])
        weight_val = n_real / float(n_fake)
        return torch.tensor([weight_val], dtype=torch.float32)

