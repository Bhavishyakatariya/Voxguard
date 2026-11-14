"""
VoxGuard - Stratified Train / Validation / Test Splitting
=========================================================
Implements stratified dataset splitting strictly adhering to:
- Training: 70%
- Validation: 15%
- Testing: 15%
- Fixed reproducible random seed (default: 42)
- Zero sample leakage (disjoint sets across splits)
- Joint stratification preserving both class balance (real/fake)
  and dataset balance (asvspoof/inthewild) across all splits.
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Set

from .config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    SPLITS,
)
from .dataset_loader import DatasetSample

logger = logging.getLogger(__name__)


def stratified_split(
    samples: List[DatasetSample],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, List[DatasetSample]]:
    """
    Split a list of DatasetSample objects into stratified train, val, and test subsets.

    Stratification is performed jointly over (dataset, label) strata to guarantee
    that real vs fake ratio and dataset origin proportions are preserved equally
    across all three splits.

    Args:
        samples: List of DatasetSample objects to split.
        train_ratio: Fraction for training set (default: 0.70).
        val_ratio: Fraction for validation set (default: 0.15).
        test_ratio: Fraction for testing set (default: 0.15).
        random_seed: Seed for pseudo-random number generator (default: 42).

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing a list of DatasetSample.

    Raises:
        ValueError: If split ratios do not sum to approximately 1.0 or if samples list is empty.
    """
    if not samples:
        return {split: [] for split in SPLITS}

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    # Initialize random generator with deterministic seed
    rng = random.Random(random_seed)

    # Group samples by joint stratum: (dataset, label)
    strata: Dict[tuple, List[DatasetSample]] = defaultdict(list)
    for sample in samples:
        key = (sample.dataset, sample.label)
        strata[key].append(sample)

    split_results: Dict[str, List[DatasetSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Split each stratum independently
    for stratum_key, stratum_samples in strata.items():
        # Shuffle stratum deterministically
        shuffled = list(stratum_samples)
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Allocate remainder to test to ensure exact total conservation
        n_test = n - n_train - n_val

        # Handle rounding edge cases for tiny sets
        if n >= 3 and n_val == 0:
            n_val = 1
            n_train -= 1
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train -= 1

        train_part = shuffled[:n_train]
        val_part = shuffled[n_train:n_train + n_val]
        test_part = shuffled[n_train + n_val:]

        split_results["train"].extend(train_part)
        split_results["val"].extend(val_part)
        split_results["test"].extend(test_part)

        logger.debug(
            f"Stratum {stratum_key}: Total={n} -> Train={len(train_part)}, Val={len(val_part)}, Test={len(test_part)}"
        )

    # Final shuffle per split to interleave strata
    for split_name in SPLITS:
        rng.shuffle(split_results[split_name])

    # Rigorous verification: Disjoint sets (Zero Data Leakage)
    train_ids: Set[str] = {s.sample_id for s in split_results["train"]}
    val_ids: Set[str] = {s.sample_id for s in split_results["val"]}
    test_ids: Set[str] = {s.sample_id for s in split_results["test"]}

    leakage_train_val = train_ids.intersection(val_ids)
    leakage_train_test = train_ids.intersection(test_ids)
    leakage_val_test = val_ids.intersection(test_ids)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError(
            f"Critical Data Leakage Detected! Intersections: "
            f"train&val={len(leakage_train_val)}, "
            f"train&test={len(leakage_train_test)}, "
            f"val&test={len(leakage_val_test)}"
        )

    return split_results

