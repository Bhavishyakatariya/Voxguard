"""
VoxGuard Data Modules
"""

from src.data.dataset import VoxGuardDataset
from src.data.dummy_data import generate_dummy_dataset

__all__ = ["VoxGuardDataset", "generate_dummy_dataset"]

