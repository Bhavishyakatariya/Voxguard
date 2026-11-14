"""
VoxGuard - Training Command-Line Interface
=========================================
Fulfills TASK 2 Section 4: Training Pipeline.
Configurable entrypoint supporting CNN, Transformer, Dummy Data, Real Data, and Checkpoints.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import VoxGuardTrainer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("VoxGuard.TrainCLI")


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Neural Classifier Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--arch", type=str, default=None, choices=["cnn", "transformer"], help="Model architecture")
    parser.add_argument("--data-source", type=str, default=None, choices=["real", "dummy"], help="Data source toggle")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples (0 = full)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Cap validation samples")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--augment", action="store_true", help="Apply SpecAugment to training set")
    args = parser.parse_args()

    # Load base config
    cfg = load_config(args.config)

    # CLI overrides
    if args.data_source is not None:
        cfg["data_source"] = args.data_source
        if args.data_source == "dummy":
            cfg["metadata_path"] = "data/dummy/metadata.csv"
        else:
            cfg["metadata_path"] = "data/processed/metadata.csv"

    arch = args.arch or cfg.get("model", {}).get("arch", "cnn")
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 5)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 64)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 1e-3)
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else cfg.get("training", {}).get("max_train_samples", 0)
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else cfg.get("training", {}).get("max_val_samples", 2000)
    device = args.device or cfg.get("training", {}).get("device", "auto")
    patience = cfg.get("early_stopping", {}).get("patience", 3)

    metadata_path = cfg["metadata_path"]

    logger.info(f"Loaded config from {args.config} (data_source: {cfg.get('data_source')})")
    logger.info(f"Target metadata path: {metadata_path}")

    trainer = VoxGuardTrainer(
        arch=arch,
        metadata_path=metadata_path,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        patience=patience,
        max_train_samples=max_train_samples if max_train_samples > 0 else None,
        max_val_samples=max_val_samples if max_val_samples > 0 else None,
        device=device,
        resume_checkpoint=args.resume,
        use_augmentation=args.augment,
    )

    trainer.fit()


if __name__ == "__main__":
    main()

