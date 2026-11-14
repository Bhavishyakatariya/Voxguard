"""
VoxGuard - Main Audio Preprocessing Pipeline
============================================
Orchestrates end-to-end dataset ingestion, stratified train/val/test splitting,
audio preprocessing, (128, 128) Mel-spectrogram extraction, metadata tracking,
and visual verification.

Interface Contract Enforcement:
- Output feature path: data/processed/{split}/{filename}.npy
- Feature dimension: Exactly (128, 128)
- Metadata file: data/processed/metadata.csv
- Metadata columns: filepath,label,dataset,split
- Label mapping: 0 = Real, 1 = Fake
- Split distribution: 70% train, 15% val, 15% test
- Safe append & deduplication for metadata
- Resume capability: skips already-processed files
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Enable relative and absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.config import (
    ensure_directories,
    get_split_dir,
    PROJECT_ROOT,
    PROCESSED_DIR,
    ASVSPOOF_RAW_DIR,
    INTHEWILD_RAW_DIR,
    METADATA_FILE,
    METADATA_COLUMNS,
    LABEL_REAL,
    LABEL_FAKE,
    LABEL_NAME_MAP,
    DATASET_ASVSPOOF,
    DATASET_INTHEWILD,
    SPLITS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    TARGET_SHAPE,
    SAMPLE_RATE,
)
from src.preprocessing.dataset_loader import (
    DatasetSample,
    load_asvspoof_dataset,
    load_inthewild_dataset,
)
from src.preprocessing.dataset_split import stratified_split
from src.preprocessing.feature_extraction import process_and_extract
from src.preprocessing.visualize_spectrograms import visualize_from_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VoxGuard.Preprocessing")


def _process_task(task_args: Tuple) -> Tuple[bool, str, int, str, str, str, str]:
    """
    Worker function for parallel audio preprocessing and Mel-spectrogram extraction.
    Must be top-level for Windows multiprocessing pickling.
    """
    audio_path, dest_npy_path, sample_id, label, dataset, split_name, rel_metadata_path = task_args
    success, message = process_and_extract(
        audio_path=audio_path,
        output_path=dest_npy_path,
        sr=SAMPLE_RATE,
    )
    return (success, rel_metadata_path, label, dataset, split_name, sample_id, message)


# ==============================================================================
# METADATA MANAGEMENT (DEDUPLICATION & RESUMPTION)
# ==============================================================================
def load_existing_metadata(metadata_path: Path) -> Tuple[Set[str], List[Dict[str, str]]]:
    """
    Read existing metadata.csv to index completed files and prevent duplicates.

    Args:
        metadata_path: Path to metadata.csv.

    Returns:
        Tuple of (set_of_existing_relative_filepaths, list_of_existing_rows).
    """
    existing_paths: Set[str] = set()
    existing_rows: List[Dict[str, str]] = []

    if not metadata_path.is_file():
        return existing_paths, existing_rows

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.strip() for c in (reader.fieldnames or [])]
            if fieldnames != METADATA_COLUMNS:
                logger.warning(
                    f"Existing metadata columns {fieldnames} differ from contract {METADATA_COLUMNS}."
                )

            for row in reader:
                norm_fp = str(Path(row["filepath"])).replace("\\", "/")
                existing_paths.add(norm_fp)
                # Also store base filename stem for robust cross-checking
                existing_paths.add(Path(row["filepath"]).stem)
                existing_rows.append(row)

        logger.info(f"Loaded {len(existing_rows)} existing records from {metadata_path.name}")
    except Exception as exc:
        logger.warning(f"Could not read existing metadata file ({exc}). Starting fresh index.")

    return existing_paths, existing_rows


def append_metadata_record(
    metadata_path: Path,
    filepath: str,
    label: int,
    dataset: str,
    split: str,
    write_header_if_missing: bool = True,
) -> None:
    """
    Safely append a single verified record to data/processed/metadata.csv.

    Interface contract:
        Columns: filepath,label,dataset,split
    """
    file_exists = metadata_path.is_file() and metadata_path.stat().st_size > 0
    norm_filepath = str(Path(filepath)).replace("\\", "/")

    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists and write_header_if_missing:
            writer.writerow(METADATA_COLUMNS)
        writer.writerow([norm_filepath, label, dataset, split])


# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================
def run_pipeline(
    asvspoof_dir: Optional[Path] = None,
    inthewild_dir: Optional[Path] = None,
    asvspoof_protocol: Optional[Path] = None,
    inthewild_meta: Optional[Path] = None,
    max_samples_per_dataset: Optional[int] = None,
    random_seed: int = RANDOM_SEED,
    generate_viz: bool = True,
    num_workers: Optional[int] = None,
    processed_dir: Optional[Path] = None,
    metadata_file: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Execute the full preprocessing pipeline across ASVspoof and In-the-Wild datasets.

    Args:
        asvspoof_dir: Custom root directory for ASVspoof raw files.
        inthewild_dir: Custom root directory for In-the-Wild raw files.
        asvspoof_protocol: Custom protocol file for ASVspoof.
        inthewild_meta: Custom meta.csv file for In-the-Wild.
        max_samples_per_dataset: Cap on samples per dataset (None for all).
        random_seed: Seed for reproducible stratified split.
        generate_viz: Whether to automatically create visualization after processing.
        num_workers: Number of parallel worker processes (default: up to 6 cores).
        processed_dir: Custom destination directory for feature files (default: PROCESSED_DIR).
        metadata_file: Custom destination path for metadata CSV (default: METADATA_FILE).

    Returns:
        Summary metrics dictionary.
    """
    logger.info("=" * 70)
    logger.info("VOXGUARD DATA PREPROCESSING PIPELINE INITIATED")
    logger.info("Interface Contract: (128, 128) Mel-Spectrograms | 16 kHz Mono")
    logger.info("=" * 70)

    target_processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    target_metadata_file = Path(metadata_file) if metadata_file else METADATA_FILE

    # 1. Initialize directory layout
    if processed_dir:
        for sp in SPLITS:
            (target_processed_dir / sp).mkdir(parents=True, exist_ok=True)
    else:
        ensure_directories()

    # 2. Index existing completed files (Resume Capability)
    existing_keys, _ = load_existing_metadata(target_metadata_file)

    # 3. Load datasets
    logger.info("Phase 1: Ingesting dataset protocols and audio catalogues...")
    asv_dir = asvspoof_dir if asvspoof_dir is not None else ASVSPOOF_RAW_DIR
    wild_dir = inthewild_dir if inthewild_dir is not None else INTHEWILD_RAW_DIR

    asv_samples = load_asvspoof_dataset(
        raw_dir=asv_dir,
        protocol_file=asvspoof_protocol,
        max_samples=max_samples_per_dataset,
    )
    wild_samples = load_inthewild_dataset(
        raw_dir=wild_dir,
        meta_file=inthewild_meta,
        max_samples=max_samples_per_dataset,
    )

    all_samples = asv_samples + wild_samples

    if not all_samples:
        logger.error(
            "No audio samples found across configured directories.\n"
            f"  - ASVspoof directory: {asv_dir}\n"
            f"  - In-The-Wild directory: {wild_dir}\n"
            "Please ensure raw audio files and metadata/protocol files are present."
        )
        return {
            "status": "No samples found",
            "total_found": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
        }

    logger.info(
        f"Ingestion complete: {len(all_samples)} total samples indexed "
        f"({len(asv_samples)} ASVspoof, {len(wild_samples)} In-The-Wild)."
    )

    # 4. Stratified train/val/test splitting (70 / 15 / 15)
    logger.info("Phase 2: Computing reproducible stratified split (70% Train, 15% Val, 15% Test)...")
    splits = stratified_split(
        all_samples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=random_seed,
    )

    for sp_name in SPLITS:
        sp_samples = splits[sp_name]
        reals = sum(1 for s in sp_samples if s.label == LABEL_REAL)
        fakes = sum(1 for s in sp_samples if s.label == LABEL_FAKE)
        logger.info(
            f"  * Split '{sp_name:<5}': {len(sp_samples):>5} samples "
            f"(Real: {reals:>4}, Fake: {fakes:>4})"
        )

    # 5. Feature extraction & serialization loop
    logger.info("Phase 3: Preprocessing audio and extracting (128, 128) Mel-spectrograms...")

    if num_workers is None:
        cpu_c = os.cpu_count() or 1
        num_workers = max(1, min(6, cpu_c - 1 if cpu_c > 2 else cpu_c))
    logger.info(f"Parallel worker processes: {num_workers}")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    counts = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }
    split_class_counts = {
        sp: {LABEL_REAL: 0, LABEL_FAKE: 0} for sp in SPLITS
    }

    total_tasks = sum(len(s_list) for s_list in splits.values())
    pbar = tqdm(total=total_tasks, desc="VoxGuard Processing", unit="audio") if has_tqdm else None

    # Identify pending tasks vs already completed tasks
    pending_tasks = []
    for split_name in SPLITS:
        split_dir = target_processed_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        sample_list = splits[split_name]

        for sample in sample_list:
            dest_filename = f"{sample.sample_id}.npy"
            dest_npy_path = split_dir / dest_filename
            try:
                rel_metadata_path = dest_npy_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_metadata_path = f"data/processed/{split_name}/{dest_filename}"

            already_done = (
                dest_npy_path.is_file()
                and (
                    rel_metadata_path in existing_keys
                    or sample.sample_id in existing_keys
                )
            )

            if already_done:
                counts["skipped"] += 1
                split_class_counts[split_name][sample.label] += 1
                if pbar:
                    pbar.update(1)
            else:
                pending_tasks.append((
                    sample.audio_path,
                    dest_npy_path,
                    sample.sample_id,
                    sample.label,
                    sample.dataset,
                    split_name,
                    rel_metadata_path,
                ))

    # Process pending tasks
    if pending_tasks:
        file_exists = target_metadata_file.is_file() and target_metadata_file.stat().st_size > 0
        with open(target_metadata_file, "a", newline="", encoding="utf-8") as f_meta:
            meta_writer = csv.writer(f_meta)
            if not file_exists:
                meta_writer.writerow(METADATA_COLUMNS)
                f_meta.flush()

            if num_workers > 1 and len(pending_tasks) > 1:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for res in executor.map(_process_task, pending_tasks, chunksize=16):
                        success, rel_meta_path, label, dataset, split_name, sample_id, message = res
                        if success:
                            counts["processed"] += 1
                            split_class_counts[split_name][label] += 1
                            meta_writer.writerow([rel_meta_path, label, dataset, split_name])
                            existing_keys.add(rel_meta_path)
                            existing_keys.add(sample_id)
                        else:
                            counts["failed"] += 1
                            logger.warning(f"Skipping corrupt or unprocessable file: {sample_id} ({message})")

                        if pbar:
                            pbar.update(1)

                        if counts["processed"] % 50 == 0:
                            f_meta.flush()
            else:
                for task in pending_tasks:
                    success, rel_meta_path, label, dataset, split_name, sample_id, message = _process_task(task)
                    if success:
                        counts["processed"] += 1
                        split_class_counts[split_name][label] += 1
                        meta_writer.writerow([rel_meta_path, label, dataset, split_name])
                        existing_keys.add(rel_meta_path)
                        existing_keys.add(sample_id)
                    else:
                        counts["failed"] += 1
                        logger.warning(f"Skipping corrupt or unprocessable file: {sample_id} ({message})")

                    if pbar:
                        pbar.update(1)

            f_meta.flush()

    if pbar:
        pbar.close()

    # 6. Verification of interface contract
    logger.info("Phase 4: Contract Verification & Diagnostics...")
    verification_passed = True

    if not target_metadata_file.is_file():
        logger.error(f"Metadata file was not created: {target_metadata_file}")
        verification_passed = False
    else:
        with open(target_metadata_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            logger.info(f"Metadata CSV verified: {len(reader)} total records.")

            check_sample_size = min(20, len(reader))
            import numpy as np
            for row in reader[:check_sample_size]:
                fp = PROJECT_ROOT / row["filepath"]
                if not fp.is_file():
                    logger.error(f"Integrity check failed: referenced feature file does not exist: {fp}")
                    verification_passed = False
                    break
                arr = np.load(fp)
                if arr.shape != TARGET_SHAPE:
                    logger.error(f"Contract violation: feature file {fp.name} has shape {arr.shape}, expected {TARGET_SHAPE}")
                    verification_passed = False
                    break

    # 7. Visualization
    viz_file = None
    if generate_viz and target_metadata_file.is_file() and counts["processed"] + counts["skipped"] > 0:
        logger.info("Phase 5: Generating visual Mel-spectrogram diagnostic comparison...")
        try:
            viz_file = visualize_from_metadata(target_metadata_file, num_samples_per_class=2)
        except Exception as exc:
            logger.warning(f"Visualization generation encountered a warning: {exc}")

    # 8. Print Executive Summary
    logger.info("=" * 70)
    logger.info("VOXGUARD PREPROCESSING PIPELINE COMPLETED")
    logger.info(f"  * Total Samples Indexed : {total_tasks}")
    logger.info(f"  * Successfully Processed: {counts['processed']}")
    logger.info(f"  * Skipped (Resumed)     : {counts['skipped']}")
    logger.info(f"  * Failed / Corrupt      : {counts['failed']}")
    logger.info("-" * 70)
    logger.info("Split Distribution Breakdown:")
    for sp_name in SPLITS:
        r = split_class_counts[sp_name][LABEL_REAL]
        f = split_class_counts[sp_name][LABEL_FAKE]
        logger.info(f"  * {sp_name.upper():<5} -> Total: {r + f:>5} | Real (0): {r:>4} | Fake (1): {f:>4}")
    logger.info(f"Metadata CSV: {target_metadata_file}")
    if viz_file:
        logger.info(f"Inspection Figure: {viz_file}")
    logger.info("=" * 70)

    return {
        "status": "Success" if verification_passed else "Verification Failed",
        "total_indexed": total_tasks,
        "processed": counts["processed"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "split_counts": split_class_counts,
        "metadata_file": str(METADATA_FILE),
        "visualization_file": str(viz_file) if viz_file else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VoxGuard - Data Ingestion and Audio Preprocessing Module"
    )
    parser.add_argument(
        "--asvspoof-dir",
        type=str,
        default=None,
        help="Path to raw ASVspoof 2021 LA dataset folder",
    )
    parser.add_argument(
        "--inthewild-dir",
        type=str,
        default=None,
        help="Path to raw In-The-Wild dataset folder",
    )
    parser.add_argument(
        "--asvspoof-protocol",
        type=str,
        default=None,
        help="Path to ASVspoof trial protocol/key text file",
    )
    parser.add_argument(
        "--inthewild-meta",
        type=str,
        default=None,
        help="Path to In-The-Wild meta.csv file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process per dataset (optional limit for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for stratified splitting (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: auto)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generating spectrogram visualization figure",
    )

    args = parser.parse_args()

    run_pipeline(
        asvspoof_dir=Path(args.asvspoof_dir) if args.asvspoof_dir else None,
        inthewild_dir=Path(args.inthewild_dir) if args.inthewild_dir else None,
        asvspoof_protocol=Path(args.asvspoof_protocol) if args.asvspoof_protocol else None,
        inthewild_meta=Path(args.inthewild_meta) if args.inthewild_meta else None,
        max_samples_per_dataset=args.max_samples,
        random_seed=args.seed,
        generate_viz=not args.no_viz,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()

