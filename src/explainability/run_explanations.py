"""
VoxGuard - Batch Explanation & Presentation Review Runner
=========================================================
Fulfills TASK 3 Part 6: Executes batch explanations across genuine real and fake
audio samples from the dataset, generating presentation-ready heatmaps, a summary review
log, and a structured `explanations.json` record.
"""

from pathlib import Path
import json
import argparse
from typing import List, Dict, Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.explainability.model_loader import load_trained_model
from src.explainability.explain import explain
from src.utils.logger import get_logger

logger = get_logger("VoxGuard.XAI.BatchRunner")


def run_batch_explanations(
    metadata_path: Path = PROJECT_ROOT / "data" / "processed" / "metadata.csv",
    output_dir: Path = PROJECT_ROOT / "outputs" / "explainability",
    num_samples_per_category: int = 3,
) -> List[Dict[str, Any]]:
    """
    Select representative audio files (Real, Fake, In-The-Wild, ASVspoof) and run explanations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir = output_dir / "heatmaps"
    review_dir = output_dir / "review"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing trained VoxGuard classifier for batch explanation...")
    model, device = load_trained_model()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    test_df = df[df["split"] == "test"]
    if len(test_df) == 0:
        test_df = df

    selected_samples = []

    # 1. Authentic Real voices (In-The-Wild & ASVspoof)
    real_itw = test_df[(test_df["label"] == 0) & (test_df["dataset"] == "inthewild")].head(num_samples_per_category)
    real_asv = test_df[(test_df["label"] == 0) & (test_df["dataset"] == "asvspoof")].head(num_samples_per_category)

    # 2. Cloned Fake voices (In-The-Wild & ASVspoof)
    fake_itw = test_df[(test_df["label"] == 1) & (test_df["dataset"] == "inthewild")].head(num_samples_per_category)
    fake_asv = test_df[(test_df["label"] == 1) & (test_df["dataset"] == "asvspoof")].head(num_samples_per_category)

    batch_df = pd.concat([real_itw, real_asv, fake_itw, fake_asv]).drop_duplicates(subset=["filepath"])
    logger.info(f"Selected {len(batch_df)} genuine test samples for XAI evaluation.")

    results: List[Dict[str, Any]] = []

    for idx, row in batch_df.iterrows():
        rel_path = row["filepath"]
        ground_truth = int(row["label"])
        dataset_name = row.get("dataset", "unknown")

        npy_path = PROJECT_ROOT / rel_path

        # Find corresponding raw audio if present, or use npy
        audio_candidate = None
        base_stem = Path(rel_path).stem
        if dataset_name == "inthewild":
            raw_wav = PROJECT_ROOT / "data" / "raw" / "inthewild" / f"{base_stem}.wav"
            if raw_wav.is_file():
                audio_candidate = raw_wav
        elif dataset_name == "asvspoof":
            raw_flac = PROJECT_ROOT / "data" / "raw" / "asvspoof" / "flac" / f"{base_stem}.flac"
            if raw_flac.is_file():
                audio_candidate = raw_flac

        target_file = audio_candidate if audio_candidate is not None else npy_path
        if not target_file.is_file():
            logger.warning(f"File not found: {target_file}. Skipping.")
            continue

        gt_str = "Real" if ground_truth == 0 else "Fake"
        prefix = f"{dataset_name}_gt_{gt_str.lower()}"

        try:
            explanation_res = explain(
                model=model,
                audio_path=target_file,
                output_dir=heatmaps_dir,
                filename_prefix=prefix,
            )
            # Enrich with ground truth validation
            is_correct = (explanation_res["predicted_class"] == ground_truth)
            explanation_res["ground_truth_label"] = gt_str
            explanation_res["ground_truth_class"] = ground_truth
            explanation_res["dataset"] = dataset_name
            explanation_res["classification_correct"] = is_correct

            results.append(explanation_res)
            logger.info(
                f"[{len(results)}/{len(batch_df)}] Processed {target_file.name} | "
                f"GT: {gt_str} -> Pred: {explanation_res['predicted_label']} "
                f"({explanation_res['confidence']*100:.1f}%) | Correct: {is_correct}"
            )
        except Exception as exc:
            logger.error(f"Error explaining {target_file}: {exc}")

    # Save structured JSON record
    json_path = output_dir / "explanations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} XAI records to: {json_path}")

    # Generate Markdown Presentation Summary in review/
    md_summary_path = review_dir / "batch_xai_summary.md"
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write("# VoxGuard Forensic Explainability — Batch Review Summary\n\n")
        f.write(f"Generated from actual trained model weights on `{len(results)}` genuine test samples.\n\n")
        f.write("| Sample | Dataset | Ground Truth | Prediction | Confidence | Correct | Frequency Band | Caption |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in results:
            file_name = Path(r["audio_path"]).name
            f.write(
                f"| `{file_name}` | {r['dataset']} | **{r['ground_truth_label']}** | "
                f"**{r['predicted_label']}** | {r['confidence']*100:.1f}% | "
                f"{'✅' if r['classification_correct'] else '❌'} | "
                f"{r['frequency_band']} | {r['caption']} |\n"
            )

    logger.info(f"Saved Markdown presentation review to: {md_summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Batch XAI Execution Script")
    parser.add_argument("--samples-per-category", type=int, default=2, help="Number of samples to explain per category")
    args = parser.parse_args()

    run_batch_explanations(num_samples_per_category=args.samples_per_category)


if __name__ == "__main__":
    main()

