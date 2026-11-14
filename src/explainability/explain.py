"""
VoxGuard - Central Explainability Engine
========================================
Fulfills TASK 3 Part 2 & Part 5: Main explain(model, audio_path) implementation.
Preprocesses audio conforming exactly to training-time specifications, performs
inference with the actual trained VoxGuard model, computes Grad-CAM attribution,
renders high-resolution overlay heatmaps, generates natural-language captions,
and returns the standard forensic dictionary.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.preprocessing.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    TARGET_SHAPE,
    TARGET_TIME_STEPS,
)
from src.preprocessing.audio_preprocessing import (
    load_audio,
    trim_silence,
    normalize_audio,
)
from src.preprocessing.feature_extraction import (
    extract_mel_spectrogram,
    pad_or_crop_spectrogram,
)
from src.explainability.gradcam import GradCAM
from src.explainability.caption_generator import generate_explanation_caption
from src.explainability.visualization import save_visualization
from src.explainability.model_loader import load_trained_model
from src.utils.logger import get_logger

logger = get_logger("VoxGuard.XAI.Explain")


def preprocess_audio_for_model(audio_path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Standardize raw audio file or load precomputed .npy into the exact
    (1, 1, 128, 128) tensor expected by VoxGuardCNN.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input audio/feature file not found: {path.resolve()}")

    if path.suffix.lower() == ".npy":
        # Direct load of precomputed Mel-spectrogram
        mel_spectrogram = np.load(path).astype(np.float32)
        if mel_spectrogram.shape != TARGET_SHAPE:
            mel_spectrogram = pad_or_crop_spectrogram(mel_spectrogram, TARGET_TIME_STEPS)
    else:
        # Raw audio pipeline: Load -> Mono -> 16kHz -> Silence Trim -> Peak Normalization
        waveform, sr = load_audio(path, target_sr=SAMPLE_RATE, mono=True)
        waveform = trim_silence(waveform, top_db=20)
        waveform = normalize_audio(waveform)

        # Extract Mel-spectrogram -> Pad/Crop to (128, 128)
        raw_mel = extract_mel_spectrogram(
            waveform,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )
        mel_spectrogram = pad_or_crop_spectrogram(raw_mel, TARGET_TIME_STEPS)

    # Convert to PyTorch float32 tensor of shape (1, 1, 128, 128)
    input_tensor = torch.from_numpy(mel_spectrogram).float().unsqueeze(0).unsqueeze(0)
    return input_tensor, mel_spectrogram


def explain(
    model: Optional[nn.Module] = None,
    audio_path: Union[str, Path] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename_prefix: str = "",
    target_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Primary Explainability Function conforming to the VoxGuard Shared Interface Contract.

    Args:
        model: Trained VoxGuard PyTorch model. If None, loads canonical trained checkpoint.
        audio_path: Path to raw audio (.wav, .flac, .mp3) or precomputed (.npy) feature.
        output_dir: Destination directory for the generated PNG visualization.
                    Defaults to 'outputs/explainability/heatmaps'.
        filename_prefix: Optional prefix for saved image file.
        target_class: Optional class to explain (0 = Real, 1 = Fake).
                      Defaults to predicted class.

    Returns:
        Structured forensic result dictionary matching contract specifications.
    """
    if audio_path is None:
        raise ValueError("audio_path must be provided to explain().")

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path.resolve()}")

    # 1. Resolve model
    if model is None:
        model, _ = load_trained_model()
    elif isinstance(model, (str, Path)):
        model, _ = load_trained_model(checkpoint_path=model)
    model.eval()

    # 2. Preprocess audio identically to training time
    input_tensor, mel_spectrogram = preprocess_audio_for_model(path)

    # 3. Generate Grad-CAM attribution heatmap
    gradcam_engine = GradCAM(model)
    heatmap, pred_class, confidence, prob_fake = gradcam_engine.generate_heatmap(
        input_tensor=input_tensor,
        target_class=target_class,
    )

    # 4. Generate natural-language caption and structured anomaly metadata
    caption_info = generate_explanation_caption(
        heatmap=heatmap,
        predicted_class=pred_class,
        confidence=confidence,
    )

    # 5. Save professional visual explanation
    if output_dir is None:
        out_folder = PROJECT_ROOT / "outputs" / "explainability" / "heatmaps"
    else:
        out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)

    base_name = path.stem
    if filename_prefix:
        save_name = f"{filename_prefix}_{base_name}_heatmap.png"
    else:
        save_name = f"{base_name}_heatmap.png"

    heatmap_path = out_folder / save_name
    saved_heatmap_path = save_visualization(
        mel_spectrogram=mel_spectrogram,
        heatmap=heatmap,
        predicted_class=pred_class,
        confidence=confidence,
        output_path=heatmap_path,
        title_suffix=f"File: {path.name}",
    )

    pred_label_str = "Fake" if pred_class == 1 else "Real"
    verdict_str = "fake" if pred_class == 1 else "real"

    # Return structured contract dictionary (supporting both contract formats)
    return {
        "verdict": verdict_str,
        "confidence": float(round(confidence, 4)),
        "heatmap_path": str(saved_heatmap_path),
        "caption": caption_info["caption"],
        "audio_path": str(path.resolve()),
        "predicted_label": pred_label_str,
        "predicted_class": int(pred_class),
        "fake_probability": float(round(prob_fake, 4)),
        "time_interval_seconds": caption_info["time_interval"],
        "frequency_range_hz": caption_info["frequency_range_hz"],
        "frequency_band": caption_info["frequency_band"],
        "attribution_strength": caption_info["attribution_strength"],
        "model_architecture": model.__class__.__name__,
    }
