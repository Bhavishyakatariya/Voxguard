# VoxGuard: Explainable AI (XAI) Engine & Forensic Attribution (Task 3: Amrita)

## 1. Module Overview

The **VoxGuard Explainable AI (XAI) Module** provides mathematically grounded, transparent, and courtroom-admissible forensic explanations for deepfake audio detections. Rather than treating neural classifiers as uninterpretable black boxes, this module connects model predictions directly to physical acoustic phenomena (vocal tract anomalies, formant distortion, and vocoder synthesis artifacts).

### Key Highlights
- **100% Genuine Explanations:** Powered by the actual trained `VoxGuardCNN` classifier weights (`models/voxguard_cnn_best.pt` or `models/voxguard_model.pt`).
- **No Dummy / Synthetic Predictions:** Directly computes gradients of class logits with respect to convolutional feature activations.
- **Symmetrical Class Attribution:** Explains both **FAKE** predictions (detecting synthetic vocoder anomalies) and **REAL** predictions (detecting organic harmonic continuity).
- **Physical Coordinates:** Bounded by exact time (seconds, $0.0 - 4.1\text{ s}$) and frequency ($0.0 - 8.0\text{ kHz}$) axes.
- **Natural Language Captioning:** Synthesizes human-readable acoustic summaries for presentation slides, viva defense, and evidentiary documentation.

---

## 2. Directory Structure

```text
src/
└── explainability/
    ├── __init__.py               # Public API exports
    ├── model_loader.py           # Safe checkpoint restoration into eval mode
    ├── gradcam.py                # PyTorch forward/backward hook Grad-CAM engine
    ├── caption_generator.py      # Coordinate-to-text semantic captioning engine
    ├── visualization.py          # Dual-panel 300-DPI Mel-spectrogram & heatmap plotter
    ├── explain.py                # Main explain(model, audio_path) contract interface
    ├── shap_explainer.py         # Optional game-theoretic SHAP wrapper
    └── run_explanations.py       # Batch CLI runner for test dataset evaluation

outputs/
└── explainability/
    ├── heatmaps/                 # High-resolution PNG attribution visualizations
    ├── review/                   # batch_xai_summary.md presentation table
    └── explanations.json         # Structured forensic audit trail records
```

---

## 3. Actual Trained Model Integration

### Checkpoint Restoration
The module loads trained weights from:
- Primary Checkpoint: `models/voxguard_cnn_best.pt` (Val Loss: `0.1287`, AUC: `0.9654`)
- Fallback Checkpoint: `models/voxguard_model.pt`

```python
from src.explainability.model_loader import load_trained_model

# Loads model in eval mode on CPU or CUDA automatically
model, device = load_trained_model("models/voxguard_cnn_best.pt")
```

### Target Layer for Grad-CAM
The 4th convolutional block (`model.conv4` / `model.target_layer`) serves as the gradient interception layer:
- **Architecture Block:** `Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`
- **Spatial Resolution:** Activations of shape `(Batch, 256, 8, 8)` capturing high-level semantic acoustic structures before global average pooling.

---

## 4. Grad-CAM Methodology

For input Mel-spectrogram tensor $X \in \mathbb{R}^{1 \times 1 \times 128 \times 128}$:
1. **Forward Pass:** Compute class logit $y^c$. For Fake explanation ($c=1$), $y^c = z$; for Real explanation ($c=0$), $y^c = -z$.
2. **Gradient Interception:** Backward pass calculates $\frac{\partial y^c}{\partial A^k}$, where $A^k$ represents the $k$-th feature activation map of `conv4`.
3. **Neuron Importance Weights:**
   $$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. **Weighted Activation Map:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. **Bilinear Upsampling & Normalization:** Upsampled to $(128, 128)$ and min-max scaled into $[0.0, 1.0]$.

---

## 5. Main Contract Interface: `explain()`

Function:
```python
from src.explainability.explain import explain

result = explain(
    model=model,
    audio_path="data/raw/inthewild/13213.wav",
    output_dir="outputs/explainability/heatmaps"
)
```

### Returned Forensic Dictionary
Conforms strictly to Section 3 of the VoxGuard Shared Interface Contract:
```json
{
  "audio_path": "C:\\...\\data\\raw\\inthewild\\13213.wav",
  "predicted_label": "Real",
  "predicted_class": 0,
  "confidence": 1.0,
  "fake_probability": 0.0,
  "heatmap_path": "C:\\...\\outputs\\explainability\\heatmaps\\13213_heatmap.png",
  "caption": "The model classified this recording as REAL (100.0% confidence). The decision was primarily supported by natural acoustic consistency across the high-frequency region (0.4–8.0 kHz) during 0.0–3.9 seconds, exhibiting organic high-frequency spectral envelope and phase continuity.",
  "time_interval_seconds": [0.0, 3.9],
  "frequency_range_hz": [441.0, 8000.0],
  "frequency_band": "high",
  "attribution_strength": 0.812,
  "model_architecture": "VoxGuardCNN"
}
```

---

## 6. Execution Commands

### 1. Run Automated Test Verification
```bash
python test_xai_pipeline.py
```
*Validates model loading, hook cleanup, heatmap normalization, caption generation, and contract key compliance.*

### 2. Generate Batch Explanations on Test Dataset
```bash
python -m src.explainability.run_explanations --samples-per-category 2
```
*Processes authentic Real and Fake samples across In-The-Wild and ASVspoof, generating visualizations in `outputs/explainability/heatmaps/` and audit records in `outputs/explainability/explanations.json`.*

### 3. Single-File CLI Explanation
```python
from src.explainability import explain, load_trained_model

model, _ = load_trained_model()
res = explain(model, "data/raw/inthewild/9258.wav")
print(res["caption"])
```

---

## 7. Forensic Interpretation Guidelines (Viva & Presentation)

1. **What Grad-CAM Shows:**
   - Highlights the spectro-temporal coordinates that gave the highest mathematical gradient contribution to the classifier's verdict.
2. **What Grad-CAM Does NOT Mean:**
   - It is not an arbitrary measurement of "how fake" audio is.
   - It does not physically identify a specific speaker or vocoder algorithm by name; rather, it highlights acoustic anomalies like unnatural formant jitter, silence floor manipulation, or high-frequency phase cancellation.

