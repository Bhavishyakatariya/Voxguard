# VoxGuard: Model Development & Training Module (`src/models/`)

## 1. Module Overview
The **VoxGuard Model Development & Training Module** implements the core acoustic classification engine for explainable deepfake voice detection. It provides two complementary neural backbones engineered against the $(128, 128)$ Mel-spectrogram feature contract:

1. **`VoxGuardCNN`**: A 4-block deep 2D Convolutional Neural Network with Batch Normalization, Dropout, and Adaptive Average Pooling. Critically, it explicitly exposes `self.target_layer` (pointing to Block 4's Conv2D layer) to enable Gradient-Weighted Class Activation Mapping (Grad-CAM) by Amrita's explainability pipeline.
2. **`VoxGuardTransformer`**: An Audio Spectrogram Transformer (AST) that partitions $(128, 128)$ spectrograms into 64 non-overlapping $(16\times16)$ patches, projects them into a 128-dimensional embedding space, prepends a learnable `[CLS]` token, adds 1D positional embeddings, and passes them through 4 Multi-Head Self-Attention encoder layers with GELU activations.
3. **`VoxGuardNet`**: A backwards-compatible drop-in alias in `voxguard_net.py` enabling seamless plug-and-play integration with Amrita's explainability module (`explain.py`) and Hiral's Streamlit dashboard (`src/dashboard/app.py`).

---

## 2. Shared Interface Contract Conformance

| Interface Component | Contract Specification | Implementation Detail |
|---|---|---|
| **Input Shape** | `(B, 1, 128, 128)` | Float32 normalized Mel-spectrogram tensor |
| **Output Shape** | `(B, 1)` | Float32 posterior probability $P(\text{fake}) \in [0.0, 1.0]$ |
| **Classification Threshold** | $\tau = 0.50$ | FAKE if $P(\text{fake}) \ge 0.50$, REAL if $P(\text{fake}) < 0.50$ |
| **Grad-CAM Target Layer** | `self.target_layer` | Conv2D layer of Block 4 (`self.conv4`) |
| **Checkpoint Storage** | `models/voxguard_model.pt` | PyTorch binary `state_dict` |
| **Dataset Source** | `data/processed/metadata.csv` | Reads preprocessed `.npy` tensors across `train`, `val`, `test` splits |

---

## 3. Architecture Specifications

### A. VoxGuardCNN (4-Block 2D CNN)
```
Input: (B, 1, 128, 128)
  │
  ├── Block 1: Conv2D(1 -> 32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> (B, 32, 64, 64)
  ├── Block 2: Conv2D(32 -> 64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> (B, 64, 32, 32)
  ├── Block 3: Conv2D(64 -> 128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> (B, 128, 16, 16)
  └── Block 4: Conv2D(128 -> 256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> (B, 256, 8, 8)
        [self.target_layer -> hook for Grad-CAM]
  │
  ├── Global Average Pooling: (B, 256, 8, 8) -> (B, 256, 1, 1) -> Flatten -> (B, 256)
  ├── Dropout(0.3)
  ├── Linear(256 -> 64) -> ReLU
  ├── Dropout(0.2)
  └── Linear(64 -> 1) -> Sigmoid -> (B, 1) P(fake)
```

### B. VoxGuardTransformer (Audio Spectrogram Transformer)
```
Input: (B, 1, 128, 128)
  │
  ├── Patch Projection: Conv2D(1 -> 128, kernel=16, stride=16) -> (B, 128, 8, 8) -> (B, 64, 128)
  ├── Concat [CLS] Token: (B, 1, 128) + (B, 64, 128) -> (B, 65, 128)
  ├── Add Positional Embeddings: (1, 65, 128)
  ├── Transformer Encoder (4 Layers, 4 Heads, MLP Dim 512, GELU)
  ├── Extract [CLS] Token at index 0 -> (B, 128)
  ├── LayerNorm(128)
  └── Linear(128 -> 1) -> Sigmoid -> (B, 1) P(fake)
```

---

## 4. How to Train the Classifiers

### Standard Training (VoxGuardCNN):
```bash
python src/models/train.py --arch cnn --epochs 5 --batch-size 32 --max-train-samples 4000 --max-val-samples 1000
```

### Training Audio Spectrogram Transformer (AST):
```bash
python src/models/train.py --arch transformer --epochs 5 --batch-size 32 --max-train-samples 4000 --max-val-samples 1000
```

### Command-Line Arguments:
- `--arch`: Model backbone (`cnn` or `transformer`, default: `cnn`).
- `--epochs`: Total training epochs (default: `5`).
- `--batch-size`: Batch size (default: `32`).
- `--lr`: Initial learning rate for Adam (default: `1e-3`).
- `--max-train-samples`: Cap training samples with balanced real/fake distribution (default: `4000`).
- `--max-val-samples`: Cap validation samples (default: `1000`).

---

## 5. How to Evaluate Models

Run the evaluation suite against the held-out test split:
```bash
python src/models/evaluate.py --checkpoint models/voxguard_model.pt --arch cnn --max-test-samples 2000
```

Outputs generated:
1. `reports/evaluation_report.json`: Comprehensive metrics (Accuracy, F1, AUC, EER, and cross-dataset generalization).
2. `reports/roc_curve_cnn.png`: High-resolution ROC curve with EER operating point.

---

## 6. Python Usage Example
```python
import torch
from src.models import VoxGuardCNN, load_cnn_model

# 1. Instantiate or load trained checkpoint
model = load_cnn_model("models/voxguard_model.pt", device="cpu")

# 2. Forward pass with Mel-spectrogram tensor (1, 1, 128, 128)
spectrogram = torch.randn(1, 1, 128, 128)
prob_fake = model(spectrogram).item()

print(f"Deepfake Probability P(fake): {prob_fake:.4f}")
print("Verdict:", "FAKE" if prob_fake >= 0.50 else "REAL")
```

