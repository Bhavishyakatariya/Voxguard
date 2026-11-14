# VoxGuard: Explainable Deepfake Voice Detection & Forensic Attribution
**UPES B.Tech CSE (Data Science) — Major Academic Project**

VoxGuard is an Explainable Artificial Intelligence (XAI) system for detecting deepfake voices and attributing acoustic synthesis anomalies for cybercrime forensics. 

This repository contains the complete, modular codebase across:
1. **Task 1 — Data Collection & Audio Preprocessing** (`src/preprocessing/`)
2. **Task 2 — Model Development & Training** (`src/models/`, `src/training/`, `src/evaluation/`, `src/data/`, `src/utils/`)
3. **Task 3 — Explainability Engine** (Grad-CAM hooking into `self.target_layer`)
4. **Task 4 — Interactive Forensic Dashboard** (Streamlit user interface)

---

## 1. Project Overview & Architecture

VoxGuard addresses the growing threat of AI-synthesized voice cloning used in social engineering and cyber extortion. The system takes raw speech audio, extracts standardized log-Mel spectrograms, classifies the audio using deep neural architectures, and generates forensic visual heatmaps.

```
       +--------------------+
       |  Raw Speech Audio  | (WAV / FLAC / MP3)
       +---------+----------+
                 |
                 v
+-----------------------------------+
| Audio Preprocessing Pipeline     |
| - Mono 16 kHz, silence trimming   |
| - 128 Mel bands, 1024 FFT window  |
| - Exact shape (128, 128) float32  |
+-----------------+-----------------+
                  |
        (128, 128) Mel-Spectrogram
                  |
        +---------+---------+
        |                   |
        v                   v
+---------------+   +-----------------------+
|  Model A:     |   |  Model B:             |
|  2D CNN       |   |  Spectrogram          |
|  Classifier   |   |  Transformer (AST)    |
+-------+-------+   +-----------+-----------+
        |                       |
        +-----------+-----------+
                    |
                    v
       +-------------------------+
       | Probability P(fake)     |  [0.0 = Real, 1.0 = Fake]
       +------------+------------+
                    |
                    v
       +-------------------------+
       | Explainability Hook     |  (Grad-CAM via `self.target_layer`)
       +-------------------------+
```

---

## 2. Model Architectures

Both models strictly satisfy the **Shared Interface Contract #2**:
- **Input Tensor**: `(batch_size, 1, 128, 128)` float32 log-Mel spectrogram.
- **Output Tensor**: Continuous scalar `P(fake) \in [0.0, 1.0]` (0 = Real/Bonafide, 1 = Fake/Deepfake).
- **Target Layer**: Exposes `self.target_layer` for Grad-CAM explainability without code alteration.

### Model A: 2D Convolutional Neural Network (`CNNClassifier`)
- **Location**: `src/models/cnn_model.py` (aliased as `VoxGuardCNN` in `src/models/voxguard_cnn.py`)
- **Block 1**: Conv2D(1 $\rightarrow$ 32, $3\times 3$) + BatchNorm2D + ReLU + MaxPool2D(2, 2) $\rightarrow$ `(32, 64, 64)`
- **Block 2**: Conv2D(32 $\rightarrow$ 64, $3\times 3$) + BatchNorm2D + ReLU + MaxPool2D(2, 2) $\rightarrow$ `(64, 32, 32)`
- **Block 3**: Conv2D(64 $\rightarrow$ 128, $3\times 3$) + BatchNorm2D + ReLU + MaxPool2D(2, 2) $\rightarrow$ `(128, 16, 16)`
- **Block 4**: Conv2D(128 $\rightarrow$ 256, $3\times 3$) + BatchNorm2D + ReLU + MaxPool2D(2, 2) $\rightarrow$ `(256, 8, 8)`
- **Target Hook**: `self.target_layer = self.conv4` (Block 4 convolutional feature map)
- **Classification Head**: AdaptiveAvgPool2D(1, 1) $\rightarrow$ Flatten $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(256, 64) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(64, 1) $\rightarrow$ Sigmoid

### Model B: Audio Spectrogram Transformer (`TransformerClassifier`)
- **Location**: `src/models/transformer_model.py` (aliased as `VoxGuardTransformer` in `src/models/voxguard_transformer.py`)
- **Patch Projection**: Non-overlapping $16\times 16$ convolutional patches $\rightarrow$ $8\times 8 = 64$ patch tokens with embedding dimension $d=128$.
- **Class Token & Position**: Learnable `[CLS]` token prepended (sequence length $N = 65$) + 1D learnable positional embeddings + Dropout(0.1).
- **Transformer Encoder**: 4 stacked Encoder layers (4 attention heads, feedforward dimension 512, GELU activation, pre-LayerNorm).
- **Target Hook**: `self.target_layer = self.transformer_encoder.layers[-1]`
- **Classification Head**: LayerNorm(128) $\rightarrow$ Linear(128, 1) $\rightarrow$ Sigmoid

---

## 3. Project Directory Structure

```
voxguard/
|-- configs/
|   `-- config.yaml               # Centralized configuration (data source toggle, hyperparams)
|-- data/
|   |-- processed/                # Real preprocessed corpora (213,345 samples)
|   |   |-- train/
|   |   |-- val/
|   |   |-- test/
|   |   `-- metadata.csv          # Master metadata (filepath,label,dataset,split)
|   `-- dummy/                    # Isolated synthetic dataset for testing
|       |-- train/
|       |-- val/
|       |-- test/
|       `-- metadata.csv
|-- models/
|   |-- voxguard_model.pt         # Canonical 34-key checkpoint (compatible with dashboard)
|   `-- voxguard_cnn_best.pt
|-- checkpoints/                  # Architecture-specific checkpoints
|   |-- cnn/
|   `-- transformer/
|-- results/
|   |-- metrics/                  # Structured JSON and CSV evaluation reports
|   `-- training_history/         # Per-epoch loss/accuracy history logs
|-- src/
|   |-- models/
|   |   |-- cnn_model.py          # Model A (CNNClassifier)
|   |   |-- transformer_model.py  # Model B (TransformerClassifier)
|   |   |-- model_factory.py      # Factory loader: get_model()
|   |   |-- voxguard_cnn.py       # Canonical implementation
|   |   |-- voxguard_transformer.py
|   |   `-- voxguard_net.py       # Dashboard & Grad-CAM backward-compat alias
|   |-- training/
|   |   |-- train.py              # CLI training entrypoint
|   |   |-- trainer.py            # VoxGuardTrainer orchestrator
|   |   `-- early_stopping.py     # Early stopping monitor
|   |-- evaluation/
|   |   |-- evaluate.py           # Comprehensive evaluation CLI
|   |   `-- metrics.py            # Accuracy, Precision, Recall, F1, EER, AUC-ROC
|   |-- data/
|   |   |-- dataset.py            # Memory-mapped PyTorch Dataset
|   |   `-- dummy_data.py         # Synthetic dummy dataset generator
|   |-- utils/
|   |   |-- config.py             # YAML loader with fallback parser
|   |   |-- checkpoint.py         # Checkpoint saving & loading
|   |   `-- logger.py             # Structured logging utility
|   `-- preprocessing/            # Task 1 Data Engineering module
|-- test_model_pipeline.py        # Automated test suite (11 unit tests)
|-- requirements.txt
`-- README.md
```

---

## 4. Dependencies & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Windows, Linux, or macOS

### Installation
```bash
# Clone or navigate to repository
cd voxguard

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. How to Switch from Dummy Data to Real Data

The entire pipeline switches between dummy test data and the actual preprocessed dataset by modifying **ONE value** in `configs/config.yaml`:

```yaml
# configs/config.yaml
data_source: "real"    # Change to "dummy" for synthetic dry-runs, or "real" for full corpora
```

You can also override this on the command line for any script:
```bash
python -m src.training.train --data-source dummy
python -m src.training.train --data-source real
```

---

## 6. Execution Commands

### A. Generate Synthetic Dummy Data
```bash
python -m src.data.dummy_data --samples-per-split 24
```
Generates 72 synthetic `.npy` files with shape `(128, 128)` and `data/dummy/metadata.csv`.

### B. Run Automated Verification Test Suite
```bash
python test_model_pipeline.py
```
Executes 11 automated unit tests verifying interface contracts, models, training loops, checkpoint loading, and metric calculations.

### C. Train Model A: 2D CNN
```bash
# Train on full real dataset (CPU or GPU auto-detected):
python -m src.training.train --arch cnn --epochs 5 --batch-size 64

# Resume training from existing checkpoint:
python -m src.training.train --arch cnn --resume models/voxguard_model.pt

# Fast dry-run on dummy data:
python -m src.training.train --arch cnn --data-source dummy --epochs 1 --batch-size 8
```

### D. Train Model B: Spectrogram Transformer
```bash
# Train Transformer on real dataset:
python -m src.training.train --arch transformer --epochs 5 --batch-size 64

# Dry-run Transformer on dummy data:
python -m src.training.train --arch transformer --data-source dummy --epochs 1 --batch-size 8
```

### E. Evaluate Models & Generate Forensic Reports
```bash
# Evaluate CNN on full held-out test partition:
python -m src.evaluation.evaluate --arch cnn --checkpoint models/voxguard_model.pt

# Evaluate on dummy data:
python -m src.evaluation.evaluate --arch cnn --data-source dummy --checkpoint models/voxguard_model.pt
```

---

## 7. Forensic Evaluation Metrics & Equal Error Rate (EER)

VoxGuard evaluates classifiers on standard forensic biometric metrics:

1. **Accuracy**: Overall fraction of correct classifications:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision**: Positive predictive value (how many detected fakes were truly fake):
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity)**: Deepfake detection sensitivity (fraction of all fakes successfully caught):
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score**: Harmonic mean of Precision and Recall:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve.

### How EER is Calculated
In voice forensics, the decision threshold $\theta \in [0.0, 1.0]$ controls the balance between two error rates:
- **False Acceptance Rate (FAR)**: Proportion of Real/Bonafide voices mistakenly flagged as Fake:
  $$\text{FAR}(\theta) = \frac{FP}{N_{\text{real}}}$$
- **False Rejection Rate (FRR)**: Proportion of Fake/Spoofed voices mistakenly accepted as Real:
  $$\text{FRR}(\theta) = \frac{FN}{N_{\text{fake}}}$$

As $\theta$ increases from 0 to 1, FAR decreases while FRR increases. The **Equal Error Rate (EER)** is the operating point where FAR and FRR intersect:
$$\text{EER} = \text{FAR}(\theta^*) = \text{FRR}(\theta^*), \quad \text{where } \theta^* = \arg\min_{\theta} |\text{FAR}(\theta) - \text{FRR}(\theta)|$$

A lower EER percentage indicates superior forensic reliability.

---

## 8. Cross-Dataset Generalization Gap

Deepfake detectors frequently suffer from domain overfitting: they achieve near-perfect metrics on in-domain benchmarks (e.g., ASVspoof) but degrade significantly when deployed on real-world compressed audio (In-The-Wild).

`src/evaluation/evaluate.py` automatically decomposes the test partition into:
- **ASVspoof logical access partition** (`dataset == "asvspoof"`)
- **In-The-Wild unconstrained partition** (`dataset == "inthewild"`)

The script reports:
- $\Delta \text{EER} = |\text{EER}_{\text{inthewild}} - \text{EER}_{\text{asvspoof}}|$
- $\Delta \text{AUC} = |\text{AUC}_{\text{asvspoof}} - \text{AUC}_{\text{inthewild}}|$

This metric diagnoses out-of-domain robustness for forensic law enforcement deployment.

---

## 9. Checkpoint Management

Trained weights are automatically managed and serialized:
- [`models/voxguard_model.pt`](models/voxguard_model.pt): Canonical checkpoint with exact 34 weight keys, guaranteeing zero breakage with downstream Grad-CAM explainability (`src/xai/`) and the Streamlit dashboard (`src/dashboard/app.py`).
- `models/voxguard_cnn_best.pt` & `models/voxguard_transformer_best.pt`: Best validation checkpoints.
- `results/metrics/`: Structured evaluation outputs (`evaluation_cnn.json` and `evaluation_cnn_summary.csv`).
- `results/training_history/`: Full training epoch history logs (`cnn_training_history.json`).
