# VoxGuard: Data Collection & Preprocessing Module

## 1. Module Overview
The **VoxGuard Preprocessing Module** (`src/preprocessing/`) forms the foundational data engineering tier of the VoxGuard Explainable Deepfake Voice Detection system. It is engineered to ingest raw, heterogeneous audio corpora from:
1. **ASVspoof 2021 Logical Access (LA)**: Standardized challenge evaluations containing bona-fide speech and synthetic voice attacks generated across state-of-the-art neural vocoders and voice conversion systems.
2. **In-The-Wild Dataset**: Real-world unconstrained deepfake speech harvested from internet media platforms.

The module enforces strict acoustic standardization (16 kHz mono resampling, silence boundary trimming, peak normalization) and extracts log-magnitude **Mel-spectrograms** formatted to an exact shape of `(128, 128)` saved as `.npy` matrices. It partitions the dataset into stratified subsets (**70% Training, 15% Validation, 15% Testing**) with zero data leakage, and maintains an auditable `metadata.csv` registry.

---

## 2. Shared Interface Contract Specifications

All downstream model training scripts (e.g. `VoxGuardCNN`, `VoxGuardTransformer`) rely on this strict contract:

| Property | Contract Specification | Description |
|---|---|---|
| **Feature File Format** | `.npy` (NumPy binary array) | Serialized 2D float32 array |
| **Feature Tensor Shape** | Exactly `(128, 128)` | `(n_mels=128, time_steps=128)` |
| **Output Directory** | `data/processed/{split}/{filename}.npy` | `{split}` $\in$ `['train', 'val', 'test']` |
| **Metadata File** | `data/processed/metadata.csv` | Single master index of all feature files |
| **Metadata Columns** | `filepath,label,dataset,split` | Strict 4-column CSV format |
| **Label Mapping** | `0` = Real, `1` = Fake | Binary classification contract |
| **Sampling Rate** | 16,000 Hz | Resampled via `librosa` |
| **Channels** | Mono (1-channel) | Downmixed from multi-channel audio |
| **Split Ratios** | Train: 70%, Val: 15%, Test: 15% | Stratified jointly by `(dataset, label)` |

---

## 3. Required Dependencies

The preprocessing module requires Python 3.10 or 3.11 with the following libraries:

- `librosa >= 0.10.0`: Audio loading, resampling, STFT, and Mel filterbank extraction.
- `soundfile >= 0.12.0`: Audio file decoding backend (`.flac`, `.wav`).
- `scipy >= 1.7.0`: Digital signal processing and interpolation.
- `numpy >= 1.20.0`: Feature array manipulation and `.npy` storage.
- `pandas >= 1.3.0`: Dataset cataloging and tabular processing.
- `scikit-learn >= 1.0.0`: Reproducible stratified sampling.
- `matplotlib >= 3.5.0`: Diagnostic spectrogram rendering.
- `tqdm >= 4.65.0`: Visual terminal progress telemetry.

---

## 4. Installation & Environment Setup

### Step 1: Create a Virtual Environment
From the project root directory:

```bash
# Using Python 3.10 or 3.11
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# Activate on Linux / macOS
source .venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Dataset Directory Configuration

The pipeline defaults to reading from `data/raw/` in the workspace:

```text
data/
└── raw/
    ├── asvspoof/
    │   ├── flac/                                # Directory with .flac audio files
    │   └── ASVspoof2021.LA.cm.eval.trl.txt      # Trial key / protocol text file
    └── inthewild/
        ├── audio/                               # Directory with .wav or .mp3 audio files
        └── meta.csv                             # In-The-Wild metadata CSV
```

### Flexible Path Configuration:
You can configure dataset locations without moving files:

1. **Option A: Command-Line Flags (Recommended)**
   ```bash
   python -m src.preprocessing.main --asvspoof-dir "D:/Datasets/asvspoof2021" --inthewild-dir "D:/Datasets/in_the_wild"
   ```

2. **Option B: Environment Variables**
   ```bash
   # Windows PowerShell
   $env:VOXGUARD_ASVSPOOF_DIR="D:\Datasets\asvspoof2021"
   $env:VOXGUARD_INTHEWILD_DIR="D:\Datasets\in_the_wild"

   # Linux / macOS
   export VOXGUARD_ASVSPOOF_DIR="/path/to/asvspoof"
   export VOXGUARD_INTHEWILD_DIR="/path/to/inthewild"
   ```

3. **Option C: Edit `src/preprocessing/config.py`**
   Update `ASVSPOOF_RAW_DIR` and `INTHEWILD_RAW_DIR` default variables.

---

## 6. How to Run the Preprocessing Pipeline

### Standard End-to-End Execution
Runs full loading, stratified splitting, preprocessing, extraction, metadata creation, and visual inspection:

```bash
python -m src.preprocessing.main
```

### Processing with Explicit Custom Paths
```bash
python -m src.preprocessing.main \
    --asvspoof-dir "data/raw/asvspoof" \
    --inthewild-dir "data/raw/inthewild" \
    --asvspoof-protocol "data/raw/asvspoof/ASVspoof2021.LA.cm.eval.trl.txt" \
    --inthewild-meta "data/raw/inthewild/meta.csv"
```

### Fast Prototyping / Benchmark Runs (`--max-samples`)
Process a small subset (e.g. 500 samples per dataset) for fast testing:

```bash
python -m src.preprocessing.main --max-samples 500
```

### Resuming Interrupted Processing
If processing is interrupted, re-run the exact same command:
```bash
python -m src.preprocessing.main
```
The pipeline automatically indexes existing `.npy` files and rows in `metadata.csv`. Previously processed files are **skipped without duplicate computation or duplicate metadata entries**.

---

## 7. How to Generate Train / Val / Test Splits

The stratified splitting logic is encapsulated in `src/preprocessing/dataset_split.py`.

It performs **joint stratification** across `(dataset, label)` pairs:
- **70% Training (`train`)**
- **15% Validation (`val`)**
- **15% Testing (`test`)**
- Reproducibility guaranteed via fixed seed `RANDOM_SEED = 42`.
- Strict mathematical disjoint set validation: verifies zero sample leakage ($S_{train} \cap S_{val} = \emptyset$, $S_{train} \cap S_{test} = \emptyset$, $S_{val} \cap S_{test} = \emptyset$).

To run or test splitting independently in Python:
```python
from src.preprocessing.dataset_loader import load_asvspoof_dataset
from src.preprocessing.dataset_split import stratified_split

samples = load_asvspoof_dataset("data/raw/asvspoof")
splits = stratified_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
```

---

## 8. How to Visualize Mel-Spectrograms

Generate diagnostic visual comparison figures comparing Real (`label=0`) and Fake (`label=1`) spectrograms:

```bash
python -m src.preprocessing.visualize_spectrograms
```

### Optional Flags:
- `--samples-per-class 4`: Display 4 real and 4 fake samples in an 8-panel grid.
- `--output "reports/figures/my_spectrograms.png"`: Specify a custom output path.
- Default output location: `data/processed/visualizations/mel_spectrogram_inspection.png`.

---

## 9. Expected Output Directory Structure

After execution, the `data/` directory will adhere to this structure:

```text
data/
├── raw/
│   ├── asvspoof/
│   │   ├── flac/
│   │   └── ASVspoof2021.LA.cm.eval.trl.txt
│   └── inthewild/
│       ├── audio/
│       └── meta.csv
└── processed/
    ├── metadata.csv                       # Master dataset registry
    ├── train/
    │   ├── LA_E_1000147.npy               # (128, 128) float32
    │   ├── wild_sample_001.npy
    │   └── ...
    ├── val/
    │   ├── LA_E_2004581.npy
    │   └── ...
    ├── test/
    │   ├── LA_E_3009941.npy
    │   └── ...
    └── visualizations/
        └── mel_spectrogram_inspection.png  # Diagnostic figure
```

---

## 10. Metadata CSV Format

The file `data/processed/metadata.csv` strictly contains:

```csv
filepath,label,dataset,split
data/processed/train/LA_E_1000147.npy,0,asvspoof,train
data/processed/train/wild_clip_042.npy,1,inthewild,train
data/processed/val/LA_E_2004581.npy,1,asvspoof,val
data/processed/test/wild_clip_108.npy,0,inthewild,test
```

### Column Definitions:
1. `filepath`: Path to the extracted `.npy` file (relative to project root with forward slashes).
2. `label`: `0` for Real (bonafide), `1` for Fake (spoof).
3. `dataset`: Source corpus identifier (`asvspoof` or `inthewild`).
4. `split`: Dataset partition assignment (`train`, `val`, or `test`).

---

## 11. Technical Specifications of Mel-Spectrograms

Parameters implemented via `librosa.feature.melspectrogram`:
- **Audio Sample Rate ($f_s$)**: 16,000 Hz
- **FFT Window Size ($N_{fft}$)**: 1024 points
- **Hop Length ($H$)**: 512 points
- **Mel Filterbanks ($M$)**: 128 frequency bands
- **Frequency Range ($f_{min} \rightarrow f_{max}$)**: 0 Hz to 8,000 Hz (Nyquist limit)
- **Magnitude Scale**: Decibels (dB), computed via `librosa.power_to_db(ref=np.max, top_db=80.0)`
- **Temporal Alignment**: Explicit center-cropping or symmetric silence-floor padding along the time axis to enforce an exact `(128, 128)` matrix shape.

---

## 12. Troubleshooting Common Errors

### 1. `FileNotFoundError: Audio file not found`
- **Cause**: Raw audio files are missing or filenames in protocol do not match audio file stems.
- **Fix**: Verify audio files are inside `data/raw/asvspoof/` or pass `--asvspoof-dir` pointing directly to where the files are stored.

### 2. `ValueError: Interface contract violation: feature matrix has shape ...`
- **Cause**: Corrupted audio file or zero-length signal.
- **Fix**: The pipeline automatically catches corrupted audio and logs a warning while continuing. Unprocessable files are tracked under `counts['failed']`.

### 3. `ModuleNotFoundError: No module named 'librosa'`
- **Cause**: Dependencies are not installed in the active environment.
- **Fix**: Run `pip install -r requirements.txt`.

### 4. `Permission denied on metadata.csv`
- **Cause**: `metadata.csv` is currently open in Microsoft Excel.
- **Fix**: Close Excel or file viewers and rerun the script.

---

## 13. Automated Self-Test Suite

To verify all components and interface contract compliance on synthetic audio before connecting the full multi-gigabyte datasets:

```bash
python -m src.preprocessing.test_pipeline
```

This generates synthetic audio samples, runs the full processing pipeline, validates the output tensors, verifies metadata integrity, and tests resume idempotency.

