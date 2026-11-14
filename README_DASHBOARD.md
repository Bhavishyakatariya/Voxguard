# VoxGuard Interactive Web Dashboard

Modern, full-stack Explainable Deepfake Voice Detection Dashboard built for academic major project presentation.

---

## 1. System Architecture

```
                                    +------------------------------+
                                    |       React 18 + Vite        |
                                    |     Tailwind CSS / Lucide    |
                                    |     (Port 5173 - Frontend)   |
                                    +--------------+---------------+
                                                   | HTTP / REST API
                                                   v
                                    +------------------------------+
                                    |   Python REST API Server     |
                                    |    (Port 8000 - Backend)     |
                                    +--------------+---------------+
                                                   |
                     +-----------------------------+-----------------------------+
                     |                                                           |
                     v                                                           v
+------------------------------------------+        +------------------------------------------+
|       Trained PyTorch Model              |        |            XAI Grad-CAM Engine           |
|      (VoxGuardCNN - 4-Block)             |        |    (Target Layer: conv4 / 256 feature)   |
|   `models/voxguard_model.pt`             |        |       `src/explainability/`              |
|   (Calibrated Test Acc: 93.03%)          |        |   - 300-DPI Spectrogram Heatmap Overlay  |
|   (In-The-Wild Benchmark Acc: 96.94%)    |        |   - Natural Language Decision Caption    |
+------------------------------------------+        +------------------------------------------+
```

---

## 2. Features

- **Genuine Neural Inference:** Connected directly to `models/voxguard_model.pt` via PyTorch (`eval()` mode, single startup load).
- **Explainable AI (Grad-CAM):** Generates dual-panel Mel-spectrogram and high-resolution class activation heatmaps highlighting spectral artifacts.
- **Natural Language Rationale:** Contextual captions explaining model focus (temporal windows, frequency bands, spectral distribution).
- **Live Sample Audio Selector:** Instantly test authentic ground-truth real (`sample_real.wav`) and deepfake (`sample_fake.wav`) recordings.
- **Drag & Drop Audio Upload:** Supports `.wav`, `.flac`, and `.mp3` audio files with real-time waveform playback.
- **Audit History Log:** Persistent session history tracking previous predictions, confidence scores, and XAI links.
- **Cyber-Defense Dark Theme:** Clean, academic presentation aesthetic built with Tailwind CSS.

---

## 3. How to Run the Dashboard

### Step 1: Start the Python Backend Server

Open a terminal (PowerShell or Command Prompt) in the project root:

```powershell
.\.venv\Scripts\python.exe -m src.dashboard.backend.main
```

The server will initialize the trained model and start listening on:
`http://localhost:8000`

### Step 2: Start the React Frontend

Open a second terminal window:

```powershell
cd src\dashboard\frontend
npm run dev
```

The Vite dev server will start at:
`http://localhost:5173`

Open your browser to `http://localhost:5173` to interact with the dashboard.

---

## 4. REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health status and loaded model metadata |
| `/api/samples` | `GET` | List available pre-loaded sample audios |
| `/api/analyze/sample` | `POST` | Run inference on a sample file (`{"filename": "..."}`) |
| `/api/analyze` | `POST` | Multipart upload for custom audio analysis |
| `/api/heatmap/<file>` | `GET` | Serve generated Grad-CAM heatmap visualization |
| `/api/history` | `GET` | Retrieve session prediction audit trail |

---

## 5. Sample Testing Files

The dashboard comes with pre-loaded authentic test samples located in `data/sample_audios/`:
- `sample_real.wav`: Ground-truth human voice recording (evaluated with 99.9% REAL confidence).
- `sample_fake.wav`: Synthesized AI deepfake voice recording (evaluated with 100.0% FAKE confidence).

