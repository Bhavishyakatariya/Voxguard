"""
VoxGuard Dashboard Backend - Configuration
==========================================
Centralized configuration parameters, directory paths, and CORS settings.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Checkpoint paths
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "voxguard_model.pt"
CHECKPOINT_FALLBACK = PROJECT_ROOT / "models" / "voxguard_cnn_best.pt"

# Data paths
SAMPLE_AUDIOS_DIR = PROJECT_ROOT / "data" / "sample_audios"
HEATMAPS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability" / "heatmaps"
HISTORY_FILE = PROJECT_ROOT / "outputs" / "explainability" / "analysis_history.json"
TEMP_UPLOAD_DIR = PROJECT_ROOT / "outputs" / "explainability" / "temp_uploads"
FRONTEND_DIST_DIR = PROJECT_ROOT / "src" / "dashboard" / "frontend" / "dist"

# Server configuration
HOST = "127.0.0.1"
PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

