"""
VoxGuard Dashboard Backend - High-Performance Standard HTTP API Server
========================================================================
Built with Python's standard library http.server to run natively on any system
without requiring external network package installations, providing all REST API
endpoints for the React/Tailwind web dashboard:
- GET  /api/health
- POST /api/analyze
- POST /api/analyze/sample
- GET  /api/samples
- GET  /api/heatmap/<filename>
- GET  /api/history
"""

import cgi
import io
import json
import mimetypes
import os
import shutil
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse, parse_qs

from src.dashboard.backend.config import (
    HOST,
    PORT,
    TEMP_UPLOAD_DIR,
    HEATMAPS_OUTPUT_DIR,
    SAMPLE_AUDIOS_DIR,
    HISTORY_FILE,
    FRONTEND_DIST_DIR,
)
from src.dashboard.backend.inference import model_service
from src.utils.logger import get_logger

logger = get_logger("VoxGuard.Dashboard.Server")

# Ensure required runtime output directories exist
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_AUDIOS_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> List[Dict[str, Any]]:
    """Loads past analysis history from JSON file."""
    if not HISTORY_FILE.is_file():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(record: Dict[str, Any]):
    """Appends an analysis record to history file."""
    history = load_history()
    # Insert newest at top
    history.insert(0, record)
    # Keep last 50 entries
    history = history[:50]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


class VoxGuardAPIHandler(BaseHTTPRequestHandler):
    """
    RESTful Request Handler implementing VoxGuard Dashboard API specifications.
    """

    def _set_cors_headers(self, content_type: str = "application/json"):
        """Attach permissive CORS headers for local React/Vite development."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)

    def do_OPTIONS(self):
        """Handle pre-flight CORS requests."""
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def _send_json(self, data: Any, status_code: int = 200):
        """Helper to send JSON response."""
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self._set_cors_headers("application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_error_json(self, message: str, status_code: int = 400):
        """Helper to send structured error response."""
        logger.warning(f"API Error ({status_code}): {message}")
        self._send_json({"error": True, "message": message}, status_code=status_code)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # 1. Health check: /api/health
        if path == "/api/health":
            status = model_service.get_status()
            self._send_json({
                "status": "online" if status["model_loaded"] else "degraded",
                "backend": "VoxGuard Python API Engine",
                "model": status,
                "timestamp": datetime.now().isoformat(),
            })
            return

        # 2. List available sample audios: /api/samples
        if path == "/api/samples":
            samples = []
            if SAMPLE_AUDIOS_DIR.is_dir():
                for p in sorted(SAMPLE_AUDIOS_DIR.glob("*.wav")):
                    samples.append({
                        "filename": p.name,
                        "size_bytes": p.stat().st_size,
                        "display_name": p.stem.replace("_", " ").title(),
                    })
            self._send_json({"samples": samples})
            return

        # 3. History endpoint: /api/history
        if path == "/api/history":
            history = load_history()
            real_count = sum(1 for h in history if h.get("verdict") == "REAL")
            fake_count = sum(1 for h in history if h.get("verdict") == "FAKE")
            self._send_json({
                "history": history,
                "total_analyses": len(history),
                "real_count": real_count,
                "fake_count": fake_count,
            })
            return

        # 4. Serve heatmap image: /api/heatmap/<filename>
        if path.startswith("/api/heatmap/"):
            filename = Path(path[len("/api/heatmap/"):].strip("/")).name  # Path traversal protection
            target_img = HEATMAPS_OUTPUT_DIR / filename
            if not target_img.is_file():
                self._send_error_json("Heatmap image not found", status_code=404)
                return

            try:
                with open(target_img, "rb") as f:
                    img_bytes = f.read()
                self.send_response(200)
                self._set_cors_headers("image/png")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.end_headers()
                self.wfile.write(img_bytes)
                return
            except Exception as e:
                self._send_error_json(f"Failed to read image: {e}", status_code=500)
                return

        # 5. Serve sample audio: /api/audio/sample/<filename>
        if path.startswith("/api/audio/sample/"):
            filename = Path(path[len("/api/audio/sample/"):].strip("/")).name
            target_wav = SAMPLE_AUDIOS_DIR / filename
            if not target_wav.is_file():
                self._send_error_json("Sample audio not found", status_code=404)
                return

            try:
                with open(target_wav, "rb") as f:
                    audio_bytes = f.read()
                self.send_response(200)
                self._set_cors_headers("audio/wav")
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.end_headers()
                self.wfile.write(audio_bytes)
                return
            except Exception as e:
                self._send_error_json(f"Failed to read audio: {e}", status_code=500)
                return

        # 6. Static frontend assets serving (Vite build)
        if not path.startswith("/api"):
            if FRONTEND_DIST_DIR.is_dir():
                clean_path = path.lstrip("/")
                target_file = FRONTEND_DIST_DIR / clean_path if clean_path else FRONTEND_DIST_DIR / "index.html"
                if not target_file.is_file():
                    target_file = FRONTEND_DIST_DIR / "index.html"

                content_type, _ = mimetypes.guess_type(str(target_file))
                if not content_type:
                    content_type = "text/html" if target_file.suffix == ".html" else "application/octet-stream"

                try:
                    with open(target_file, "rb") as f:
                        file_data = f.read()
                    self.send_response(200)
                    self._set_cors_headers(content_type)
                    self.send_header("Content-Length", str(len(file_data)))
                    self.end_headers()
                    self.wfile.write(file_data)
                    return
                except Exception as e:
                    self._send_error_json(f"Static file error: {e}", status_code=500)
                    return

        self._send_error_json(f"Endpoint not found: {self.path}", status_code=404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # 1. Analyze sample audio: /api/analyze/sample
        if path == "/api/analyze/sample":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body) if body else {}
            except Exception:
                self._send_error_json("Invalid JSON payload", status_code=400)
                return

            sample_filename = payload.get("filename")
            if not sample_filename:
                self._send_error_json("Parameter 'filename' is required", status_code=400)
                return

            safe_name = Path(sample_filename).name  # Prevent directory traversal
            sample_path = SAMPLE_AUDIOS_DIR / safe_name
            if not sample_path.is_file():
                self._send_error_json(f"Sample audio not found: {safe_name}", status_code=404)
                return

            try:
                result = model_service.analyze_audio(sample_path)
                result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_history(result)
                self._send_json(result)
                return
            except Exception as exc:
                self._send_error_json(f"Inference error: {exc}", status_code=500)
                return

        # 2. Upload and analyze audio: /api/analyze
        if path == "/api/analyze":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_error_json("Expected multipart/form-data content type", status_code=400)
                return

            try:
                # Parse multipart form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers["Content-Type"],
                    },
                )

                if "file" not in form:
                    self._send_error_json("No audio file provided in field 'file'", status_code=400)
                    return

                file_item = form["file"]
                if not file_item.filename:
                    self._send_error_json("Uploaded file must have a filename", status_code=400)
                    return

                # Validate extension (.wav, .flac, .mp3)
                safe_orig_name = Path(file_item.filename).name
                ext = Path(safe_orig_name).suffix.lower()
                if ext not in [".wav", ".flac", ".mp3", ".ogg"]:
                    self._send_error_json(f"Unsupported file format '{ext}'. Upload .wav, .flac, or .mp3.", status_code=400)
                    return

                # Save file temporarily with unique timestamp
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = f"upload_{timestamp_str}_{safe_orig_name}"
                temp_path = TEMP_UPLOAD_DIR / temp_filename

                with open(temp_path, "wb") as f_out:
                    shutil.copyfileobj(file_item.file, f_out)

                # Check file size (ensure non-empty)
                if temp_path.stat().st_size == 0:
                    temp_path.unlink(missing_ok=True)
                    self._send_error_json("Uploaded file is empty (0 bytes).", status_code=400)
                    return

                # Run inference & explainability
                result = model_service.analyze_audio(temp_path)
                result["filename"] = safe_orig_name
                result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_history(result)

                self._send_json(result)
                return
            except Exception as exc:
                self._send_error_json(f"Analysis processing failure: {exc}", status_code=500)
                return

        self._send_error_json(f"Endpoint not found: {self.path}", status_code=404)


def run_server(host: str = HOST, port: int = PORT):
    """Start the VoxGuard HTTP API server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, VoxGuardAPIHandler)
    logger.info(f"VoxGuard API Server running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server()

