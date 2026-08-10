"""Flask Web Dashboard Backend for Cephalometric Radiograph Analysis.

Provides API endpoints for starting/stopping training, polling live training metrics
and logs, and running inference on uploaded images.
"""

import os
import sys
import subprocess
import signal
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from inference import CephalometricPredictor, PredictionResult

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = PROJECT_ROOT / "temp_uploads"
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
app.config["STATIC_TEMP"] = PROJECT_ROOT / "static" / "temp"
app.config["STATIC_TEMP"].mkdir(parents=True, exist_ok=True)

# Global variables for training process management
training_process: Optional[subprocess.Popen] = None
training_log_path = PROJECT_ROOT / "reports" / "full_model" / "training.log"
training_log_path.parent.mkdir(parents=True, exist_ok=True)

# Lazy-loaded predictor to save memory and load only when needed
_predictor: Optional[CephalometricPredictor] = None
predictor_lock = threading.Lock()

def get_predictor() -> CephalometricPredictor:
    """Thread-safe lazy-load predictor."""
    global _predictor
    with predictor_lock:
        if _predictor is None:
            config = Config()
            # Try to load the best checkpoint if it exists
            checkpoint_path = PROJECT_ROOT / "checkpoints" / "full_model" / "best_checkpoint.pth"
            if not checkpoint_path.exists():
                checkpoint_path = None
            _predictor = CephalometricPredictor(config=config, checkpoint_path=checkpoint_path)
        return _predictor


# ============================================================
# ROUTING
# ============================================================

@app.route("/")
def index():
    """Render index dashboard page."""
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get active model and training config settings."""
    config = Config()
    return jsonify({
        "ablation": {
            "use_afem": config.ablation.use_afem,
            "use_cffm": config.ablation.use_cffm,
            "use_acam": config.ablation.use_acam,
            "use_grlm": config.ablation.use_grlm,
        },
        "backbone": {
            "model_name": config.backbone.model_name,
            "pretrained": config.backbone.pretrained,
        },
        "training": {
            "epochs": config.training.num_epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "early_stopping_patience": config.training.early_stopping_patience,
        }
    })


@app.route("/api/train/start", methods=["POST"])
def start_train():
    """Start full model training process in background with optional parameter overrides."""
    global training_process
    if training_process is not None and training_process.poll() is None:
        return jsonify({"status": "error", "message": "Training is already running."}), 400

    # Extract overrides from JSON request
    req_data = request.json or {}
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "train_full_model.py")]

    if "epochs" in req_data and req_data["epochs"] is not None:
        cmd.extend(["--epochs", str(req_data["epochs"])])
    if "batch_size" in req_data and req_data["batch_size"] is not None:
        cmd.extend(["--batch_size", str(req_data["batch_size"])])
    if "learning_rate" in req_data and req_data["learning_rate"] is not None:
        cmd.extend(["--lr", str(req_data["learning_rate"])])
    if "early_stopping_patience" in req_data and req_data["early_stopping_patience"] is not None:
        cmd.extend(["--patience", str(req_data["early_stopping_patience"])])
    if "phase1" in req_data and req_data["phase1"] is not None:
        cmd.extend(["--phase1", str(req_data["phase1"])])
    if "phase2" in req_data and req_data["phase2"] is not None:
        cmd.extend(["--phase2", str(req_data["phase2"])])

    # Reset logs
    with open(training_log_path, "w", encoding="utf-8") as f:
        f.write("[System] Starting training process with command: " + " ".join(cmd) + "\n\n")

    # Start subprocess running python train_full_model.py
    try:
        # We redirect stdout/stderr to a file that the logs endpoint will read
        log_file = open(training_log_path, "a", encoding="utf-8")
        training_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        return jsonify({"status": "success", "message": "Training started successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start training: {str(e)}"}), 500


@app.route("/api/train/stop", methods=["POST"])
def stop_train():
    """Terminate the training process."""
    global training_process
    if training_process is None or training_process.poll() is not None:
        return jsonify({"status": "error", "message": "Training is not running."}), 400

    try:
        if sys.platform == "win32":
            # Send CTRL_BREAK to process group
            training_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            training_process.terminate()
        
        training_process.wait(timeout=5)
        training_process = None
        
        with open(training_log_path, "a", encoding="utf-8") as f:
            f.write("\n[System] Training process terminated by user.\n")
            
        return jsonify({"status": "success", "message": "Training stopped successfully."})
    except Exception as e:
        # Force kill as fallback
        try:
            training_process.kill()
            training_process = None
            return jsonify({"status": "success", "message": "Training force-killed."})
        except Exception as ke:
            return jsonify({"status": "error", "message": f"Failed to stop training: {str(ke)}"}), 500


@app.route("/api/train/status", methods=["GET"])
def train_status():
    """Get training running status and latest metrics."""
    global training_process
    is_running = training_process is not None and training_process.poll() is None

    history_path = PROJECT_ROOT / "reports" / "full_model" / "training_history.csv"
    latest_metrics = {}
    current_epoch = 0

    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if reader:
                    latest = reader[-1]
                    current_epoch = int(latest["epoch"])
                    latest_metrics = {
                        "train_loss": float(latest["train_loss"]),
                        "val_loss": float(latest["val_loss"]),
                        "accuracy": float(latest["accuracy"]),
                        "macro_f1": float(latest["macro_f1"]),
                        "balanced_accuracy": float(latest["balanced_accuracy"]),
                        "learning_rate": float(latest["learning_rate"]),
                    }
        except Exception as e:
            app.logger.error(f"Error reading history CSV: {e}")

    # Check if final report exists
    report_exists = (PROJECT_ROOT / "reports" / "full_model" / "full_model_report.md").exists()

    return jsonify({
        "is_running": is_running,
        "current_epoch": current_epoch,
        "latest_metrics": latest_metrics,
        "report_exists": report_exists,
    })


@app.route("/api/train/history", methods=["GET"])
def train_history():
    """Parse training history CSV and return series data for charts."""
    history_path = PROJECT_ROOT / "reports" / "full_model" / "training_history.csv"
    epochs = []
    train_loss = []
    val_loss = []
    val_accuracy = []
    val_f1 = []

    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row["epoch"]))
                    train_loss.append(float(row["train_loss"]))
                    val_loss.append(float(row["val_loss"]))
                    val_accuracy.append(float(row["accuracy"]))
                    val_f1.append(float(row["macro_f1"]))
        except Exception as e:
            app.logger.error(f"Error parsing history: {e}")

    return jsonify({
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
    })


@app.route("/api/train/logs", methods=["GET"])
def train_logs():
    """Return the tail of training logs (last 100 lines)."""
    if not training_log_path.exists():
        return jsonify({"logs": ""})

    try:
        with open(training_log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            tail_lines = lines[-100:]
            return jsonify({"logs": "".join(tail_lines)})
    except Exception as e:
        return jsonify({"logs": f"Error loading logs: {str(e)}"})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept file upload, perform inference, and return structured results."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty file name."}), 400

    # Save original image temporarily
    orig_path = app.config["UPLOAD_FOLDER"] / "temp_input.png"
    file.save(str(orig_path))

    try:
        predictor = get_predictor()
        # Predict, requesting embedding and enhanced processed image
        result: PredictionResult = predictor.predict(
            image=orig_path,
            return_embedding=True,
            return_processed_image=True,
        )

        # Save processed enhanced image to static temp so the UI can display it
        enhanced_url = None
        if result.processed_image is not None:
            enhanced_path = app.config["STATIC_TEMP"] / "temp_enhanced.png"
            cv2.imwrite(str(enhanced_path), result.processed_image)
            # URL to access it in template
            enhanced_url = "/static/temp/temp_enhanced.png?t=" + str(time.time())

        # Copy original image to static temp for UI display
        static_orig_path = app.config["STATIC_TEMP"] / "temp_orig.png"
        import shutil
        shutil.copy2(orig_path, static_orig_path)
        orig_url = "/static/temp/temp_orig.png?t=" + str(time.time())

        # Create structured output
        serialized = result.to_dict(include_arrays=False)
        serialized["orig_url"] = orig_url
        serialized["enhanced_url"] = enhanced_url
        
        # Format embedding stats
        if result.embedding is not None:
            norm = float(np.linalg.norm(result.embedding))
            serialized["embedding_stats"] = {
                "shape": list(result.embedding.shape),
                "l2_norm": norm,
                "first_elements": [float(v) for v in result.embedding[:5]],
            }

        return jsonify({"status": "success", "result": serialized})

    except Exception as e:
        app.logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Inference failed: {str(e)}"}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static assets."""
    return send_from_directory(app.config["STATIC_TEMP"].parent, filename)


if __name__ == "__main__":
    print("=" * 65)
    print("STARTING CEPHALOMETRIC ANALYSIS WEB DASHBOARD")
    print("=" * 65)
    print("To open the dashboard, open your browser and visit:")
    print("http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=True)
