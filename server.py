"""
Flask Server for NeuroScan AI - Dyslexia Handwriting Screening Web Application.
Serves dedicated HTML, CSS, and JS frontend with REST API endpoints for
single-image and multi-image hybrid inference with result averaging.
"""

import os
import io
import sys
import base64
from pathlib import Path
from typing import List
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CLASSES, CLASS_RISK_MAP, RESULTS_DIR, MODELS_DIR,
    CNN_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH
)
from src.hybrid_pipeline import HybridDyslexiaDetector
from app.utils import get_demo_samples, generate_batch_clinical_report

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Initialize Hybrid AI Detector
print("[INFO] Initializing Hybrid AI Dyslexia Detector in Flask server...")
detector = HybridDyslexiaDetector(
    cnn_path=CNN_MODEL_PATH,
    svm_path=SVM_MODEL_PATH,
    rf_path=RF_MODEL_PATH,
    scaler_path=SCALER_PATH
)
print("[SUCCESS] Hybrid AI Detector loaded and ready.")


def image_to_base64(pil_img: Image.Image, size=(100, 100)) -> str:
    """Helper to convert PIL image to base64 data URI for thumbnail display."""
    thumb = pil_img.copy().convert("RGB")
    thumb.thumbnail(size)
    buffered = io.BytesIO()
    thumb.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.route("/")
def index():
    """Renders the dedicated HTML5 frontend."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models": {
            "CNN": "Active (98.8% Val Acc)",
            "SVM": "Active (97.0% Train Acc)",
            "Random_Forest": "Active (98.8% Train Acc)",
            "Hybrid_Consensus": "Active (88.67% Test Acc)"
        }
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts 1 or multiple uploaded images, runs hybrid inference,
    and returns aggregated average results and individual breakdowns.
    """
    if "images" not in request.files:
        return jsonify({"error": "No image files provided in request."}), 400

    uploaded_files = request.files.getlist("images")
    if not uploaded_files or uploaded_files[0].filename == "":
        return jsonify({"error": "No files selected."}), 400

    images: List[Image.Image] = []
    filenames: List[str] = []
    thumbnails: List[str] = []

    for f in uploaded_files:
        try:
            img = Image.open(f.stream).convert("L")
            images.append(img)
            filenames.append(f.filename)
            thumbnails.append(image_to_base64(img))
        except Exception as e:
            return jsonify({"error": f"Failed to process image '{f.filename}': {str(e)}"}), 400

    # Execute batch prediction and result averaging
    batch_result = detector.predict_batch(images, filenames)

    # Attach base64 thumbnails to each individual result
    for idx, ind_res in enumerate(batch_result["individual_results"]):
        ind_res["thumbnail_base64"] = thumbnails[idx]
        try:
            prep_pil = detector.preprocess_to_pil(images[idx])
            ind_res["processed_thumbnail_base64"] = image_to_base64(prep_pil)
        except Exception:
            ind_res["processed_thumbnail_base64"] = thumbnails[idx]

    return jsonify(batch_result)


@app.route("/api/demo", methods=["GET"])
def demo():
    """
    Executes prediction on real benchmark test samples.
    Query param ?type=normal | reversal | corrected | suite
    """
    demo_type = request.args.get("type", "normal").lower()
    demo_samples = get_demo_samples(num_per_class=2)

    images: List[Image.Image] = []
    filenames: List[str] = []
    thumbnails: List[str] = []

    if demo_type == "suite":
        # 2 from each class (6 total)
        for c in CLASSES:
            for p, name in demo_samples.get(c, []):
                img = Image.open(p).convert("L")
                images.append(img)
                filenames.append(f"{c}_{name}")
                thumbnails.append(image_to_base64(img))
    elif demo_type == "reversal":
        for p, name in demo_samples.get("Reversal", [])[:2]:
            img = Image.open(p).convert("L")
            images.append(img)
            filenames.append(f"Reversal_{name}")
            thumbnails.append(image_to_base64(img))
    elif demo_type == "corrected":
        for p, name in demo_samples.get("Corrected", [])[:2]:
            img = Image.open(p).convert("L")
            images.append(img)
            filenames.append(f"Corrected_{name}")
            thumbnails.append(image_to_base64(img))
    else:  # normal
        for p, name in demo_samples.get("Normal", [])[:2]:
            img = Image.open(p).convert("L")
            images.append(img)
            filenames.append(f"Normal_{name}")
            thumbnails.append(image_to_base64(img))

    if not images:
        return jsonify({"error": "No demo images available."}), 500

    batch_result = detector.predict_batch(images, filenames)
    for idx, ind_res in enumerate(batch_result["individual_results"]):
        ind_res["thumbnail_base64"] = thumbnails[idx]

    return jsonify(batch_result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[INFO] Starting NeuroScan AI Web Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
