"""
CNN-only fallback server for NeuroScan AI.
Avoids sklearn (blocked by Application Control policy on this machine).
Uses the trained CNN classification head directly; SVM/RF fields are
mirrored from CNN so the existing frontend keeps working.
Run with Python 3.11:  .\.venv311\Scripts\python.exe server_cnn_only.py
"""
import os
import io
import sys
import base64
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from src.config import CLASSES, CLASS_RISK_MAP, IMG_HEIGHT, IMG_WIDTH, CNN_MODEL_PATH

print("[INFO] Loading CNN (fallback mode, no sklearn)...")
cnn_model = tf.keras.models.load_model(str(CNN_MODEL_PATH))
print("[SUCCESS] CNN loaded:", CNN_MODEL_PATH.name)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def preprocess_image(pil_img: Image.Image):
    """Same adaptive preprocessing as hybrid pipeline (grayscale 64x64)."""
    arr = np.array(pil_img.convert("L"), dtype=np.float32)
    h, w = arr.shape
    border = np.concatenate([
        arr[:max(2, int(h * 0.05)), :].flatten(),
        arr[-max(2, int(h * 0.05)):, :].flatten(),
        arr[:, :max(2, int(w * 0.05))].flatten(),
        arr[:, -max(2, int(w * 0.05)):].flatten(),
    ])
    bg_val = float(np.median(border))
    if bg_val > 110.0:
        arr = np.clip(bg_val - arr, 0.0, 255.0)
        if arr.max() > 0:
            arr = (arr / arr.max()) * 255.0
        arr[arr < 0.25 * arr.max()] = 0.0
        nz = np.argwhere(arr > 0)
        if len(nz) >= 20:
            ymin, xmin = nz.min(axis=0)
            ymax, xmax = nz.max(axis=0)
            cropped = arr[ymin:ymax + 1, xmin:xmax + 1]
            ch, cw = cropped.shape
            side = max(ch, cw)
            pad = int(side * 1.35)
            padded = np.zeros((pad, pad), dtype=np.float32)
            padded[(pad - ch) // 2:(pad - ch) // 2 + ch,
                   (pad - cw) // 2:(pad - cw) // 2 + cw] = cropped
            arr = padded
    proc = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    resized = proc.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    norm = np.array(resized, dtype=np.float32) / 255.0
    tensor = np.expand_dims(norm, axis=(0, -1))
    return tensor, resized


def image_to_base64(pil_img: Image.Image, size=(100, 100)) -> str:
    thumb = pil_img.copy().convert("RGB")
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict_one(pil_img: Image.Image):
    x, proc_pil = preprocess_image(pil_img)
    probs = cnn_model(x, training=False).numpy()[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    conf = float(probs[idx])
    risk = CLASS_RISK_MAP[label]
    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    entry = {
        "prediction": label,
        "confidence": conf,
        "confidence_percentage": f"{conf * 100:.1f}%",
        "backbone": "custom_cnn (CNN-only fallback)",
        "risk_level": risk["level"],
        "risk_description": risk["description"] + " [CNN-only mode: SVM/RF unavailable on this PC due to security policy.]",
        "risk_color": risk["color"],
        "unanimous_agreement": True,
        "vote_distribution": {c: (1 if c == label else 0) for c in CLASSES},
        "ensemble_probabilities": prob_dict,
        "models": {
            "CNN": {"backbone": "custom_cnn", "prediction": label, "confidence": conf, "probabilities": prob_dict},
            "SVM": {"prediction": label + " (unavailable)", "confidence": conf, "probabilities": prob_dict},
            "Random_Forest": {"prediction": label + " (unavailable)", "confidence": conf, "probabilities": prob_dict},
        },
        "feature_vector_sample": [],
        "mode": "cnn_only",
    }
    return entry, proc_pil


def aggregate(images, filenames):
    individuals = []
    sum_probs = np.zeros(len(CLASSES))
    for i, img in enumerate(images):
        res, _ = predict_one(img)
        res["filename"] = filenames[i] if i < len(filenames) else f"Image_{i+1}"
        individuals.append(res)
        for ci, c in enumerate(CLASSES):
            sum_probs[ci] += res["ensemble_probabilities"][c]
    n = len(images)
    avg = sum_probs / max(n, 1)
    oi = int(np.argmax(avg))
    olabel = CLASSES[oi]
    oconf = float(avg[oi])
    preds = [r["prediction"] for r in individuals]
    counts = {c: preds.count(c) for c in CLASSES}
    rev_rate = counts.get("Reversal", 0) / n * 100 if n else 0
    corr_rate = counts.get("Corrected", 0) / n * 100 if n else 0
    norm_rate = counts.get("Normal", 0) / n * 100 if n else 0
    if rev_rate >= 20.0 or olabel == "Reversal":
        rl, rc = "High Risk (Dyslexia Indicator)", "#dc3545"
        rd = f"Significant reversal frequency ({counts.get('Reversal',0)}/{n}, {rev_rate:.1f}%). CNN-only screening."
    elif corr_rate >= 30.0 or olabel == "Corrected":
        rl, rc = "Moderate Risk", "#ffc107"
        rd = f"Frequent overwriting ({counts.get('Corrected',0)}/{n}, {corr_rate:.1f}%). CNN-only screening."
    else:
        rl, rc = "Low Risk", "#28a745"
        rd = f"Predominantly normal patterns ({counts.get('Normal',0)}/{n}, {norm_rate:.1f}%). CNN-only screening."
    avg_dict = {CLASSES[i]: float(avg[i]) for i in range(len(CLASSES))}
    return {
        "total_images": n,
        "overall_prediction": olabel,
        "overall_confidence": oconf,
        "overall_confidence_percentage": f"{oconf * 100:.1f}%",
        "average_probabilities": avg_dict,
        "class_counts": counts,
        "reversal_count": counts.get("Reversal", 0),
        "reversal_rate": rev_rate,
        "corrected_count": counts.get("Corrected", 0),
        "corrected_rate": corr_rate,
        "normal_count": counts.get("Normal", 0),
        "normal_rate": norm_rate,
        "overall_risk_level": rl,
        "overall_risk_color": rc,
        "overall_risk_description": rd,
        "individual_results": individuals,
        "model_averages": {"CNN": avg_dict, "SVM": avg_dict, "Random_Forest": avg_dict},
        "mode": "cnn_only",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "backbone": "custom_cnn",
                    "mode": "cnn_only_fallback",
                    "models": {"CNN": "Active", "SVM": "Blocked by Application Control policy",
                               "Random_Forest": "Blocked by Application Control policy",
                               "Hybrid_Consensus": "CNN-only fallback"}})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify({"error": "No image files provided."}), 400
    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files selected."}), 400
    images, filenames, thumbs, proc_thumbs = [], [], [], []
    for f in files:
        try:
            img = Image.open(f.stream).convert("L")
            images.append(img)
            filenames.append(f.filename)
            thumbs.append(image_to_base64(img))
            _, proc = preprocess_image(img)
            proc_thumbs.append(image_to_base64(proc))
        except Exception as e:
            return jsonify({"error": f"Failed to process '{f.filename}': {e}"}), 400
    out = aggregate(images, filenames)
    for i, r in enumerate(out["individual_results"]):
        r["thumbnail_base64"] = thumbs[i]
        r["processed_thumbnail_base64"] = proc_thumbs[i]
    return jsonify(out)


@app.route("/api/demo", methods=["GET"])
def demo():
    from app.utils import get_demo_samples
    dtype = request.args.get("type", "normal").lower()
    samples = get_demo_samples(num_per_class=2)
    if dtype == "suite":
        wanted = [(c, p, n) for c in CLASSES for p, n in samples.get(c, [])]
    else:
        key = {"reversal": "Reversal", "corrected": "Corrected"}.get(dtype, "Normal")
        wanted = [(key, p, n) for p, n in samples.get(key, [])[:2]]
    images, filenames, thumbs = [], [], []
    for c, p, n in wanted:
        img = Image.open(p).convert("L")
        images.append(img)
        filenames.append(f"{c}_{n}")
        thumbs.append(image_to_base64(img))
    if not images:
        return jsonify({"error": "No demo images available."}), 500
    out = aggregate(images, filenames)
    for i, r in enumerate(out["individual_results"]):
        r["thumbnail_base64"] = thumbs[i]
    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] CNN-only server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
