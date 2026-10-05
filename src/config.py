"""
Configuration settings for Hybrid AI Dyslexia Detection System.
Includes paths, image parameters, model hyper-parameters, and class definitions.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset" / "extracted" / "Dataset Dyslexia_Password WanAsy321" / "Gambo"
TRAIN_DIR = DATASET_DIR / "Train"
TEST_DIR = DATASET_DIR / "Test"

# Output directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Image parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Target Classes
CLASSES = ["Corrected", "Normal", "Reversal"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

# Clinical Risk Mapping
CLASS_RISK_MAP = {
    "Normal": {
        "level": "Low Risk",
        "description": "Standard stroke patterns and orientation. No characteristic dyslexic reversal detected.",
        "color": "#28a745"
    },
    "Corrected": {
        "level": "Moderate Risk",
        "description": "Evidence of repeated overwriting / stroke alterations, indicating handwriting uncertainty or compensatory correction.",
        "color": "#ffc107"
    },
    "Reversal": {
        "level": "High Risk (Dyslexia Indicator)",
        "description": "Strong mirror reversal / inverted letter orientation detected, a key motor-perceptual marker associated with dyslexia.",
        "color": "#dc3545"
    }
}

# Saved Model Filenames
CNN_MODEL_PATH = MODELS_DIR / "cnn_feature_extractor.keras"
SVM_MODEL_PATH = MODELS_DIR / "svm_classifier.joblib"
RF_MODEL_PATH = MODELS_DIR / "rf_classifier.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Backbone selection: "custom_cnn" (default, backward compatible) or "mobilenetv2"
# Override with env var: DYSLEXIA_BACKBONE=mobilenetv2
BACKBONE = os.environ.get("DYSLEXIA_BACKBONE", "custom_cnn").lower()
VALID_BACKBONES = ("custom_cnn", "mobilenetv2")

# MobileNetV2 backend settings
# MobileNetV2 requires 3-channel RGB input, min 32x32. 96x96 is a good
# speed/accuracy tradeoff for 28-29px handwriting glyphs upscaled from Gambo.
MOBILENET_INPUT_SIZE = 96
MOBILENET_INPUT_SHAPE = (MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE, 3)
MOBILENET_ALPHA = 1.0
MOBILENET_DROPOUT = 0.4
MOBILENET_FREEZE_BASE = True
# Fine-tune last N layers of MobileNetV2 when unfreezing (None = keep frozen)
MOBILENET_FINE_TUNE_FROM = None
MOBILENET_MODEL_PATH = MODELS_DIR / "mobilenetv2_feature_extractor.keras"
# Separate SVM/RF/scaler artifacts per backbone to avoid mixing 64x64-gray
# custom-CNN features with 96x96-RGB MobileNetV2 features.
MOBILENET_SVM_PATH = MODELS_DIR / "mobilenetv2_svm_classifier.joblib"
MOBILENET_RF_PATH = MODELS_DIR / "mobilenetv2_rf_classifier.joblib"
MOBILENET_SCALER_PATH = MODELS_DIR / "mobilenetv2_feature_scaler.joblib"


def backbone_model_paths(backbone: str = BACKBONE):
    """Return (cnn_path, svm_path, rf_path, scaler_path) for a backbone."""
    b = (backbone or "custom_cnn").lower()
    if b == "mobilenetv2":
        return MOBILENET_MODEL_PATH, MOBILENET_SVM_PATH, MOBILENET_RF_PATH, MOBILENET_SCALER_PATH
    return CNN_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH

# Training parameters for full dataset scaling
RANDOM_SEED = 42
BATCH_SIZE = 256
EPOCHS = 2
LEARNING_RATE = 3e-4
FEATURE_DIM = 128

# Dataset sampling for balanced ML classifiers
TRAIN_SAMPLES_PER_CLASS = 12000  # 36,000 total balanced samples for SVM & RF
VAL_SAMPLES_PER_CLASS = 2000    # 6,000 validation samples
TEST_SAMPLES_PER_CLASS = None   # None = evaluate on ALL 56,723 test images!

