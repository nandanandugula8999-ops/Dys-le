"""
Full-Scale Master Execution Pipeline for Hybrid AI Dyslexia Detection.
Trains on ALL 151,649 training images and benchmarks across ALL 56,723 test images.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import (
    MODELS_DIR, RESULTS_DIR, CNN_MODEL_PATH, SVM_MODEL_PATH,
    RF_MODEL_PATH, SCALER_PATH, METADATA_PATH, CLASSES,
    TRAIN_SAMPLES_PER_CLASS, EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_DIR
)
from src.data_loader import (
    get_full_train_and_val_datasets, get_full_test_dataset,
    get_balanced_filepaths, create_tf_dataset
)
from src.cnn_model import train_cnn
from src.feature_extractor import DyslexiaFeatureExtractor
from src.train_ml_classifiers import (
    train_svm, train_random_forest, save_ml_models
)
from src.evaluate import run_comprehensive_evaluation


def main():
    total_start = time.time()
    print("=" * 80)
    print("  FULL-SCALE HYBRID AI DYSLEXIA DETECTION PIPELINE (ALL 208,372 IMAGES)")
    print("=" * 80)

    # Step 1: Initialize Streaming Pipelines for Full Dataset
    print("\n[STEP 1/6] Initializing full dataset streams...")
    t0 = time.time()
    train_ds, val_ds = get_full_train_and_val_datasets(batch_size=BATCH_SIZE, val_split=0.10)
    test_ds = get_full_test_dataset(batch_size=BATCH_SIZE)
    print(f"  Streaming pipelines ready in {time.time() - t0:.2f}s")

    # Step 2: CNN Training on Full Training Dataset (151,649 images)
    print("\n[STEP 2/6] Training / Fine-tuning CNN on ALL 151,649 training images...")
    t0 = time.time()
    
    # Balanced class weights for raw training set (65k Corrected, 39k Normal, 46k Reversal)
    class_weights = {
        0: 0.7714,  # Corrected
        1: 1.2851,  # Normal
        2: 1.0806   # Reversal
    }
    print(f"  Applied balanced class weights: {class_weights}")

    initial_model = None
    if CNN_MODEL_PATH.exists():
        print(f"  Loading existing checkpoint from {CNN_MODEL_PATH} for full-scale fine-tuning...")
        initial_model = tf.keras.models.load_model(str(CNN_MODEL_PATH))

    cnn_model, history = train_cnn(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS,
        model_save_path=CNN_MODEL_PATH,
        class_weight=class_weights,
        initial_model=initial_model,
        learning_rate=LEARNING_RATE
    )
    print(f"  CNN full dataset training completed in {time.time() - t0:.2f}s")

    # Step 3: Deep Feature Extraction
    print("\n[STEP 3/6] Extracting 128-dimensional latent representations...")
    t0 = time.time()
    feature_extractor = DyslexiaFeatureExtractor(cnn_model)

    # Prepare 36,000 balanced images for ML classifier training
    print(f"  Preparing {TRAIN_SAMPLES_PER_CLASS * 3} balanced samples for SVM & RF training...")
    train_paths, train_labels = get_balanced_filepaths(TRAIN_DIR, samples_per_class=TRAIN_SAMPLES_PER_CLASS)
    train_ml_ds = create_tf_dataset(train_paths, train_labels, batch_size=512, is_training=False)

    print("  Extracting training features...")
    X_train_feats, y_train = feature_extractor.extract_from_dataset(train_ml_ds)
    print(f"  Training features extracted: {X_train_feats.shape}")

    print("  Extracting test features across ALL 56,723 test images...")
    test_extract_ds = get_full_test_dataset(batch_size=512)
    X_test_feats, y_test = feature_extractor.extract_from_dataset(test_extract_ds)
    print(f"  Test features extracted: {X_test_feats.shape}")
    print(f"  Total feature extraction time: {time.time() - t0:.2f}s")

    # Step 4: Train SVM Classifier
    print(f"\n[STEP 4/6] Training Support Vector Machine on {len(y_train)} features...")
    t0 = time.time()
    svm_model, scaler = train_svm(X_train_feats, y_train, c_val=10.0)
    print(f"  SVM training completed in {time.time() - t0:.2f}s")

    # Step 5: Train Random Forest Classifier
    print(f"\n[STEP 5/6] Training Random Forest Classifier (150 trees) on {len(y_train)} features...")
    t0 = time.time()
    rf_model = train_random_forest(X_train_feats, y_train, n_estimators=150, max_depth=20)
    print(f"  Random Forest training completed in {time.time() - t0:.2f}s")

    # Save Models
    save_ml_models(svm_model, rf_model, scaler, SVM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH)

    # Step 6: Benchmark Evaluation on ALL 56,723 Test Images
    print(f"\n[STEP 6/6] Evaluating all models on ALL {len(y_test)} unseen test images...")
    t0 = time.time()
    eval_results = run_comprehensive_evaluation(
        cnn_model=cnn_model,
        svm_model=svm_model,
        rf_model=rf_model,
        scaler=scaler,
        X_test_imgs=test_ds,
        X_test_feats=X_test_feats,
        y_test=y_test
    )
    print(f"  Evaluation and reporting completed in {time.time() - t0:.2f}s")

    # Save Pipeline Metadata
    metadata = {
        "creation_timestamp": datetime.now().isoformat(),
        "classes": CLASSES,
        "dataset_scope": "Full Scale (151,649 Train / 56,723 Test)",
        "train_image_count": 151649,
        "test_sample_count": len(y_test),
        "ml_train_samples": len(y_train),
        "feature_dimension": 128,
        "input_shape": [64, 64, 1],
        "ensemble_weights": {"CNN": 0.35, "SVM": 0.35, "Random_Forest": 0.30},
        "results_summary": {
            m_name: {
                "accuracy": round(data["metrics"]["accuracy"] * 100, 2),
                "macro_f1": round(data["metrics"]["macro_f1"] * 100, 2)
            }
            for m_name, data in eval_results.items()
        }
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVED] Metadata written to: {METADATA_PATH}")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  FULL-SCALE PIPELINE COMPLETE IN {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
