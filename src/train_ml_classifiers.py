"""
Training and optimization module for classical ML classifiers (SVM and Random Forest).
Trained on CNN-extracted 128-dimensional handwriting feature embeddings.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    SVM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH,
    CLASSES, RANDOM_SEED
)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_val: float = 10.0,
    gamma: str = "scale"
) -> Tuple[SVC, StandardScaler]:
    """
    Train an RBF kernel Support Vector Machine on scaled CNN features.
    Probability estimation enabled for hybrid ensemble confidence.
    """
    print(f"\n[INFO] Standardizing features and training SVM (C={c_val}, gamma={gamma})...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svm = SVC(
        C=c_val,
        kernel="rbf",
        gamma=gamma,
        probability=True,
        random_state=RANDOM_SEED,
        cache_size=1000
    )
    svm.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))
    print(f"[SVM] Training Accuracy: {train_acc * 100:.2f}%")
    return svm, scaler


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 20
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on CNN features.
    """
    print(f"\n[INFO] Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"[Random Forest] Training Accuracy: {train_acc * 100:.2f}%")
    return rf


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Classifier",
    scaler: StandardScaler = None
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test data and returns metrics dictionary.
    """
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n==================== {model_name} Test Evaluation ====================")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred,
        "probabilities": y_prob
    }


def save_ml_models(
    svm_model: SVC,
    rf_model: RandomForestClassifier,
    scaler: StandardScaler,
    svm_path: Path = SVM_MODEL_PATH,
    rf_path: Path = RF_MODEL_PATH,
    scaler_path: Path = SCALER_PATH
):
    """
    Save trained ML classifiers and feature scaler to disk.
    """
    print(f"\n[INFO] Saving models to {svm_path.parent}...")
    joblib.dump(svm_model, svm_path)
    joblib.dump(rf_model, rf_path)
    joblib.dump(scaler, scaler_path)
    print("[SUCCESS] SVM, Random Forest, and Scaler saved successfully.")


def load_ml_models(
    svm_path: Path = SVM_MODEL_PATH,
    rf_path: Path = RF_MODEL_PATH,
    scaler_path: Path = SCALER_PATH
) -> Tuple[SVC, RandomForestClassifier, StandardScaler]:
    """
    Load saved ML classifiers and scaler from disk.
    """
    svm = joblib.load(svm_path)
    rf = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    return svm, rf, scaler
