"""
Evaluation and visualization module for the Hybrid Dyslexia Detection system.
Generates comprehensive comparative performance charts, confusion matrices, and ROC curves.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

from src.config import RESULTS_DIR, CLASSES, NUM_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes accuracy, precision, recall, f1, and per-class metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i, c_name in enumerate(CLASSES):
        p, r, f1, s = precision_recall_fscore_support(y_true == i, y_pred == i, average="binary")
        per_class[c_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "support": int(np.sum(y_true == i))
        }

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            roc_auc = {}
            for i, c_name in enumerate(CLASSES):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc[c_name] = float(auc(fpr, tpr))
            metrics["roc_auc"] = roc_auc
        except Exception as e:
            print(f"Warning: ROC AUC computation failed: {e}")

    return metrics


def plot_confusion_matrices(results: Dict[str, Dict[str, Any]], y_true: np.ndarray, save_path: Path):
    """
    Plots side-by-side 2x2 confusion matrices for CNN, SVM, RF, and Hybrid Ensemble.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    model_keys = list(results.keys())[:4]
    palette = ["Blues", "Greens", "Purples", "Oranges"]

    for i, m_name in enumerate(model_keys):
        cm = np.array(results[m_name]["metrics"]["confusion_matrix"])
        # Normalize by true row sum
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=palette[i % len(palette)],
            ax=axes[i],
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"}
        )
        acc = results[m_name]["metrics"]["accuracy"] * 100
        axes[i].set_title(f"{m_name} (Acc: {acc:.2f}%)", fontsize=14, weight="bold", pad=10)
        axes[i].set_xlabel("Predicted Class", fontsize=11)
        axes[i].set_ylabel("True Class", fontsize=11)

    plt.suptitle("Confusion Matrix Comparison: Dyslexia Handwriting Screening", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrices plot saved to: {save_path}")


def plot_model_comparison_bar(summary_df: pd.DataFrame, save_path: Path):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1.
    """
    plt.figure(figsize=(12, 6))

    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(len(summary_df))
    width = 0.2

    colors = ["#2b5c8f", "#38a169", "#d69e2e", "#e53e3e"]

    for i, metric in enumerate(metrics):
        values = summary_df[metric].values
        plt.bar(x + (i - 1.5) * width, values, width, label=metric, color=colors[i], alpha=0.9)

    plt.xlabel("Model Architecture", fontsize=12, weight="bold")
    plt.ylabel("Score (%)", fontsize=12, weight="bold")
    plt.title("Comparative Performance Analysis (CNN vs SVM vs RF vs Hybrid)", fontsize=14, weight="bold", pad=15)
    plt.xticks(x, summary_df["Model"], fontsize=11, weight="bold")
    plt.ylim(0, 105)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Model comparison bar chart saved to: {save_path}")


def plot_roc_curves(results: Dict[str, Dict[str, Any]], y_true: np.ndarray, save_path: Path):
    """
    Plots One-vs-Rest multiclass ROC curves for all models.
    """
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    styles = {
        "Standalone CNN": ("#1f77b4", "-"),
        "CNN + SVM (RBF)": ("#2ca02c", "--"),
        "CNN + Random Forest": ("#ff7f0e", "-."),
        "Hybrid Ensemble (Consensus)": ("#d62728", "-")
    }

    for class_idx, class_name in enumerate(CLASSES):
        ax = axes[class_idx]
        for m_name, res in results.items():
            probs = res.get("probabilities")
            if probs is not None:
                fpr, tpr, _ = roc_curve(y_bin[:, class_idx], probs[:, class_idx])
                roc_auc = auc(fpr, tpr)
                color, ls = styles.get(m_name, ("#7f7f7f", "-"))
                lw = 2.5 if "Hybrid" in m_name else 1.8
                ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                        label=f"{m_name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
        ax.set_title(f"ROC: {class_name}", fontsize=13, weight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(alpha=0.3)

    plt.suptitle("Multiclass ROC Curves for Dyslexia Detection Models", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] ROC curves saved to: {save_path}")


def plot_feature_pca(features: np.ndarray, labels: np.ndarray, save_path: Path):
    """
    Visualizes 128-dimensional CNN features reduced to 2D using PCA.
    Demonstrates class separability learned by the deep feature extractor.
    """
    print("[INFO] Computing 2D PCA on 128-d CNN features...")
    if len(features) > 6000:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(len(features), size=6000, replace=False)
        features = features[idx_sample]
        labels = labels[idx_sample]

    pca = PCA(n_components=2, random_state=42)
    feats_2d = pca.fit_transform(features)


    plt.figure(figsize=(10, 7))
    colors = {"Corrected": "#f59e0b", "Normal": "#10b981", "Reversal": "#ef4444"}

    for idx, c_name in enumerate(CLASSES):
        mask = labels == idx
        plt.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            c=colors[c_name],
            label=f"{c_name} ({mask.sum()} samples)",
            alpha=0.6,
            edgecolors="none",
            s=25
        )

    plt.title(
        f"2D PCA Projection of CNN Feature Embeddings\n"
        f"Explained Variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%",
        fontsize=13,
        weight="bold",
        pad=12
    )
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] PCA feature projection plot saved to: {save_path}")


def run_comprehensive_evaluation(
    cnn_model,
    svm_model,
    rf_model,
    scaler,
    X_test_imgs: np.ndarray,
    X_test_feats: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates all individual models and the hybrid ensemble,
    saves all figures, CSV table, and JSON metrics.
    """
    print("\n" + "="*70)
    print("      RUNNING COMPREHENSIVE HYBRID MODEL EVALUATION")
    print("="*70)

    # 1. Standalone CNN
    print("\n[1/4] Evaluating Standalone CNN...")
    if hasattr(X_test_imgs, "take") or str(type(X_test_imgs)).find("Dataset") != -1:
        cnn_probs = cnn_model.predict(X_test_imgs, verbose=0)
    else:
        cnn_probs = cnn_model.predict(X_test_imgs, batch_size=256, verbose=0)
    cnn_preds = np.argmax(cnn_probs, axis=1)
    cnn_metrics = compute_metrics(y_test, cnn_preds, cnn_probs)


    # 2. CNN + SVM
    print("[2/4] Evaluating CNN + SVM (RBF)...")
    X_test_feats_scaled = scaler.transform(X_test_feats)
    svm_probs = svm_model.predict_proba(X_test_feats_scaled)
    svm_preds = svm_model.predict(X_test_feats_scaled)
    svm_metrics = compute_metrics(y_test, svm_preds, svm_probs)

    # 3. CNN + Random Forest
    print("[3/4] Evaluating CNN + Random Forest...")
    rf_probs = rf_model.predict_proba(X_test_feats)
    rf_preds = rf_model.predict(X_test_feats)
    rf_metrics = compute_metrics(y_test, rf_preds, rf_probs)

    # 4. Hybrid Consensus Ensemble (Soft Voting)
    print("[4/4] Evaluating Hybrid Consensus Ensemble...")
    hybrid_probs = (0.35 * cnn_probs) + (0.35 * svm_probs) + (0.30 * rf_probs)
    hybrid_preds = np.argmax(hybrid_probs, axis=1)
    hybrid_metrics = compute_metrics(y_test, hybrid_preds, hybrid_probs)

    results = {
        "Standalone CNN": {"metrics": cnn_metrics, "predictions": cnn_preds, "probabilities": cnn_probs},
        "CNN + SVM (RBF)": {"metrics": svm_metrics, "predictions": svm_preds, "probabilities": svm_probs},
        "CNN + Random Forest": {"metrics": rf_metrics, "predictions": rf_preds, "probabilities": rf_probs},
        "Hybrid Ensemble (Consensus)": {"metrics": hybrid_metrics, "predictions": hybrid_preds, "probabilities": hybrid_probs}
    }

    # Build Summary Table
    rows = []
    for m_name, data in results.items():
        m = data["metrics"]
        rows.append({
            "Model": m_name,
            "Accuracy": round(m["accuracy"] * 100, 2),
            "Macro Precision": round(m["macro_precision"] * 100, 2),
            "Macro Recall": round(m["macro_recall"] * 100, 2),
            "Macro F1": round(m["macro_f1"] * 100, 2),
            "Weighted F1": round(m["weighted_f1"] * 100, 2)
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("                     BENCHMARK SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Save CSV and JSON
    csv_path = RESULTS_DIR / "model_comparison_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Benchmark summary CSV saved to: {csv_path}")

    metrics_json_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_json_path, "w") as f:
        serializable_results = {
            m_name: {
                "accuracy": data["metrics"]["accuracy"],
                "macro_precision": data["metrics"]["macro_precision"],
                "macro_recall": data["metrics"]["macro_recall"],
                "macro_f1": data["metrics"]["macro_f1"],
                "weighted_f1": data["metrics"]["weighted_f1"],
                "per_class": data["metrics"]["per_class"],
                "confusion_matrix": data["metrics"]["confusion_matrix"],
                "roc_auc": data["metrics"].get("roc_auc", {})
            }
            for m_name, data in results.items()
        }
        json.dump(serializable_results, f, indent=4)
    print(f"[SAVED] Metrics JSON saved to: {metrics_json_path}")

    # Generate Plots
    cm_plot_path = RESULTS_DIR / "confusion_matrices.png"
    plot_confusion_matrices(results, y_test, cm_plot_path)

    bar_plot_path = RESULTS_DIR / "model_comparison.png"
    plot_model_comparison_bar(summary_df, bar_plot_path)

    roc_plot_path = RESULTS_DIR / "roc_curves.png"
    plot_roc_curves(results, y_test, roc_plot_path)

    pca_plot_path = RESULTS_DIR / "feature_tsne.png"
    plot_feature_pca(X_test_feats, y_test, pca_plot_path)

    return results
