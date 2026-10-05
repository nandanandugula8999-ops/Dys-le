"""
Generate clean interactive tutorial and evaluation notebook.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Hybrid AI-Based Dyslexia Detection System Using Handwriting Images\n",
                "\n",
                "## Project Overview\n",
                "Early identification of dyslexia is critical for effective pedagogical intervention. "
                "Children and adults with dyslexia frequently display distinctive handwriting anomalies, including:\n",
                "- **Reversal (Mirror Writing)**: Inverted or mirrored letterforms (e.g., *b/d*, *p/q*, *s/z*, mirrored numerals).\n",
                "- **Corrected (Overwriting)**: Retraced strokes and multi-layer modifications reflecting uncertainty or motor hesitation.\n",
                "- **Normal**: Fluid stroke trajectories and standard directional orientations.\n",
                "\n",
                "### Hybrid Architecture\n",
                "- **CNN Backbone**: Extracts 128-dimensional dense visual features from 64x64 handwriting images.\n",
                "- **Support Vector Machine (SVM)**: RBF-kernel maximum-margin decision boundary on scaled embeddings.\n",
                "- **Random Forest (RF)**: 150 non-linear decision trees with bagging ensemble aggregation.\n",
                "- **Consensus Ensemble**: Soft-voting aggregation delivering high-confidence risk screening."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from PIL import Image\n",
                "\n",
                "# Add project root to sys.path\n",
                "PROJECT_ROOT = Path('..').resolve()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "from src.config import CLASSES, CLASS_RISK_MAP, RESULTS_DIR, MODELS_DIR\n",
                "from src.hybrid_pipeline import HybridDyslexiaDetector\n",
                "from app.utils import get_demo_samples\n",
                "print('Environment initialized successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Visualizing Benchmark Handwriting Patterns\n",
                "Inspect representative samples of Corrected, Normal, and Reversal handwriting."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "demo_samples = get_demo_samples(num_per_class=3)\n",
                "fig, axes = plt.subplots(3, 3, figsize=(10, 10))\n",
                "\n",
                "for row_idx, cls_name in enumerate(CLASSES):\n",
                "    samples = demo_samples.get(cls_name, [])\n",
                "    for col_idx in range(3):\n",
                "        ax = axes[row_idx, col_idx]\n",
                "        if col_idx < len(samples):\n",
                "            img_path, img_name = samples[col_idx]\n",
                "            im = Image.open(img_path)\n",
                "            ax.imshow(im, cmap='gray')\n",
                "            ax.set_title(f'{cls_name}\\n({img_name})', fontsize=10, weight='bold')\n",
                "        ax.axis('off')\n",
                "\n",
                "plt.suptitle('Representative Handwriting Categories', fontsize=14, weight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Comparative Benchmark Performance\n",
                "Summary table of accuracy, precision, recall, and F1 scores across individual models and the hybrid ensemble."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "csv_path = RESULTS_DIR / 'model_comparison_table.csv'\n",
                "if csv_path.exists():\n",
                "    df = pd.read_csv(csv_path)\n",
                "    display(df)\n",
                "else:\n",
                "    print('Results table not found. Please verify run_pipeline.py has completed.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Evaluation Artifacts and Visual Insights\n",
                "Load and display confusion matrices, ROC curves, and PCA feature separability."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plots = ['confusion_matrices.png', 'model_comparison.png', 'roc_curves.png', 'feature_tsne.png']\n",
                "for p_name in plots:\n",
                "    p_path = RESULTS_DIR / p_name\n",
                "    if p_path.exists():\n",
                "        plt.figure(figsize=(11, 7))\n",
                "        plt.imshow(Image.open(p_path))\n",
                "        plt.axis('off')\n",
                "        plt.title(p_name, fontsize=12, weight='bold')\n",
                "        plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. End-to-End Hybrid Screening Inference"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "detector = HybridDyslexiaDetector()\n",
                "\n",
                "for cls_name in CLASSES:\n",
                "    sample_path = demo_samples[cls_name][0][0]\n",
                "    res = detector.predict(sample_path)\n",
                "    print('=' * 65)\n",
                "    print(f'Actual Class:        {cls_name}')\n",
                "    print(f'Consensus Decision:  {res[\"prediction\"]} ({res[\"confidence_percentage\"]})')\n",
                "    print(f'Risk Level:          {res[\"risk_level\"]}')\n",
                "    print(f'Model Agreement:     {\"Unanimous\" if res[\"unanimous_agreement\"] else \"Majority\"}')\n",
                "    print(f'CNN: {res[\"models\"][\"CNN\"][\"prediction\"]} ({res[\"models\"][\"CNN\"][\"confidence\"]*100:.1f}%)')\n",
                "    print(f'SVM: {res[\"models\"][\"SVM\"][\"prediction\"]} ({res[\"models\"][\"SVM\"][\"confidence\"]*100:.1f}%)')\n",
                "    print(f'RF:  {res[\"models\"][\"Random_Forest\"][\"prediction\"]} ({res[\"models\"][\"Random_Forest\"][\"confidence\"]*100:.1f}%)')\n"
            ]
        },

        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Multi-Image Batch Screening with Individual Result Averaging\n",
                "Screening multiple handwriting characters at once and computing the aggregated session average."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Gather a collection of handwriting images\n",
                "multi_paths = []\n",
                "multi_names = []\n",
                "for c in CLASSES:\n",
                "    for p, n in demo_samples[c][:2]:\n",
                "        multi_paths.append(p)\n",
                "        multi_names.append(f'{c}_{n}')\n",
                "\n",
                "batch_result = detector.predict_batch(multi_paths, multi_names)\n",
                "\n",
                "print('=' * 75)\n",
                "print(f'Total Samples Evaluated:     {batch_result[\"total_images\"]}')\n",
                "print(f'Aggregated Session Decision: {batch_result[\"overall_prediction\"]} ({batch_result[\"overall_confidence_percentage\"]})')\n",
                "print(f'Session Risk Assessment:     {batch_result[\"overall_risk_level\"]}')\n",
                "print(f'Reversal Rate:               {batch_result[\"reversal_rate\"]:.1f}% ({batch_result[\"reversal_count\"]}/{batch_result[\"total_images\"]} samples)')\n",
                "print(f'Averaged Probabilities:      {batch_result[\"average_probabilities\"]}')\n",
                "print('=' * 75)\n",
                "\n",
                "# Print individual sample breakdown\n",
                "print('\\nIndividual Sample Results:')\n",
                "for ind in batch_result['individual_results']:\n",
                "    print(f'  {ind[\"filename\"]:25s} -> {ind[\"prediction\"]:10s} (Conf: {ind[\"confidence_percentage\"]})')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks") / "01_hybrid_dyslexia_detection.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook successfully written to {out_path}")
