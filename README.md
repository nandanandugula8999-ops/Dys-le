# Hybrid AI-Based Dyslexia Detection System Using Handwriting Images

An automated, non-invasive, and low-cost early dyslexia screening system that analyzes handwriting patterns. This system integrates deep feature extraction with ensemble machine learning to classify handwriting characters into **Normal**, **Reversal** (dyslexic mirror/rotation patterns), and **Corrected** (overwriting/retracing compensatory strokes).

---

## 🌟 Key Highlights & Methodology

1. **The Screening Challenge**:
   Traditional psycho-educational evaluations for developmental dyslexia are time-consuming, expensive, and often administered late in a child's academic development. Persistent letter/numeral reversal (e.g., *b/d*, *p/q*, *s/z*) and erratic stroke overwriting are early behavioral biomarkers of dyslexia.

2. **Full-Scale Hybrid AI Architecture (Trained on ALL 208,372 Images)**:
   - **Deep CNN Backbone**: A 3-block convolutional neural network trained across all 151,649 training images (136k train / 15k validation), extracting **128-dimensional dense visual representations**.
   - **Support Vector Machine (SVM)**: Uses an RBF kernel to maximize decision boundary margins in the 128-D latent space.
   - **Random Forest (RF)**: An ensemble of 150 decorrelated decision trees capturing complex non-linear feature interactions.
   - **Consensus Hybrid Ensemble**: Soft-voting aggregation (35% CNN + 35% SVM + 30% RF) providing calibrated risk probabilities and clinical flags.

---

## 📊 Full-Scale Benchmark Results (ALL 56,723 Unseen Test Images)

| Model Architecture | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standalone CNN** | 85.03% | 85.20% | 85.23% | 85.07% | 84.99% |
| **CNN + SVM (RBF)** | 88.22% | 88.40% | 88.24% | 88.30% | 88.23% |
| **CNN + Random Forest** | 88.52% | 88.74% | 88.55% | 88.63% | 88.56% |
| **🏆 Hybrid Consensus Ensemble** | **88.67%** | **88.79%** | **88.73%** | **88.76%** | **88.69%** |

*All evaluation artifacts are saved in [`results/`](file:///c:/Users/dprin/OneDrive/Desktop/Dyslexia_Project/results):*
- `results/confusion_matrices.png`: 4-panel comparison of confusion matrices across all 56,723 test images.
- `results/model_comparison.png`: Grouped performance metrics bar chart.
- `results/roc_curves.png`: Multiclass One-vs-Rest ROC curves with AUC values.
- `results/feature_tsne.png`: 2D PCA projection of the 128-D CNN feature space showing class separability.

---

## 🗂️ Project Repository Structure

```
Dyslexia_Project/
├── app/
│   ├── app.py                      # Full-featured Streamlit Screening Dashboard
│   ├── utils.py                    # Demo samples loader & clinical report generator
│   └── __init__.py                 # Application package marker
├── dataset/
│   └── extracted/                  # Handwriting dataset (Train: ~151k, Test: ~56k)
├── models/
│   ├── cnn_feature_extractor.keras # Full-scale trained CNN model & feature extractor
│   ├── svm_classifier.joblib       # Trained Support Vector Machine (RBF)
│   ├── rf_classifier.joblib        # Trained Random Forest Classifier (150 trees)
│   ├── feature_scaler.joblib       # Fitted StandardScaler for 128-D embeddings
│   └── model_metadata.json         # Pipeline metadata and hyperparameters
├── notebooks/
│   ├── 01_hybrid_dyslexia_detection.ipynb # End-to-end tutorial, EDA & inference demo
│   └── Untitled.ipynb              # Initial scratchpad notebook
├── results/
│   ├── confusion_matrices.png      # Confusion matrices across all models
│   ├── model_comparison.png        # Bar chart comparing Accuracy, Precision, Recall, F1
│   ├── roc_curves.png              # Multiclass ROC curves
│   ├── feature_tsne.png            # 2D PCA latent representation plot
│   ├── model_comparison_table.csv  # Tabular benchmark metrics
│   └── evaluation_metrics.json     # Detailed per-class precision, recall, and AUC
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central configuration & paths
│   ├── data_loader.py              # Full dataset streaming & balanced loaders
│   ├── cnn_model.py                # CNN architecture & training routines
│   ├── feature_extractor.py        # 128-D latent representation extractor
│   ├── train_ml_classifiers.py     # SVM & Random Forest training & persistence
│   ├── evaluate.py                 # Evaluation metrics & visualization generators
│   └── hybrid_pipeline.py          # Unified hybrid inference engine & batch averaging
├── static/                         # Static web assets
│   ├── css/style.css               # Dedicated CSS styling (cards, animations, badges)
│   ├── js/main.js                  # Frontend JS (drag-drop, previews, API fetch, reports)
│   └── images/                     # Web charts and confusion matrix graphics
├── templates/                      # HTML templates
│   └── index.html                  # Semantic HTML5 frontend interface
├── server.py                       # Flask REST server connecting HTML to Hybrid AI
├── run_html_webapp.bat             # 1-click launcher for HTML+CSS+JS web application
├── run_streamlit_app.bat          # 1-click launcher for Streamlit dashboard
├── run_pipeline.py                 # Master full-scale pipeline script
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete documentation
```

---

## 🚀 Quickstart Guide

### 1. Option A: Traditional HTML5 + CSS3 Web Portal (Recommended)
Dedicated HTML, modern CSS styling, and Vanilla JavaScript frontend served via Flask:
```powershell
# Quick double-click or run:
run_html_webapp.bat
# Or manually in terminal:
& "C:\Users\dprin\anaconda3\python.exe" server.py
```
Open **`http://localhost:5000`** in your browser.

### 2. Option B: Streamlit Clinical Dashboard
Interactive Streamlit screening dashboard:
```powershell
# Quick double-click or run:
run_streamlit_app.bat
# Or manually in terminal:
& "C:\Users\dprin\anaconda3\python.exe" -m streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Run the Full End-to-End Pipeline
```powershell
& "C:\Users\dprin\anaconda3\python.exe" run_pipeline.py
```

---

## 🏥 Clinical Risk Mapping

| Class | Motor-Perceptual Marker | Clinical Risk Level | Recommended Next Step |
| :--- | :--- | :--- | :--- |
| **Normal** | Fluid strokes, standard directionality | **Low Risk** | Standard developmental tracking. |
| **Corrected** | Retraced strokes, multi-layer overwriting | **Moderate Risk** | Educational monitoring for motor hesitation / dysgraphia. |
| **Reversal** | Inverted/mirrored characters (*b/d*, *p/q*) | **High Risk** | Formal psycho-educational dyslexia screening recommended. |
