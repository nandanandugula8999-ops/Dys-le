"""
Utility functions for the Streamlit Dyslexia Screening Web Application.
Provides demo sample loaders, plotting helpers, and clinical report formatters.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.config import TEST_DIR, CLASSES, CLASS_RISK_MAP


def get_demo_samples(num_per_class: int = 3) -> Dict[str, List[Tuple[str, str]]]:
    """
    Retrieves representative test images for quick demo screening.
    Returns: {
        'Corrected': [(filepath, filename), ...],
        'Normal': [(filepath, filename), ...],
        'Reversal': [(filepath, filename), ...]
    }
    """
    demo_dict = {}
    if not TEST_DIR.exists():
        return demo_dict

    for class_name in CLASSES:
        cls_dir = TEST_DIR / class_name
        if cls_dir.exists():
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg'))]
            # Select evenly spaced samples
            step = max(1, len(files) // (num_per_class + 1))
            selected = []
            for i in range(num_per_class):
                idx = (i + 1) * step
                if idx < len(files):
                    selected.append((str(cls_dir / files[idx]), files[idx]))
            demo_dict[class_name] = selected
    return demo_dict


def generate_clinical_report(result: Dict, patient_id: str = "PATIENT-DEMO-001", notes: str = "") -> str:
    """
    Generates a formal, printable clinical screening summary.
    """
    prediction = result["prediction"]
    confidence = result["confidence_percentage"]
    risk = result["risk_level"]
    description = result["risk_description"]
    unanimous = "Yes (All 3 models agree)" if result["unanimous_agreement"] else "Partial (Soft-voting consensus)"

    report = f"""================================================================================
          AI-ASSISTED DYSLEXIA HANDWRITING SCREENING REPORT
================================================================================
Screening Identifier : {patient_id}
Consensus Decision   : {prediction.upper()}
Confidence Score     : {confidence}
Clinical Risk Level  : {risk}
Model Agreement      : {unanimous}

CLINICAL INTERPRETATION:
{description}

MODEL PREDICTION BREAKDOWN:
- Convolutional Neural Network (CNN):
    Prediction: {result['models']['CNN']['prediction']}
    Confidence: {result['models']['CNN']['confidence']*100:.1f}%

- Support Vector Machine (SVM - RBF):
    Prediction: {result['models']['SVM']['prediction']}
    Confidence: {result['models']['SVM']['confidence']*100:.1f}%

- Random Forest (150 Ensemble Trees):
    Prediction: {result['models']['Random_Forest']['prediction']}
    Confidence: {result['models']['Random_Forest']['confidence']*100:.1f}%

CONSENSUS PROBABILITY DISTRIBUTION:
- Corrected : {result['ensemble_probabilities']['Corrected']*100:.2f}%
- Normal    : {result['ensemble_probabilities']['Normal']*100:.2f}%
- Reversal  : {result['ensemble_probabilities']['Reversal']*100:.2f}%

EXAMINER / CLINICIAN NOTES:
{notes if notes.strip() else "None provided."}

RECOMMENDATION:
{"Formal psycho-educational diagnostic assessment recommended for persistent letter reversal patterns." if prediction == "Reversal" else "Continued developmental monitoring recommended." if prediction == "Corrected" else "Standard developmental progression; no immediate clinical intervention indicated."}

================================================================================
DISCLAIMER: This screening tool is an automated assistive diagnostic aid based 
on handwriting pattern analysis and does not replace formal clinical assessment.
================================================================================
"""
    return report


def generate_batch_clinical_report(batch_result: Dict, session_id: str = "SESSION-MULTI-001", notes: str = "") -> str:
    """
    Generates a formal clinical report summarizing multi-image averaged screening results.
    """
    overall_pred = batch_result["overall_prediction"]
    overall_conf = batch_result["overall_confidence_percentage"]
    risk = batch_result["overall_risk_level"]
    description = batch_result["overall_risk_description"]
    total = batch_result["total_images"]
    rev_count = batch_result["reversal_count"]
    rev_rate = batch_result["reversal_rate"]
    corr_count = batch_result["corrected_count"]
    norm_count = batch_result["normal_count"]

    individual_lines = []
    for idx, r in enumerate(batch_result["individual_results"]):
        fname = r.get("filename", f"Image_{idx+1}")
        pred = r["prediction"]
        conf = r["confidence_percentage"]
        unanimous = "Yes" if r["unanimous_agreement"] else "No"
        individual_lines.append(f"  {idx+1:2d}. {fname:<25s} -> {pred:<12s} ({conf}) [Unanimous: {unanimous}]")

    table_str = "\n".join(individual_lines)

    report = f"""================================================================================
     MULTI-IMAGE AI DYSLEXIA SCREENING SESSION REPORT (AVERAGED RESULTS)
================================================================================
Session Identifier   : {session_id}
Total Images Evaluated: {total}
Aggregated Decision  : {overall_pred.upper()}
Average Confidence   : {overall_conf}
Session Risk Level   : {risk}
Reversal Frequency   : {rev_count} / {total} samples ({rev_rate:.1f}%)
Corrected Frequency  : {corr_count} / {total} samples ({batch_result['corrected_rate']:.1f}%)
Normal Frequency     : {norm_count} / {total} samples ({batch_result['normal_rate']:.1f}%)

SESSION CLINICAL INTERPRETATION:
{description}

AVERAGED PROBABILITY DISTRIBUTION (ACROSS ALL {total} SAMPLES):
- Corrected : {batch_result['average_probabilities']['Corrected']*100:.2f}%
- Normal    : {batch_result['average_probabilities']['Normal']*100:.2f}%
- Reversal  : {batch_result['average_probabilities']['Reversal']*100:.2f}%

MODEL-WISE AVERAGES:
- CNN Average Probs  : Corrected: {batch_result['model_averages']['CNN']['Corrected']*100:.1f}%, Normal: {batch_result['model_averages']['CNN']['Normal']*100:.1f}%, Reversal: {batch_result['model_averages']['CNN']['Reversal']*100:.1f}%
- SVM Average Probs  : Corrected: {batch_result['model_averages']['SVM']['Corrected']*100:.1f}%, Normal: {batch_result['model_averages']['SVM']['Normal']*100:.1f}%, Reversal: {batch_result['model_averages']['SVM']['Reversal']*100:.1f}%
- RF Average Probs   : Corrected: {batch_result['model_averages']['Random_Forest']['Corrected']*100:.1f}%, Normal: {batch_result['model_averages']['Random_Forest']['Normal']*100:.1f}%, Reversal: {batch_result['model_averages']['Random_Forest']['Reversal']*100:.1f}%

INDIVIDUAL SAMPLE BREAKDOWN:
{table_str}

EXAMINER / CLINICIAN NOTES:
{notes if notes.strip() else "Multi-character handwriting assessment session."}

RECOMMENDATION:
{"High priority recommendation for psycho-educational dyslexia evaluation due to repeated letter reversals." if rev_rate >= 20.0 or overall_pred == "Reversal" else "Recommend monitoring fine-motor handwriting coordination and letter formation." if corr_count > 0 else "Normal developmental handwriting progression observed across all samples."}

================================================================================
DISCLAIMER: This screening tool provides an automated statistical summary of 
handwriting characteristics and does not substitute for clinical diagnosis.
================================================================================
"""
    return report

