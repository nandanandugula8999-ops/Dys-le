"""
Unified Hybrid Dyslexia Detection Pipeline.
Combines CNN Deep Feature Extraction with SVM and Random Forest Classifiers
for consensus screening and risk assessment.
"""

from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

from src.config import (
    CNN_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH,
    CLASSES, CLASS_RISK_MAP, IMG_HEIGHT, IMG_WIDTH,
    MOBILENET_INPUT_SIZE, BACKBONE, backbone_model_paths,
)
from src.cnn_model import get_feature_extractor_model, detect_backbone_from_model


class HybridDyslexiaDetector:
    """
    End-to-end hybrid detector integrating CNN feature extraction,
    SVM classification, and Random Forest classification.
    """

    def __init__(
        self,
        cnn_path: Path = CNN_MODEL_PATH,
        svm_path: Path = SVM_MODEL_PATH,
        rf_path: Path = RF_MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        backbone: Optional[str] = None,
    ):
        self.cnn_path = Path(cnn_path)
        self.svm_path = Path(svm_path)
        self.rf_path = Path(rf_path)
        self.scaler_path = Path(scaler_path)
        # Explicit override (e.g. HybridDyslexiaDetector(backbone="mobilenetv2")),
        # otherwise auto-detected from the loaded Keras model.
        self.backbone_override = (backbone or BACKBONE or "custom_cnn").lower() if backbone or BACKBONE else None

        self._load_models()

    def _load_models(self):
        """Load all models and preprocessors into memory."""
        print(f"[INFO] Loading backbone CNN from {self.cnn_path}...")
        self.cnn_model = tf.keras.models.load_model(str(self.cnn_path))
        self.feature_extractor = get_feature_extractor_model(self.cnn_model)
        detected = detect_backbone_from_model(self.cnn_model)
        # Explicit backbone arg wins; otherwise trust auto-detection.
        if getattr(self, "backbone_override", None) in ("custom_cnn", "mobilenetv2"):
            # Only honor override when it agrees with detection OR model is custom.
            # Auto-detection is authoritative when a MobileNetV2 file is loaded.
            self.backbone_name = detected if detected == "mobilenetv2" else self.backbone_override
        else:
            self.backbone_name = detected
        print(f"[INFO] Detected backbone: {self.backbone_name}")

        print(f"[INFO] Loading SVM, RF, and Scaler...")
        self.svm_model = joblib.load(str(self.svm_path))
        self.rf_model = joblib.load(str(self.rf_path))
        self.scaler = joblib.load(str(self.scaler_path))
        print("[SUCCESS] All hybrid detector components loaded.")

    def _to_backbone_tensor(self, gray_tensor_01: np.ndarray) -> np.ndarray:
        """
        Convert normalized grayscale tensor (1,64,64,1) in [0,1] to the
        backbone's expected input. MobileNetV2: RGB (1,96,96,3) in [0,255]
        (model Rescaling -> [-1,1]). Custom CNN: unchanged.
        """
        if getattr(self, "backbone_name", "custom_cnn") != "mobilenetv2":
            return gray_tensor_01
        arr = gray_tensor_01.astype(np.float32).squeeze(axis=0).squeeze(axis=-1)  # (64,64) [0,1]
        pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        pil = pil.resize((MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE), Image.Resampling.BILINEAR)
        rgb = np.array(pil, dtype=np.float32)  # [0,255]
        rgb = np.stack([rgb, rgb, rgb], axis=-1)  # (S,S,3)
        return np.expand_dims(rgb, axis=0)

    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any], Image.Image]]:
        """
        Adaptive preprocessor for both native dataset images (white strokes on dark canvas)
        and external camera/scan images (dark ink on light paper).

        Automated Enhancements:
        1. Polarity Auto-Detection: Samples outer margins to check if background is light paper.
        2. Inversion & Background Subtraction: Normalizes background to 0 and strokes to positive.
        3. Dynamic Contrast Stretching & Shadow Filtering: Removes paper grain and lighting gradients.
        4. Stroke Bounding-Box Centering & Padding: Extracts glyph bounds and adds proportional
           margins matching standard 28x28 Gambo / MNIST handwriting distribution.
        5. Bilinear Resampling to (64, 64) and float32 scaling to [0, 1].
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("L")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("L")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[-1] in (3, 4):
                # RGB or RGBA numpy
                pil_img = Image.fromarray(image.astype(np.uint8)).convert("L")
            elif image.ndim == 2:
                pil_img = Image.fromarray(image.astype(np.uint8))
            else:
                arr = image.astype(np.float32)
                if arr.max() > 1.0:
                    arr /= 255.0
                if arr.ndim == 3 and arr.shape == (IMG_HEIGHT, IMG_WIDTH, 1):
                    tensor = np.expand_dims(arr, axis=0)
                elif arr.ndim == 4:
                    tensor = arr
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")
                tensor = self._to_backbone_tensor(tensor) if tensor.shape[-1] == 1 else tensor
                if return_details:
                    dummy_img = Image.fromarray((arr.squeeze() * 255).astype(np.uint8))
                    return tensor, {"auto_inverted": False, "centered": False, "background_detected": "preprocessed_array"}, dummy_img
                return tensor
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        arr = np.array(pil_img, dtype=np.float32)
        h, w = arr.shape

        # 1. Sample border pixels (outer 5% perimeter) to detect background illumination
        border_pixels = np.concatenate([
            arr[:max(2, int(h * 0.05)), :].flatten(),
            arr[-max(2, int(h * 0.05)):, :].flatten(),
            arr[:, :max(2, int(w * 0.05))].flatten(),
            arr[:, -max(2, int(w * 0.05)):].flatten()
        ])
        bg_val = float(np.median(border_pixels))
        was_inverted = False
        was_centered = False

        if bg_val > 110.0:
            # External camera/scan: dark ink on light/paper background
            # Invert and subtract background so background -> 0 and strokes -> positive
            arr = np.clip(bg_val - arr, 0.0, 255.0)
            if arr.max() > 0:
                arr = (arr / arr.max()) * 255.0

            # Suppress paper noise and lighting gradients below 25% max stroke intensity
            noise_thresh = 0.25 * arr.max()
            arr[arr < noise_thresh] = 0.0
            was_inverted = True

            # 2. Glyph Bounding Box Extraction & Centering
            non_zeros = np.argwhere(arr > 0)
            if len(non_zeros) >= 20:  # Valid stroke present
                ymin, xmin = non_zeros.min(axis=0)
                ymax, xmax = non_zeros.max(axis=0)
                cropped = arr[ymin:ymax + 1, xmin:xmax + 1]
                ch, cw = cropped.shape
                side = max(ch, cw)
                # Pad with 35% margin matching standard 28x28 Gambo / MNIST centering
                pad_size = int(side * 1.35)
                padded = np.zeros((pad_size, pad_size), dtype=np.float32)
                y_off = (pad_size - ch) // 2
                x_off = (pad_size - cw) // 2
                padded[y_off:y_off + ch, x_off:x_off + cw] = cropped
                arr = padded
                was_centered = True

        processed_pil = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8))
        resized_pil = processed_pil.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        norm_arr = np.array(resized_pil, dtype=np.float32) / 255.0
        tensor = np.expand_dims(norm_arr, axis=(0, -1))
        tensor = self._to_backbone_tensor(tensor)

        details = {
            "auto_inverted": was_inverted,
            "centered": was_centered,
            "background_detected": "light_paper" if was_inverted else "dark_canvas",
            "background_luminosity": round(bg_val, 1),
            "backbone": getattr(self, "backbone_name", "custom_cnn"),
        }

        if return_details:
            return tensor, details, resized_pil
        return tensor

    def preprocess_to_pil(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Helper to get the preprocessed normalized PIL Image."""
        _, _, pil_processed = self.preprocess_image(image, return_details=True)
        return pil_processed

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Execute full hybrid inference:
        1. Extract 128-d deep representation using CNN backbone
        2. Infer class probabilities from CNN classification head
        3. Infer class probabilities from SVM on scaled features
        4. Infer class probabilities from Random Forest on features
        5. Compute consensus hybrid ensemble decision and clinical risk score.
        """
        # Preprocess with adaptive polarity & centering
        x, prep_details, pil_processed = self.preprocess_image(image, return_details=True)

        # 1. CNN Predictions & Feature Extraction
        cnn_probs = self.cnn_model(x, training=False).numpy()[0]
        cnn_class_idx = int(np.argmax(cnn_probs))
        cnn_class = CLASSES[cnn_class_idx]

        # Extract 128-d embedding
        features = self.feature_extractor(x, training=False).numpy()  # shape (1, 128)

        # 2. SVM Predictions
        features_scaled = self.scaler.transform(features)
        svm_probs = self.svm_model.predict_proba(features_scaled)[0]
        svm_class_idx = int(np.argmax(svm_probs))
        svm_class = CLASSES[svm_class_idx]

        # 3. Random Forest Predictions
        rf_probs = self.rf_model.predict_proba(features)[0]
        rf_class_idx = int(np.argmax(rf_probs))
        rf_class = CLASSES[rf_class_idx]

        # 4. Hybrid Soft Voting Ensemble
        # Weights: CNN 35%, SVM 35%, RF 30%
        ensemble_probs = (0.35 * cnn_probs) + (0.35 * svm_probs) + (0.30 * rf_probs)
        ensemble_class_idx = int(np.argmax(ensemble_probs))
        ensemble_class = CLASSES[ensemble_class_idx]
        confidence = float(ensemble_probs[ensemble_class_idx])

        # Agreement analysis
        predictions_list = [cnn_class, svm_class, rf_class]
        unanimous = len(set(predictions_list)) == 1
        majority_votes = {c: predictions_list.count(c) for c in CLASSES}

        # Clinical Risk Mapping
        risk_info = CLASS_RISK_MAP[ensemble_class]

        return {
            "prediction": ensemble_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "backbone": getattr(self, "backbone_name", "custom_cnn"),
            "risk_level": risk_info["level"],
            "risk_description": risk_info["description"],
            "risk_color": risk_info["color"],
            "unanimous_agreement": unanimous,
            "vote_distribution": majority_votes,
            "ensemble_probabilities": {CLASSES[i]: float(ensemble_probs[i]) for i in range(len(CLASSES))},
            "preprocessing": prep_details,
            "models": {
                "CNN": {
                    "backbone": getattr(self, "backbone_name", "custom_cnn"),
                    "prediction": cnn_class,
                    "confidence": float(cnn_probs[cnn_class_idx]),
                    "probabilities": {CLASSES[i]: float(cnn_probs[i]) for i in range(len(CLASSES))}
                },
                "SVM": {
                    "prediction": svm_class,
                    "confidence": float(svm_probs[svm_class_idx]),
                    "probabilities": {CLASSES[i]: float(svm_probs[i]) for i in range(len(CLASSES))}
                },
                "Random_Forest": {
                    "prediction": rf_class,
                    "confidence": float(rf_probs[rf_class_idx]),
                    "probabilities": {CLASSES[i]: float(rf_probs[i]) for i in range(len(CLASSES))}
                }
            },
            "feature_vector_sample": features[0][:8].tolist()  # first 8 feature values for inspection
        }

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Accepts multiple handwriting images at once, computes individual hybrid predictions
        for each image, and calculates the aggregated result average across all images.

        Returns:
            Dict containing:
            - 'total_images': Number of images evaluated
            - 'overall_prediction': Winning class by average probability
            - 'overall_confidence': Average confidence score for the winning class
            - 'overall_confidence_percentage': Formatted string (e.g. '89.4%')
            - 'average_probabilities': Averaged probabilities for each class across all images
            - 'class_counts': Frequency count per class
            - 'reversal_count' & 'reversal_rate': Dyslexia reversal frequency and percentage
            - 'overall_risk_level': Clinical risk level (High / Moderate / Low)
            - 'overall_risk_description': Detailed diagnostic explanation
            - 'overall_risk_color': Status color code
            - 'individual_results': Complete per-image prediction breakdowns
            - 'model_averages': Average probabilities broken down by CNN, SVM, and Random Forest
        """
        if not images:
            raise ValueError("No images provided for batch prediction.")

        individual_results = []
        sum_ensemble_probs = np.zeros(len(CLASSES), dtype=np.float64)
        sum_cnn_probs = np.zeros(len(CLASSES), dtype=np.float64)
        sum_svm_probs = np.zeros(len(CLASSES), dtype=np.float64)
        sum_rf_probs = np.zeros(len(CLASSES), dtype=np.float64)

        for idx, img in enumerate(images):
            fname = filenames[idx] if filenames and idx < len(filenames) else f"Image_{idx + 1}"
            res = self.predict(img)
            res["filename"] = fname
            individual_results.append(res)

            # Accumulate probability vectors
            for c_i, c_name in enumerate(CLASSES):
                sum_ensemble_probs[c_i] += res["ensemble_probabilities"][c_name]
                sum_cnn_probs[c_i] += res["models"]["CNN"]["probabilities"][c_name]
                sum_svm_probs[c_i] += res["models"]["SVM"]["probabilities"][c_name]
                sum_rf_probs[c_i] += res["models"]["Random_Forest"]["probabilities"][c_name]

        n = len(images)
        avg_ensemble_probs = sum_ensemble_probs / n
        avg_cnn_probs = sum_cnn_probs / n
        avg_svm_probs = sum_svm_probs / n
        avg_rf_probs = sum_rf_probs / n

        overall_pred_idx = int(np.argmax(avg_ensemble_probs))
        overall_pred_class = CLASSES[overall_pred_idx]
        overall_confidence = float(avg_ensemble_probs[overall_pred_idx])

        # Frequency statistics
        pred_classes = [r["prediction"] for r in individual_results]
        class_counts = {c: pred_classes.count(c) for c in CLASSES}
        reversal_count = class_counts.get("Reversal", 0)
        corrected_count = class_counts.get("Corrected", 0)
        normal_count = class_counts.get("Normal", 0)

        reversal_rate = (reversal_count / n) * 100.0
        corrected_rate = (corrected_count / n) * 100.0
        normal_rate = (normal_count / n) * 100.0

        # Clinical Risk Assessment on multi-image session
        if reversal_rate >= 20.0 or overall_pred_class == "Reversal":
            risk_level = "High Risk (Dyslexia Indicator)"
            risk_color = "#dc3545"
            risk_description = (
                f"Significant letter reversal frequency ({reversal_count}/{n} images, {reversal_rate:.1f}%). "
                f"Persistent reversal patterns across multiple handwriting samples strongly correlate with motor-perceptual dyslexia biomarkers."
            )
        elif corrected_rate >= 30.0 or overall_pred_class == "Corrected":
            risk_level = "Moderate Risk"
            risk_color = "#ffc107"
            risk_description = (
                f"Frequent stroke alteration and overwriting detected ({corrected_count}/{n} images, {corrected_rate:.1f}%). "
                f"Indicates motor hesitation, dysgraphic tendencies, or compensatory retracing."
            )
        else:
            risk_level = "Low Risk"
            risk_color = "#28a745"
            risk_description = (
                f"Predominantly standard handwriting patterns ({normal_count}/{n} images, {normal_rate:.1f}%). "
                f"Stroke orientations and directional trajectories are consistent with normal developmental milestones."
            )

        return {
            "total_images": n,
            "overall_prediction": overall_pred_class,
            "overall_confidence": overall_confidence,
            "overall_confidence_percentage": f"{overall_confidence * 100:.1f}%",
            "average_probabilities": {CLASSES[i]: float(avg_ensemble_probs[i]) for i in range(len(CLASSES))},
            "class_counts": class_counts,
            "reversal_count": reversal_count,
            "reversal_rate": reversal_rate,
            "corrected_count": corrected_count,
            "corrected_rate": corrected_rate,
            "normal_count": normal_count,
            "normal_rate": normal_rate,
            "overall_risk_level": risk_level,
            "overall_risk_color": risk_color,
            "overall_risk_description": risk_description,
            "individual_results": individual_results,
            "model_averages": {
                "CNN": {CLASSES[i]: float(avg_cnn_probs[i]) for i in range(len(CLASSES))},
                "SVM": {CLASSES[i]: float(avg_svm_probs[i]) for i in range(len(CLASSES))},
                "Random_Forest": {CLASSES[i]: float(avg_rf_probs[i]) for i in range(len(CLASSES))}
            }
        }

