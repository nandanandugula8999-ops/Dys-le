"""
Feature extraction module for the Hybrid Dyslexia Detection system.
Extracts 128-dimensional deep feature embeddings from handwriting images using the CNN backbone.
"""

from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CNN_MODEL_PATH, IMG_HEIGHT, IMG_WIDTH
from src.cnn_model import get_feature_extractor_model


class DyslexiaFeatureExtractor:
    """
    CNN-based visual feature extractor for handwriting images.
    Outputs dense 128-dimensional feature vectors.
    """

    def __init__(self, model_or_path: Union[str, Path, tf.keras.Model] = CNN_MODEL_PATH):
        if isinstance(model_or_path, (str, Path)):
            self.cnn_model = tf.keras.models.load_model(str(model_or_path))
        elif hasattr(model_or_path, "predict") or isinstance(model_or_path, tf.keras.models.Model):
            self.cnn_model = model_or_path
        else:
            raise ValueError(f"Invalid model or path provided: {type(model_or_path)}")


        self.extractor_model = get_feature_extractor_model(self.cnn_model)

    def extract_from_array(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Extract features from a numpy array of shape (N, 64, 64, 1) with values in [0, 1].
        """
        features = self.extractor_model.predict(images, batch_size=batch_size, verbose=0)
        return features

    def extract_from_single_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract 128-d feature vector from a single image.
        Accepts: filepath, PIL Image, or preprocessed numpy array.
        Returns: 1D array of shape (128,)
        """
        if isinstance(image, str):
            img = Image.open(image).convert("L")
        elif isinstance(image, Image.Image):
            img = image.convert("L")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img_arr = image.astype(np.float32)
                if img_arr.max() > 1.0:
                    img_arr /= 255.0
                img_arr = np.expand_dims(img_arr, axis=(0, -1))
                return self.extractor_model(img_arr, training=False).numpy().flatten()
            elif image.ndim == 3 and image.shape[-1] == 1:
                img_arr = image.astype(np.float32)
                if img_arr.max() > 1.0:
                    img_arr /= 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                return self.extractor_model(img_arr, training=False).numpy().flatten()
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        else:
            raise TypeError("Unsupported image type")

        # Resize and normalize PIL image
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))

        feat = self.extractor_model(arr, training=False).numpy().flatten()
        return feat

    def extract_from_dataset(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from a tf.data.Dataset pipeline.
        """
        all_features = []
        all_labels = []

        for batch_x, batch_y in dataset:
            feats = self.extractor_model(batch_x, training=False)
            all_features.append(feats.numpy())
            all_labels.append(batch_y.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)
