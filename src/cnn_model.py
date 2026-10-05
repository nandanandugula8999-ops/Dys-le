"""
CNN Model architecture and training module for handwriting feature extraction and classification.
Extracts 128-dimensional latent representations for downstream ML classifiers.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from src.config import (
    INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, EPOCHS,
    CNN_MODEL_PATH, RANDOM_SEED, FEATURE_DIM,
    MOBILENET_INPUT_SHAPE, MOBILENET_ALPHA, MOBILENET_DROPOUT,
    MOBILENET_FREEZE_BASE, BACKBONE,
)


def build_cnn_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Builds a robust Convolutional Neural Network for handwriting feature extraction.
    Note: Horizontal flips are intentionally omitted because letter mirroring
    is the defining trait of dyslexia reversals.
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Mild data augmentation that preserves character orientation
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.04, fill_mode="nearest"),
        layers.RandomZoom(0.04, fill_mode="nearest"),
        layers.RandomTranslation(0.04, 0.04, fill_mode="nearest")
    ], name="augmentation_layer")

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Flatten & Dense Embedding
    x = layers.Flatten(name="flatten")(x)
    features = layers.Dense(128, activation="relu", name="feature_dense")(x)
    x = layers.Dropout(0.4, name="dropout")(features)

    # Output Classification Head
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Dyslexia_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = MOBILENET_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    feature_dim: int = FEATURE_DIM,
    alpha: float = MOBILENET_ALPHA,
    dropout_rate: float = MOBILENET_DROPOUT,
    freeze_base: bool = MOBILENET_FREEZE_BASE,
    fine_tune_from: Optional[int] = None,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """
    Builds a MobileNetV2 transfer-learning backbone for handwriting screening.

    - Expects RGB input (e.g. 96x96x3). Grayscale 64x64x1 images must be
      converted upstream (see src.data_loader.to_mobilenet_input /
      HybridDyslexiaDetector.preprocess_image).
    - Uses ImageNet weights with global-average pooling (1280-d), then a
      128-d `feature_dense` embedding so downstream SVM/RF code stays
      compatible with the custom CNN (same FEATURE_DIM).
    - Horizontal flips are intentionally NOT used in augmentation because
      letter mirroring is the dyslexia reversal signal.
    """
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    if len(input_shape) != 3 or input_shape[2] != 3:
        raise ValueError(f"MobileNetV2 requires 3-channel RGB input, got {input_shape}")
    if min(input_shape[0], input_shape[1]) < 32:
        raise ValueError(f"MobileNetV2 requires min 32x32 input, got {input_shape}")

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.04, fill_mode="nearest"),
        layers.RandomZoom(0.04, fill_mode="nearest"),
        layers.RandomTranslation(0.04, 0.04, fill_mode="nearest"),
    ], name="augmentation_layer")

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = data_augmentation(inputs)
    # MobileNetV2 preprocessing: RGB 0-255 -> [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        alpha=alpha,
        pooling="avg",
        name="mobilenetv2_base",
    )
    base.trainable = not freeze_base
    if fine_tune_from is not None:
        # Freeze all layers except the last `fine_tune_from` layers.
        for layer in base.layers[:-fine_tune_from]:
            layer.trainable = False
        for layer in base.layers[-fine_tune_from:]:
            layer.trainable = True

    x = base(x, training=False if freeze_base else True)
    features = layers.Dense(feature_dim, activation="relu", name="feature_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(features)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Dyslexia_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_backbone_model(
    backbone: str = BACKBONE,
    input_shape: Optional[Tuple[int, int, int]] = None,
    num_classes: int = NUM_CLASSES,
    **kwargs,
) -> tf.keras.Model:
    """Dispatcher: 'custom_cnn' (default) or 'mobilenetv2'."""
    b = (backbone or "custom_cnn").lower()
    if b == "mobilenetv2":
        return build_mobilenetv2_model(
            input_shape=input_shape or MOBILENET_INPUT_SHAPE,
            num_classes=num_classes,
            **kwargs,
        )
    return build_cnn_model(input_shape=input_shape or INPUT_SHAPE, num_classes=num_classes)


def train_cnn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = EPOCHS,
    model_save_path: Path = CNN_MODEL_PATH,
    class_weight: Optional[dict] = None,
    initial_model: Optional[tf.keras.Model] = None,
    learning_rate: float = LEARNING_RATE,
    backbone: str = BACKBONE,
) -> Tuple[tf.keras.Model, dict]:
    """
    Trains the backbone model with callbacks and saves the best checkpoint.
    Backward compatible: defaults to the custom CNN when `backbone` is unset
    and no `initial_model` is provided.
    """
    if initial_model is not None:
        model = initial_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    else:
        model = build_backbone_model(backbone=backbone)
    model.summary()

    model_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=model_callbacks,
        class_weight=class_weight
    )

    return model, history.history



def get_feature_extractor_model(cnn_model: tf.keras.Model) -> tf.keras.Model:
    """
    Extracts the sub-network up to 'feature_dense' (128-dimensional output).
    Works for both the custom CNN and MobileNetV2 backbones. Falls back to
    the Dense layer feeding the classification head if the name is missing
    (e.g. models loaded from older checkpoints).
    """
    try:
        target = cnn_model.get_layer("feature_dense").output
        name = "MobileNetV2_Feature_Extractor" if "mobilenet" in cnn_model.name.lower() else "CNN_Feature_Extractor"
        return tf.keras.Model(inputs=cnn_model.inputs, outputs=target, name=name)
    except (ValueError, AttributeError):
        pass
    # Fallback: find last Dense layer before the softmax head.
    dense_layers = [l for l in cnn_model.layers if isinstance(l, layers.Dense)]
    if len(dense_layers) >= 2:
        target = dense_layers[-2].output
    elif dense_layers:
        target = dense_layers[-1].output
    else:
        raise ValueError("No Dense feature layer found in model; cannot build extractor.")
    return tf.keras.Model(inputs=cnn_model.inputs, outputs=target, name="Feature_Extractor_Fallback")


def detect_backbone_from_model(model: tf.keras.Model) -> str:
    """Infer 'mobilenetv2' vs 'custom_cnn' from a loaded Keras model."""
    try:
        name = (getattr(model, "name", "") or "").lower()
        if "mobilenet" in name:
            return "mobilenetv2"
        for layer in model.layers:
            if "mobilenet" in getattr(layer, "name", "").lower():
                return "mobilenetv2"
            if getattr(layer, "name", "") == "mobilenetv2_base":
                return "mobilenetv2"
        shape = model.input_shape
        # Keras shape: (None, H, W, C)
        if isinstance(shape, (list, tuple)) and len(shape) == 4 and shape[-1] == 3:
            return "mobilenetv2"
    except Exception:
        pass
    return "custom_cnn"

