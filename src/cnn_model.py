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
    CNN_MODEL_PATH, RANDOM_SEED
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


def train_cnn(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = EPOCHS,
    model_save_path: Path = CNN_MODEL_PATH,
    class_weight: Optional[dict] = None,
    initial_model: Optional[tf.keras.Model] = None,
    learning_rate: float = LEARNING_RATE
) -> Tuple[tf.keras.Model, dict]:
    """
    Trains the CNN model with callbacks and saves the best model checkpoint.
    """
    if initial_model is not None:
        model = initial_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    else:
        model = build_cnn_model()
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
    """
    extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.get_layer("feature_dense").output,
        name="CNN_Feature_Extractor"
    )
    return extractor

