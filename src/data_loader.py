"""
Data loading and preprocessing utilities for handwriting dyslexia images.
Handles balanced dataset creation, image augmentation, and pipeline generation.
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import (
    TRAIN_DIR, TEST_DIR, CLASSES, CLASS_TO_IDX,
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED,
    TRAIN_SAMPLES_PER_CLASS, VAL_SAMPLES_PER_CLASS, TEST_SAMPLES_PER_CLASS,
    MOBILENET_INPUT_SIZE,
)


def load_and_preprocess_image(image_path: str, target_size=(IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
    """
    Load a handwriting image, convert to grayscale, resize, and normalize to [0, 1].
    Returns shape: (64, 64, 1) float32.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr


def get_balanced_filepaths(
    data_dir: Path,
    samples_per_class: Optional[int] = None,
    seed: int = RANDOM_SEED
) -> Tuple[List[str], List[int]]:
    """
    Sample an equal number of images per class to prevent bias.
    Returns: (list_of_filepaths, list_of_labels)
    """
    random.seed(seed)
    all_filepaths = []
    all_labels = []

    for class_name in CLASSES:
        class_folder = data_dir / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Class folder not found: {class_folder}")

        files = [str(class_folder / f) for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        if samples_per_class is not None and samples_per_class < len(files):
            files = files[:samples_per_class]

        label = CLASS_TO_IDX[class_name]
        all_filepaths.extend(files)
        all_labels.extend([label] * len(files))

    # Shuffle paired lists together
    combined = list(zip(all_filepaths, all_labels))
    random.shuffle(combined)
    shuffled_paths, shuffled_labels = zip(*combined)

    return list(shuffled_paths), list(shuffled_labels)


def load_dataset_arrays(
    filepaths: List[str],
    labels: List[int],
    batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image files into memory as NumPy arrays.
    Suitable for fast feature extraction and ML classifier training.
    """
    n = len(filepaths)
    X = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    for i, path in enumerate(filepaths):
        X[i] = load_and_preprocess_image(path)

    return X, y


def create_tf_dataset(
    filepaths: List[str],
    labels: List[int],
    batch_size: int = BATCH_SIZE,
    is_training: bool = False
) -> tf.data.Dataset:
    """
    Create an optimized tf.data.Dataset pipeline.
    """
    def _parse_function(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=1)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    paths_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 5000), seed=RANDOM_SEED)

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_train_val_test_splits():
    """
    Prepares balanced Train, Validation, and Test splits according to config.
    """
    train_paths, train_labels = get_balanced_filepaths(TRAIN_DIR, samples_per_class=TRAIN_SAMPLES_PER_CLASS)
    val_paths, val_labels = get_balanced_filepaths(TRAIN_DIR, samples_per_class=VAL_SAMPLES_PER_CLASS, seed=RANDOM_SEED + 1)
    test_paths, test_labels = get_balanced_filepaths(TEST_DIR, samples_per_class=TEST_SAMPLES_PER_CLASS, seed=RANDOM_SEED + 2)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_full_train_and_val_datasets(batch_size: int = BATCH_SIZE, val_split: float = 0.10):
    """
    Streams all 151,649 images from the Train directory using tf.keras.utils.image_dataset_from_directory.
    Split: 90% training (~136,484 images), 10% validation (~15,165 images).
    """
    train_full = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=val_split,
        subset="training",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True
    )
    val_full = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=val_split,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False
    )
    normalize = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    train_ds = train_full.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_full.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def get_full_test_dataset(batch_size: int = BATCH_SIZE):
    """
    Streams all 56,723 test images from the Test directory without shuffling.
    """
    test_full = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False
    )
    normalize = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    test_ds = test_full.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return test_ds


# ---------------------------------------------------------------------------
# MobileNetV2 backend helpers (RGB, 96x96, ImageNet preprocessing)
# ---------------------------------------------------------------------------

def to_mobilenet_input(gray_batch: tf.Tensor, size: int = MOBILENET_INPUT_SIZE) -> tf.Tensor:
    """
    Convert a grayscale batch (N,H,W,1) in [0,1] to MobileNetV2 RGB input
    (N,size,size,3) in [0,255] float32. The model's internal Rescaling layer
    then maps it to [-1, 1], matching tf.keras.applications.mobilenet_v2.
    """
    x = tf.cast(gray_batch, tf.float32)
    if x.shape.rank == 4 and x.shape[-1] == 1:
        x = tf.image.grayscale_to_rgb(x)
    elif x.shape.rank == 3:
        x = tf.image.grayscale_to_rgb(tf.expand_dims(x, -1))
    x = tf.image.resize(x, [size, size])
    return x * 255.0


def adapt_dataset_for_mobilenet(dataset: tf.data.Dataset, size: int = MOBILENET_INPUT_SIZE) -> tf.data.Dataset:
    """Map a normalized grayscale dataset to MobileNetV2 RGB input on the fly."""
    return dataset.map(
        lambda x, y: (to_mobilenet_input(x, size=size), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def load_and_preprocess_image_mobilenet(image_path: str, size: int = MOBILENET_INPUT_SIZE) -> np.ndarray:
    """
    Load a handwriting image for MobileNetV2: grayscale -> RGB, resize,
    scale to [0,255]. Returns shape (size, size, 3) float32.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)  # [0, 255]
    rgb = np.stack([arr, arr, arr], axis=-1)
    return rgb

