"""
Grad-CAM explainability for the Hybrid Dyslexia Detection backbones.

Works for both:
- custom_cnn  (last conv layer "conv3", input 64x64x1)
- mobilenetv2 (ImageNet base with pooling="avg" -> no spatial map, so a
  spatial twin of the base is rebuilt weight-for-weight for attribution)

Standard Grad-CAM: gradient of the predicted class score w.r.t. the last
convolutional feature map, channel-wise global-average-pooled as weights,
weighted sum + ReLU + normalization + upsampling + heatmap overlay.
"""

from io import BytesIO
import base64
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def find_target_conv_name(model) -> Tuple[str, bool]:
    """
    Returns (target_conv_name, is_nested_mobilenet_base).
    - custom CNN -> ("conv3", False)
    - MobileNetV2 -> (last Conv2D inside "mobilenetv2_base", True)
    """
    try:
        model.get_layer("conv3")
        return "conv3", False
    except (ValueError, AttributeError):
        pass
    # MobileNetV2 nested base: search inner layers for last Conv2D.
    try:
        inner = model.get_layer("mobilenetv2_base")
        inner_layers = getattr(inner, "layers", [])
        for layer in reversed(inner_layers):
            if layer.__class__.__name__ == "Conv2D":
                return layer.name, True
        # Fallback to well-known MobileNetV2 last conv names.
        for fallback in ("out_relu", "Conv_1", "block_16_expand_relu"):
            try:
                inner.get_layer(fallback)
                return fallback, True
            except (ValueError, AttributeError):
                continue
    except (ValueError, AttributeError):
        pass
    # Last resort: last Conv2D among top-level layers.
    for layer in reversed(getattr(model, "layers", [])):
        if layer.__class__.__name__ == "Conv2D":
            return layer.name, False
    raise ValueError("No convolutional layer found for Grad-CAM.")


def _build_spatial_mobilenet_classifier(trained_model):
    """
    Rebuild an inference-only spatial twin of a trained MobileNetV2 model.

    The trained model uses pooling="avg" (1280-d vector, no spatial map).
    This rebuilds pooling=None (H'xWx1280) with copied weights, then reuses
    the trained feature_dense + classification_head weights, so attribution
    reflects the actual trained classifier. Augmentation layers are identity
    at inference and are skipped.
    Returns (grad_model, target_conv_name) where grad_model outputs
    [conv_feature_map, class_probabilities].
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    input_shape = tuple(trained_model.input_shape[1:])
    inner = trained_model.get_layer("mobilenetv2_base")
    alpha = getattr(inner, "alpha", 1.0) if hasattr(inner, "alpha") else 1.0

    spatial_base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        alpha=alpha,
        pooling=None,
        name="mobilenetv2_spatial_base",
    )
    spatial_base.set_weights(inner.get_weights())

    target_conv_name, _ = find_target_conv_name(trained_model)
    try:
        conv_output = spatial_base.get_layer(target_conv_name).output
    except (ValueError, AttributeError):
        # Fallback: last Conv2D in the spatial base.
        conv_output = None
        for layer in reversed(spatial_base.layers):
            if layer.__class__.__name__ == "Conv2D":
                conv_output = layer.output
                target_conv_name = layer.name
                break
        if conv_output is None:
            raise ValueError("No Conv2D found in spatial MobileNetV2 base.")

    rescale_layer = trained_model.get_layer("mobilenet_preprocess")
    feature_dense = trained_model.get_layer("feature_dense")
    head = trained_model.get_layer("classification_head")

    inputs = tf.keras.Input(shape=input_shape, name="gradcam_input")
    # Reuse trained rescale config explicitly (scale/offset are constructor args).
    # (Augmentation layers are identity at inference and are skipped.)
    x = layers.Rescaling(
        scale=rescale_layer.scale, offset=rescale_layer.offset, name="gradcam_rescale"
    )(inputs)
    # Tap the target conv layer along the NEW input path (the layer object is
    # shared, so weights are identical to the trained model).
    tap_model = tf.keras.Model(
        inputs=spatial_base.inputs,
        outputs=spatial_base.get_layer(target_conv_name).output,
        name="gradcam_tap",
    )
    tap = tap_model(x)
    fmap = spatial_base(x, training=False)
    gap = layers.GlobalAveragePooling2D(name="gradcam_gap")(fmap)
    feats = layers.Dense.from_config(feature_dense.get_config())(gap)
    # Copy trained dense weights onto the rebuilt layers.
    # (from_config creates new layers; set weights after graph construction
    # via a second pass below using a full model.)
    x_head = layers.Dropout(rate=0.0, name="gradcam_dropout_passthrough")(feats)
    probs = layers.Dense.from_config(head.get_config(), name="gradcam_head")(x_head)

    full = tf.keras.Model(inputs=inputs, outputs=probs, name="MobilenetV2_GradCAM_Full")
    # Copy weights: GAP has none; dense/head/dropout passthrough need mapping.
    for new_layer, old_layer in zip(
        [l for l in full.layers if l.name in ("gradcam_head",)],
        [head],
    ):
        new_layer.set_weights(old_layer.get_weights())
    # Feature dense layer in `full` is unnamed from from_config; find by order.
    for new_layer in full.layers:
        if isinstance(new_layer, layers.Dense) and new_layer.name != "gradcam_head":
            try:
                new_layer.set_weights(feature_dense.get_weights())
                break
            except ValueError:
                continue

    grad_model = tf.keras.Model(
        inputs=full.inputs, outputs=[tap, full.outputs[0]], name="MobilenetV2_GradCAM"
    )
    return grad_model, target_conv_name


def _build_custom_cnn_grad_model(model):
    """Grad model for the custom 3-block CNN: [conv3_output, probs]."""
    import tensorflow as tf

    target, _ = find_target_conv_name(model)
    conv_output = model.get_layer(target).output
    return tf.keras.Model(
        inputs=model.inputs, outputs=[conv_output, model.outputs[0]], name="CNN_GradCAM"
    ), target


def compute_heatmap(
    model,
    input_tensor: np.ndarray,
    class_index: Optional[int] = None,
) -> Tuple[np.ndarray, int, str]:
    """
    Compute a normalized Grad-CAM heatmap in [0, 1] at feature-map resolution
    upsampled to the model input size. Returns (heatmap_hw, class_idx, conv_name).
    """
    import tensorflow as tf

    target_name, nested = find_target_conv_name(model)
    if nested:
        grad_model, target_name = _build_spatial_mobilenet_classifier(model)
    else:
        grad_model, target_name = _build_custom_cnn_grad_model(model)

    x = tf.cast(tf.convert_to_tensor(input_tensor), tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]).numpy())
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient is None; check model graph.")
    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)  # (1,H,W)
    cam = tf.nn.relu(cam)
    cam_max = tf.reduce_max(cam)
    cam = cam / (cam_max + 1e-8)
    heatmap = cam[0].numpy().astype(np.float32)

    # Upsample to input resolution.
    in_h, in_w = int(input_tensor.shape[1]), int(input_tensor.shape[2])
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (in_w, in_h), Image.Resampling.BILINEAR
    )
    return np.array(heatmap_img, dtype=np.float32) / 255.0, int(class_index), str(target_name)


def overlay_on_grayscale(
    gray_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a [0,1] heatmap (jet colormap) over a grayscale PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = gray_pil.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(colored).resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base, heat_pil, alpha)


def overlay_to_data_uri(overlay_pil: Image.Image, size=(160, 160)) -> str:
    """Encode overlay PIL image as a base64 data URI thumbnail."""
    thumb = overlay_pil.copy().convert("RGB")
    thumb.thumbnail(size)
    buf = BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
