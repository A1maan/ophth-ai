"""
Grad-CAM (Gradient-weighted Class Activation Mapping) service.

Generates visual explanations for CNN predictions by highlighting
important regions in the input image that influenced the classification.
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Callable, Optional, Tuple
import tensorflow as tf

from app.config import settings


def compute_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str = "block5_conv3",  # VGG16's last conv layer
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given image.
    
    Args:
        model: The trained Keras model
        img_array: Preprocessed image array, shape (1, 224, 224, 3), normalized to [0, 1]
        last_conv_layer_name: Name of the last convolutional layer to use (default: block5_conv3 for VGG16)
        pred_index: Index of the class to compute Grad-CAM for (None = use predicted class)
    
    Returns:
        Heatmap as numpy array, shape (H, W), values in [0, 1]
    """
    # Create a model that maps input to (conv outputs, predictions)
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback to finding any conv layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                if hasattr(layer, 'output'):
                    last_conv_layer = layer
                    break
        else:
            raise ValueError(f"Could not find convolutional layer. Tried: {last_conv_layer_name}")
    
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array, training=False)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        pred_score = preds[:, pred_index]
    
    # Gradients of the predicted class with respect to the conv layer output
    grads = tape.gradient(pred_score, conv_output)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get the conv output and pooled gradients as numpy arrays
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    # Weight the channels by the gradient importance
    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]
    
    # Create heatmap by averaging across channels
    heatmap = np.mean(conv_output, axis=-1)
    
    # Apply ReLU to focus on positive influences
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize to [0, 1]
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    return heatmap


def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet"
) -> Image.Image:
    """
    Create a Grad-CAM overlay on top of the original image.
    
    Args:
        original_image: Original PIL Image
        heatmap: Grad-CAM heatmap array (H, W), values in [0, 1]
        alpha: Overlay transparency (0 = only original, 1 = only heatmap)
        colormap: Matplotlib colormap to use for the heatmap
    
    Returns:
        PIL Image with Grad-CAM overlay
    """
    import matplotlib.pyplot as plt
    
    # Resize original image to standard size
    img_size = (224, 224)
    img = original_image.convert("RGB").resize(img_size)
    img_np = np.array(img)
    
    # Resize heatmap to match image size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(img_size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil)
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized / 255.0)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend heatmap with original image
    overlay = (alpha * heatmap_colored + (1 - alpha) * img_np).astype(np.uint8)
    
    return Image.fromarray(overlay)


def _default_preprocess(image: Image.Image) -> np.ndarray:
    """Fallback preprocessing if no classifier-specific function is provided."""
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def generate_gradcam_for_image(
    image: Image.Image,
    model: Optional[tf.keras.Model] = None,
    preprocess_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
    last_conv_layer_name: Optional[str] = None,
    alpha: float = 0.4,
    classifier_type: Optional[str] = None
) -> Tuple[str, np.ndarray]:
    """
    Generate Grad-CAM visualization for an image using the classifier model.
    
    Args:
        image: PIL Image to analyze
        model: Optional pre-loaded model (if None, loads the classifier model)
        preprocess_fn: Callable that prepares the image for the classifier
        last_conv_layer_name: Name of the last conv layer for Grad-CAM (auto-detected if None)
        alpha: Overlay transparency
        classifier_type: "oct" or "fundus" (default from settings if None)
    
    Returns:
        Tuple of (base64 encoded overlay image, raw heatmap array)
    """
    model_to_use = model
    layer_name = last_conv_layer_name
    preprocess = preprocess_fn or _default_preprocess

    if model_to_use is None:
        # Lazy-load classifier model if not provided (avoids circular import when called from classifier service)
        from app.services.classifier import load_classifier_model, preprocess_image

        model_to_use, default_layer = load_classifier_model(classifier_type or "oct")
        layer_name = layer_name or default_layer
        if preprocess_fn is None:
            preprocess = lambda img: preprocess_image(img, classifier_type or "oct")

    # Use appropriate default layer if still not set
    if layer_name is None:
        layer_name = settings.OCT_GRADCAM_LAYER if classifier_type != "fundus" else settings.FUNDUS_GRADCAM_LAYER

    # Preprocess image
    img_array = preprocess(image)

    # Compute Grad-CAM heatmap
    heatmap = compute_gradcam(model_to_use, img_array, layer_name)
    
    # Create overlay image
    overlay = create_gradcam_overlay(image, heatmap, alpha)
    
    # Convert to base64
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{base64_image}", heatmap


def generate_gradcam_for_both(
    image: Image.Image,
    alpha: float = 0.4
) -> dict:
    """
    Generate Grad-CAM visualizations for both OCT and Fundus classifiers.
    
    Args:
        image: PIL Image to analyze
        alpha: Overlay transparency
    
    Returns:
        Dictionary with "oct" and "fundus" keys, each containing:
            - gradcam_image: base64 encoded overlay
            - insights: heatmap insights (JSON-serializable)
    """
    results = {}
    
    for classifier_type in ["oct", "fundus"]:
        try:
            gradcam_image, heatmap = generate_gradcam_for_image(
                image, 
                classifier_type=classifier_type, 
                alpha=alpha
            )
            insights = get_gradcam_insights(heatmap)
            
            # Don't include raw heatmap - it's not JSON serializable
            results[classifier_type] = {
                "gradcam_image": gradcam_image,
                "insights": insights
            }
        except Exception as e:
            print(f"[gradcam] generation failed for {classifier_type}: {e}")
            results[classifier_type] = None
    
    return results


def get_gradcam_insights(heatmap: np.ndarray) -> dict:
    """
    Extract insights from the Grad-CAM heatmap.
    
    Args:
        heatmap: Grad-CAM heatmap array (H, W)
    
    Returns:
        Dictionary with heatmap insights (all values are JSON-serializable)
    """
    # Find the region of highest activation
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Calculate focus metrics
    total_activation = float(np.sum(heatmap))
    threshold = 0.5
    high_activation_area = float(np.sum(heatmap > threshold) / heatmap.size * 100)
    
    # Determine focus region (quadrant-based)
    h, w = heatmap.shape
    center_y, center_x = max_idx
    
    regions = []
    if center_y < h / 2:
        regions.append("upper")
    else:
        regions.append("lower")
    
    if center_x < w / 2:
        regions.append("left")
    else:
        regions.append("right")
    
    focus_region = "-".join(regions)
    
    # Calculate concentration (how focused vs diffuse the attention is)
    concentration = float(np.std(heatmap) * 100)  # Higher = more focused
    
    return {
        "focus_region": focus_region,
        "focus_center": {"x": int(center_x), "y": int(center_y)},
        "high_activation_percentage": round(high_activation_area, 1),
        "concentration_score": round(concentration, 2),
        "interpretation": _interpret_gradcam(focus_region, high_activation_area, concentration)
    }


def _interpret_gradcam(focus_region: str, activation_area: float, concentration: float) -> str:
    """Generate a human-readable interpretation of the Grad-CAM results."""
    
    interpretations = []
    
    # Focus region interpretation
    region_map = {
        "upper-left": "superior-temporal region",
        "upper-right": "superior-nasal region", 
        "lower-left": "inferior-temporal region",
        "lower-right": "inferior-nasal region"
    }
    region_name = region_map.get(focus_region, focus_region)
    interpretations.append(f"The model focused primarily on the {region_name} of the image.")
    
    # Activation area interpretation
    if activation_area > 30:
        interpretations.append("The area of interest is diffuse, suggesting widespread pathology or normal tissue examination.")
    elif activation_area > 15:
        interpretations.append("The model identified a moderately sized region of interest.")
    else:
        interpretations.append("The model identified a highly localized abnormality or feature.")
    
    # Concentration interpretation
    if concentration > 20:
        interpretations.append("High confidence in the identified region.")
    elif concentration > 10:
        interpretations.append("Moderate certainty in the region of focus.")
    else:
        interpretations.append("The model's attention was relatively distributed across the image.")
    
    return " ".join(interpretations)
