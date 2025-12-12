"""
Classifier inference utilities for OCT and fundus images plus Grad-CAM helpers.

Handles weight loading, preprocessing, modality-aware routing, and formatting
for the OctoMed VLM prompt.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess,
)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model

from app.config import settings

# Cached models keyed by classifier type ("oct" | "fundus")
_classifier_models: Dict[str, Model] = {}

DEFAULT_LABELS = ("CNV", "DME", "DRUSEN", "NORMAL")
CLASSIFIER_LABELS: Dict[str, Tuple[str, ...]] = {
    "oct": tuple(settings.CLASSIFIER_LABELS or DEFAULT_LABELS),
    "fundus": tuple(
        settings.FUNDUS_CLASSIFIER_LABELS
        or settings.CLASSIFIER_LABELS
        or DEFAULT_LABELS
    ),
}

CLASSIFIER_PATHS = {
    "oct": settings.CLASSIFIER_WEIGHTS_PATH,
    "fundus": settings.FUNDUS_CLASSIFIER_WEIGHTS_PATH,
}

DEFAULT_GRADCAM_LAYERS = {
    "oct": settings.OCT_GRADCAM_LAYER or "block5_conv3",
    "fundus": settings.FUNDUS_GRADCAM_LAYER or "Conv_1",
}


def _normalize_type(classifier_type: Optional[str]) -> str:
    ctype = (classifier_type or "oct").lower()
    return ctype if ctype in CLASSIFIER_PATHS else "oct"


def _weights_path(classifier_type: str) -> Path:
    """Resolve the classifier weights path for the given model type."""
    path = Path(CLASSIFIER_PATHS.get(classifier_type, settings.CLASSIFIER_WEIGHTS_PATH))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def _build_model(classifier_type: str, num_classes: int) -> Model:
    """Construct the classifier architecture matching training."""
    if classifier_type == "fundus":
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        outputs = Dense(num_classes, activation="softmax")(x)
        return Model(inputs=base_model.input, outputs=outputs)

    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=outputs)


def load_classifier_model(classifier_type: str = "oct") -> Tuple[Model, str]:
    """Load (or return cached) classifier model and its default Grad-CAM layer."""
    classifier_type = _normalize_type(classifier_type)

    if classifier_type in _classifier_models:
        return _classifier_models[classifier_type], DEFAULT_GRADCAM_LAYERS[classifier_type]

    weights_path = _weights_path(classifier_type)
    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier weights not found at {weights_path}")

    labels = CLASSIFIER_LABELS[classifier_type]
    num_classes = len(labels)

    model: Optional[Model] = None
    # Try to load a full saved model first (if the file contains architecture).
    try:
        model = load_model(str(weights_path))
    except Exception:
        model = None

    # Fallback: build architecture and load weights only.
    if model is None:
        model = _build_model(classifier_type, num_classes)
        model.load_weights(str(weights_path))

    _classifier_models[classifier_type] = model
    return model, DEFAULT_GRADCAM_LAYERS[classifier_type]


def preprocess_image(image: Image.Image, classifier_type: str = "oct") -> np.ndarray:
    """Resize and scale image to model input tensor."""
    classifier_type = _normalize_type(classifier_type)
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")

    if classifier_type == "fundus":
        arr = mobilenet_preprocess(arr)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def classify_image(image: Image.Image, classifier_type: str = "oct") -> Optional[Dict]:
    """
    Run inference on a PIL image and return prediction details.

    Returns None if classifier is disabled in settings.
    """
    if not settings.ENABLE_CLASSIFIER:
        return None

    classifier_type = _normalize_type(classifier_type)
    labels = CLASSIFIER_LABELS[classifier_type]
    model, _ = load_classifier_model(classifier_type)
    inputs = preprocess_image(image, classifier_type)
    preds = model.predict(inputs, verbose=0)[0]  # probability vector

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100
    probabilities = {label: float(preds[i]) * 100 for i, label in enumerate(labels)}

    return {
        "model": "VGG16-OCT-Classifier" if classifier_type == "oct" else "MobileNetV2-Fundus-Classifier",
        "label": labels[pred_idx],
        "confidence": confidence,
        "probabilities": probabilities,
        "classifier_type": classifier_type,
    }


def run_classifier_with_gradcam(image: Image.Image, modality_hint: Optional[str] = None) -> Dict:
    """
    Run modality detection (if enabled), classification, and Grad-CAM generation.

    Returns a dict with classifier output, modality metadata, Grad-CAM image, and insights.
    """
    modality = None
    classifier_type = _normalize_type(modality_hint)

    if classifier_type == "oct" and settings.ENABLE_MODALITY_DETECTOR:
        try:
            from app.services.modality_detector import detect_modality, is_modality_detector_available

            if is_modality_detector_available():
                modality = detect_modality(image)
                classifier_type = _normalize_type(modality.get("modality"))
        except Exception as exc:
            print(f"[modality] detector failed: {exc}")

    classifier_result = classify_image(image, classifier_type)

    gradcam_image = None
    gradcam_insights = None
    if classifier_result:
        try:
            model, last_conv_layer = load_classifier_model(classifier_type)
            from app.services.gradcam import generate_gradcam_for_image, get_gradcam_insights

            gradcam_image, heatmap = generate_gradcam_for_image(
                image=image,
                model=model,
                preprocess_fn=lambda img: preprocess_image(img, classifier_type),
                last_conv_layer_name=last_conv_layer,
                classifier_type=classifier_type,
            )
            gradcam_insights = get_gradcam_insights(heatmap)
        except Exception as exc:  # pragma: no cover - best effort for visualization
            print(f"[gradcam] generation failed for {classifier_type}: {exc}")

    return {
        "classifier": classifier_result,
        "modality": modality,
        "classifier_type": classifier_type,
        "gradcam_image": gradcam_image,
        "gradcam_insights": gradcam_insights,
    }


def format_classifier_context(result: Optional[Dict]) -> str:
    """Format classifier output into a concise text snippet for prompts."""
    if not result:
        return "No auxiliary classifier output is available for this image."

    probs_text = ", ".join(
        f"{label}: {prob:.1f}%" for label, prob in result["probabilities"].items()
    )
    modality_hint = result.get("classifier_type", "oct").upper()
    return (
        f"Auxiliary {modality_hint} classifier suggests '{result['label']}' "
        f"({result['confidence']:.1f}% confidence). "
        f"Class probabilities -> {probs_text}. "
        "Use this as supporting context; verify against the image itself."
    )
