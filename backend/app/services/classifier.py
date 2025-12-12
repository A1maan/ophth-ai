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

# Default label sets
DEFAULT_OCT_LABELS = ("CNV", "DME", "DRUSEN", "NORMAL")
DEFAULT_FUNDUS_LABELS = (
    "AMD",
    "CATARACT",
    "DIABETIC_RETINOPATHY",
    "GLAUCOMA",
    "HYPERTENSION",
    "MYOPIA",
    "NORMAL",
    "OTHERS",
)
CLASSIFIER_LABELS: Dict[str, Tuple[str, ...]] = {
    "oct": tuple(settings.CLASSIFIER_LABELS or DEFAULT_OCT_LABELS),
    "fundus": tuple(settings.FUNDUS_CLASSIFIER_LABELS or DEFAULT_FUNDUS_LABELS),
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


def _auto_choose_type(image: Image.Image) -> str:
    """
    Best-effort modality guess.
    - Use modality detector if enabled/available.
    - Fallback heuristic: RGB/RGBA -> fundus, grayscale -> oct.
    """
    if settings.ENABLE_MODALITY_DETECTOR:
        try:
            from app.services.modality_detector import detect_modality, is_modality_detector_available

            if is_modality_detector_available():
                detection = detect_modality(image)
                modality = detection.get("modality")
                confidence = detection.get("confidence", 0)
                print(
                    f"[classifier] Modality detector result: {modality} ({confidence:.1f}%)"
                )
                if modality:
                    return _normalize_type(modality)
        except Exception as exc:
            print(f"[classifier] Modality auto-detect failed: {exc}")

    # Heuristic fallback
    if image.mode not in ("L", "LA"):
        print("[classifier] Heuristic chose fundus (color image)")
        return "fundus"
    print("[classifier] Heuristic chose oct (grayscale image)")
    return "oct"


def _weights_path(classifier_type: str) -> Path:
    """Resolve the classifier weights path for the given model type."""
    path = Path(CLASSIFIER_PATHS.get(classifier_type, settings.CLASSIFIER_WEIGHTS_PATH))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def _build_model(classifier_type: str, num_classes: int) -> Model:
    """Construct the classifier architecture matching training."""
    if classifier_type == "fundus":
        # Matches saved MobileNetV2_custom.h5: pooling='avg' -> Flatten -> Dense256 -> Dense128 -> Dense8
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="avg",
        )
        x = Flatten()(base_model.output)  # redundant but present in saved model
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        return Model(inputs=base_model.input, outputs=outputs)

    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=outputs)


def _validate_output_shape(model: Model, labels: Tuple[str, ...], classifier_type: str) -> None:
    """Ensure the loaded model output matches the configured label count."""
    try:
        output_units = model.output_shape[-1]
    except Exception:
        return

    if output_units != len(labels):
        raise ValueError(
            f"{classifier_type.upper()} classifier mismatch: model outputs {output_units} classes "
            f"but {len(labels)} labels are configured. Update the label list or weights."
        )


def load_classifier_model(classifier_type: str = "oct") -> Tuple[Model, str]:
    """Load (or return cached) classifier model and its default Grad-CAM layer."""
    classifier_type = _normalize_type(classifier_type)
    print(f"[classifier] Loading model type={classifier_type}")

    if classifier_type in _classifier_models:
        print(f"[classifier] Using cached model for {classifier_type}")
        return _classifier_models[classifier_type], DEFAULT_GRADCAM_LAYERS[classifier_type]

    weights_path = _weights_path(classifier_type)
    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier weights not found at {weights_path}")

    labels = CLASSIFIER_LABELS[classifier_type]
    num_classes = len(labels)
    print(f"[classifier] Resolved weights: {weights_path}")
    print(f"[classifier] Labels ({num_classes}): {labels}")

    model: Optional[Model] = None
    # Try to load a full saved model first (if the file contains architecture).
    if classifier_type != "fundus":
        try:
            model = load_model(str(weights_path), compile=False, safe_mode=False)
            print(f"[classifier] Loaded full saved model for {classifier_type}")
        except Exception as exc:
            print(f"[classifier] Full model load failed ({exc}); trying architecture fallback")
            model = None

    # Fallback: build architecture and load weights only.
    if model is None:
        model = _build_model(classifier_type, num_classes)
        try:
            # Fundus weights may have truncated layer list; load by name to avoid mismatch errors.
            model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
        except TypeError:
            # Older TF may not support skip_mismatch kwarg; retry without.
            model.load_weights(str(weights_path))
        print(f"[classifier] Loaded weights into freshly built {classifier_type} model")

    _validate_output_shape(model, labels, classifier_type)
    print(f"[classifier] Model output units: {model.output_shape[-1]}")

    _classifier_models[classifier_type] = model
    return model, DEFAULT_GRADCAM_LAYERS[classifier_type]


def preprocess_image(image: Image.Image, classifier_type: str = "oct") -> np.ndarray:
    """Resize and scale image to model input tensor."""
    classifier_type = _normalize_type(classifier_type)
    print(f"[classifier] Preprocessing image for {classifier_type} (mode={image.mode}, size={image.size})")
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")

    if classifier_type == "fundus":
        arr = mobilenet_preprocess(arr)
    else:
        arr = arr / 255.0

    print(f"[classifier] Preprocess stats min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    return np.expand_dims(arr, axis=0)


def classify_image(image: Image.Image, classifier_type: Optional[str] = "oct") -> Optional[Dict]:
    """
    Run inference on a PIL image and return prediction details.

    Returns None if classifier is disabled in settings.
    """
    if not settings.ENABLE_CLASSIFIER:
        print("[classifier] Classifier disabled in settings; skipping inference")
        return None

    if classifier_type == "auto":
        classifier_type = _auto_choose_type(image)
    classifier_type = _normalize_type(classifier_type)
    print(f"[classifier] classify_image called with classifier_type={classifier_type}")
    labels = CLASSIFIER_LABELS[classifier_type]
    model, _ = load_classifier_model(classifier_type)
    inputs = preprocess_image(image, classifier_type)
    preds = model.predict(inputs, verbose=0)[0]  # probability vector
    print(f"[classifier] Raw predictions ({classifier_type}): {preds}")

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100
    probabilities = {label: float(preds[i]) * 100 for i, label in enumerate(labels)}
    print(
        f"[classifier] Top prediction {labels[pred_idx]} ({confidence:.2f}%) "
        f"| classifier_type={classifier_type}"
    )

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
                print(f"[classifier] Modality detector chose {classifier_type} ({modality})")
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
            print(f"[classifier] Grad-CAM generated for {classifier_type}")
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
