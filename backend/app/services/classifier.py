"""
VGG16-based OCT image classifier inference utilities.

Loads pretrained weights from the local `ml` directory and provides
helper functions to classify images and format context for the
OctoMed VLM prompt.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

from app.config import settings

# Globals
_classifier_model: Optional[Model] = None
CLASS_LABELS = tuple(settings.CLASSIFIER_LABELS or ["CNV", "DME", "DRUSEN", "NORMAL"])


def _weights_path() -> Path:
    """Resolve the classifier weights path, relative to backend root if needed."""
    path = Path(settings.CLASSIFIER_WEIGHTS_PATH)
    if not path.is_absolute():
        # backend/app/services -> backend -> ml/...
        path = Path(__file__).resolve().parents[2] / path
    return path


def load_classifier_model() -> Model:
    """Load (or return cached) VGG16 classifier with trained weights."""
    global _classifier_model

    if _classifier_model is not None:
        return _classifier_model

    weights_path = _weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier weights not found at {weights_path}")

    # Build the same architecture used during training.
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    outputs = Dense(len(CLASS_LABELS), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.load_weights(str(weights_path))
    _classifier_model = model
    return model


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and scale image to model input tensor."""
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def classify_image(image: Image.Image) -> Optional[Dict]:
    """
    Run inference on a PIL image and return prediction details.

    Returns None if classifier is disabled in settings.
    """
    if not settings.ENABLE_CLASSIFIER:
        return None

    model = load_classifier_model()
    inputs = _preprocess_image(image)
    preds = model.predict(inputs, verbose=0)[0]  # probability vector

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100
    probabilities = {
        label: float(preds[i]) * 100 for i, label in enumerate(CLASS_LABELS)
    }

    return {
        "model": "VGG16-OCT-Classifier",
        "label": CLASS_LABELS[pred_idx],
        "confidence": confidence,
        "probabilities": probabilities,
    }


def format_classifier_context(result: Optional[Dict]) -> str:
    """
    Format classifier output into a concise text snippet for prompts.
    """
    if not result:
        return "No auxiliary classifier output is available for this image."

    probs_text = ", ".join(
        f"{label}: {prob:.1f}%" for label, prob in result["probabilities"].items()
    )
    return (
        f"Auxiliary VGG16 OCT classifier suggests '{result['label']}' "
        f"({result['confidence']:.1f}% confidence). "
        f"Class probabilities -> {probs_text}. "
        "Use this as supporting context; verify against the image itself."
    )
