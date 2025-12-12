"""
Multi-model classifier service for OCT and Fundus images.

Supports:
- VGG16 for OCT images (4 classes: CNV, DME, DRUSEN, NORMAL)
- MobileNetV2 for Fundus images (8 classes: AMD, Cataract, etc.)

Provides helper functions to classify images and format context for the
OctoMed VLM prompt.
"""

from pathlib import Path
from typing import Dict, Optional, Literal, Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, Sequential

from app.config import settings

# Type alias for classifier types
ClassifierType = Literal["oct", "fundus"]

# Cached models
_oct_model: Optional[Model] = None
_fundus_model: Optional[Model] = None

# Class labels for each model
OCT_LABELS = tuple(settings.OCT_CLASSIFIER_LABELS)
FUNDUS_LABELS = tuple(settings.FUNDUS_CLASSIFIER_LABELS)

# Legacy alias
CLASS_LABELS = OCT_LABELS


def _resolve_weights_path(path_str: str) -> Path:
    """Resolve weights path, relative to backend root if needed."""
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def load_oct_model() -> Model:
    """Load (or return cached) VGG16 OCT classifier."""
    global _oct_model

    if _oct_model is not None:
        return _oct_model

    weights_path = _resolve_weights_path(settings.OCT_CLASSIFIER_WEIGHTS_PATH)
    if not weights_path.exists():
        raise FileNotFoundError(f"OCT classifier weights not found at {weights_path}")

    # Build VGG16 architecture
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    outputs = Dense(len(OCT_LABELS), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.load_weights(str(weights_path))
    _oct_model = model
    print(f"✅ Loaded OCT classifier (VGG16) with {len(OCT_LABELS)} classes")
    return model


def load_fundus_model() -> Model:
    """Load (or return cached) MobileNetV2 Fundus classifier."""
    global _fundus_model

    if _fundus_model is not None:
        return _fundus_model

    weights_path = _resolve_weights_path(settings.FUNDUS_CLASSIFIER_WEIGHTS_PATH)
    if not weights_path.exists():
        raise FileNotFoundError(f"Fundus classifier weights not found at {weights_path}")

    # Try loading as a complete model first (if saved with model.save())
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(str(weights_path), compile=False)
        _fundus_model = model
        print(f"✅ Loaded Fundus classifier (complete model) with {len(FUNDUS_LABELS)} classes")
        return model
    except Exception as e:
        print(f"⚠️ Could not load as complete model: {e}")
    
    # Fallback: Build MobileNetV2 architecture and load weights
    # Try different architectures to match saved weights
    architectures = [
        # Architecture 1: Simple top
        lambda: _build_mobilenet_simple(),
        # Architecture 2: With dense layers
        lambda: _build_mobilenet_with_dense(),
        # Architecture 3: Minimal custom layers
        lambda: _build_mobilenet_minimal(),
    ]
    
    for i, build_fn in enumerate(architectures):
        try:
            model = build_fn()
            model.load_weights(str(weights_path))
            _fundus_model = model
            print(f"✅ Loaded Fundus classifier (MobileNetV2 arch {i+1}) with {len(FUNDUS_LABELS)} classes")
            return model
        except Exception as e:
            print(f"⚠️ Architecture {i+1} failed: {e}")
            continue
    
    raise ValueError(f"Could not load Fundus model weights - no matching architecture found")


def _build_mobilenet_simple() -> Model:
    """MobileNetV2 with simple GlobalAveragePooling + Dense output."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    x = GlobalAveragePooling2D()(base_model.output)
    outputs = Dense(len(FUNDUS_LABELS), activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=outputs)


def _build_mobilenet_with_dense() -> Model:
    """MobileNetV2 with pooling='avg' + Flatten + Dense layers."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(len(FUNDUS_LABELS), activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=outputs)


def _build_mobilenet_minimal() -> Model:
    """MobileNetV2 with minimal custom layers (4 layers total)."""
    # This matches a model with only 4 saved layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(FUNDUS_LABELS), activation='softmax')
    ])
    return model


def load_classifier_model(classifier_type: ClassifierType = None) -> Tuple[Model, tuple, str]:
    """
    Load the appropriate classifier model.
    
    Args:
        classifier_type: "oct" for VGG16, "fundus" for MobileNetV2, None for default
        
    Returns:
        Tuple of (model, class_labels, gradcam_layer_name)
    """
    if classifier_type is None:
        classifier_type = settings.DEFAULT_CLASSIFIER
    
    if classifier_type == "fundus":
        return (
            load_fundus_model(),
            FUNDUS_LABELS,
            settings.FUNDUS_GRADCAM_LAYER
        )
    else:
        return (
            load_oct_model(),
            OCT_LABELS,
            settings.OCT_GRADCAM_LAYER
        )


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and scale image to model input tensor."""
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def classify_image(
    image: Image.Image,
    classifier_type: ClassifierType = None
) -> Optional[Dict]:
    """
    Run inference on a PIL image and return prediction details.
    """
    if not settings.ENABLE_CLASSIFIER:
        return None

    if classifier_type is None:
        classifier_type = settings.DEFAULT_CLASSIFIER

    model, labels, gradcam_layer = load_classifier_model(classifier_type)
    inputs = _preprocess_image(image)
    preds = model.predict(inputs, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx]) * 100
    probabilities = {
        label: float(preds[i]) * 100 for i, label in enumerate(labels)
    }

    model_name = "VGG16-OCT" if classifier_type == "oct" else "MobileNetV2-Fundus"
    
    return {
        "model": model_name,
        "classifier_type": classifier_type,
        "label": labels[pred_idx],
        "confidence": confidence,
        "probabilities": probabilities,
        "gradcam_layer": gradcam_layer,
    }


def classify_with_both(image: Image.Image) -> Dict[str, Optional[Dict]]:
    """Run classification with both OCT and Fundus models."""
    results = {}
    
    try:
        results["oct"] = classify_image(image, "oct")
    except Exception as e:
        print(f"⚠️ OCT classification failed: {e}")
        results["oct"] = None
    
    try:
        results["fundus"] = classify_image(image, "fundus")
    except Exception as e:
        print(f"⚠️ Fundus classification failed: {e}")
        results["fundus"] = None
    
    return results


def format_classifier_context(result: Optional[Dict]) -> str:
    """Format classifier output into a concise text snippet for prompts."""
    if not result:
        return "No auxiliary classifier output is available for this image."

    model_name = result.get("model", "Unknown")
    classifier_type = result.get("classifier_type", "unknown")
    
    probs_text = ", ".join(
        f"{label}: {prob:.1f}%" for label, prob in result["probabilities"].items()
    )
    
    return (
        f"Auxiliary {model_name} classifier ({classifier_type.upper()} mode) suggests '{result['label']}' "
        f"({result['confidence']:.1f}% confidence). "
        f"Class probabilities -> {probs_text}. "
        "Use this as supporting context; verify against the image itself."
    )


def format_dual_classifier_context(results: Dict[str, Optional[Dict]]) -> str:
    """Format results from both classifiers into a prompt context."""
    parts = []
    
    if results.get("oct"):
        oct = results["oct"]
        parts.append(
            f"OCT Classifier (VGG16): '{oct['label']}' ({oct['confidence']:.1f}% confidence)"
        )
    
    if results.get("fundus"):
        fundus = results["fundus"]
        parts.append(
            f"Fundus Classifier (MobileNetV2): '{fundus['label']}' ({fundus['confidence']:.1f}% confidence)"
        )
    
    if not parts:
        return "No auxiliary classifier output is available for this image."
    
    return (
        "Auxiliary classifier outputs (use as hints, verify visually):\n- " +
        "\n- ".join(parts)
    )
