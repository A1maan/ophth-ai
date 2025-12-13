"""
Modality Detector Service - Determines if an image is OCT or Fundus.

Uses a trained ResNet18 model to classify the image modality before
routing to the appropriate classifier (VGG16 for OCT, MobileNetV2 for Fundus).
"""

from pathlib import Path
from typing import Dict, Optional, Literal, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from app.config import settings

# Type alias
ModalityType = Literal["oct", "fundus"]

# Cached model and mappings
_modality_model: Optional[nn.Module] = None
_idx_to_class: Optional[Dict[int, str]] = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_checkpoint_path() -> Path:
    """Resolve the modality classifier checkpoint path."""
    path = Path(settings.MODALITY_CLASSIFIER_PATH)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def load_modality_model() -> Tuple[nn.Module, Dict[int, str], torch.device]:
    """
    Load (or return cached) the modality detection model.
    
    Returns:
        Tuple of (model, idx_to_class mapping, device)
    """
    global _modality_model, _idx_to_class, _device
    
    if _modality_model is not None:
        return _modality_model, _idx_to_class, _device
    
    checkpoint_path = _resolve_checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Modality classifier checkpoint not found at {checkpoint_path}")
    
    _device = _get_device()
    print(f"[modality] Loading modality detector on {_device}...")
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=_device)
    
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)
    
    # Build ResNet18 architecture (same as training)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    
    # Invert mapping: idx -> class name
    _idx_to_class = {v: k for k, v in class_to_idx.items()}
    _modality_model = model
    
    print(f"[modality] Modality detector loaded: {list(class_to_idx.keys())}")
    return _modality_model, _idx_to_class, _device


def _preprocess_for_modality(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for the modality detector.
    Uses same transforms as training.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)  # add batch dimension
    return tensor


def detect_modality(image: Image.Image) -> Dict:
    """
    Detect whether an image is OCT or Fundus.
    
    Args:
        image: PIL Image to classify
        
    Returns:
        Dictionary with:
            - modality: "oct" or "fundus"
            - confidence: float between 0 and 100
            - raw_label: original label from model (may be capitalized)
    """
    model, idx_to_class, device = load_modality_model()
    
    # Preprocess
    image_tensor = _preprocess_for_modality(image).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    
    raw_label = idx_to_class[pred_idx.item()]
    confidence = float(conf.item()) * 100
    
    # Normalize label to lowercase
    modality = raw_label.lower()
    if modality not in ["oct", "fundus"]:
        # Try to map common variations
        if "oct" in modality:
            modality = "oct"
        elif "fundus" in modality or "retina" in modality:
            modality = "fundus"
        else:
            # Default to OCT if unknown
            print(f"[modality] Unknown modality '{raw_label}', defaulting to 'oct'")
            modality = "oct"
    
    return {
        "modality": modality,
        "confidence": round(confidence, 2),
        "raw_label": raw_label,
    }


def detect_modality_from_path(image_path: str) -> Dict:
    """
    Detect modality from an image file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with modality detection results
    """
    image = Image.open(image_path).convert("RGB")
    return detect_modality(image)


def is_modality_detector_available() -> bool:
    """Check if the modality detector checkpoint exists."""
    try:
        path = _resolve_checkpoint_path()
        return path.exists()
    except Exception:
        return False
