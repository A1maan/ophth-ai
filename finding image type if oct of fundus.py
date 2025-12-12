import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def load_checkpoint(checkpoint_path, device):
    """
    Load the trained model checkpoint and rebuild the model.
    Expects a checkpoint dict with:
      - "model_state_dict"
      - "class_to_idx"
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    # Build the same architecture as training (ResNet18)
    model = models.resnet18(weights=None)  # weights=None, we will load from checkpoint
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Invert mapping: idx -> class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class


def preprocess_image(image_path):
    """
    Apply the same preprocessing used in training:
      - Resize to 256x256
      - Convert to tensor
      - Normalize with ImageNet stats
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # add batch dimension
    return tensor


def predict(image_path, checkpoint_path):
    """
    Load model, preprocess image, run inference, and return JSON-able dict:
      {
        "modality": "fundus" or "oct",
        "confidence": float between 0 and 1
      }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and mapping
    model, idx_to_class = load_checkpoint(checkpoint_path, device)

    # Preprocess the image
    image_tensor = preprocess_image(image_path).to(device)

    # Run model
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    modality = idx_to_class[pred_idx.item()]
    confidence = conf.item()

    result = {
        "modality": modality,
        "confidence": round(confidence, 4),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Infer whether an input image is fundus or OCT."
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to the input image (jpg/png)."
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default="modality_classifier.pth",
        help="Path to the trained model checkpoint (.pth)."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    result = predict(args.image, args.checkpoint)

    # Print JSON for easy parsing
    print(json.dumps(result))

    # Also print a human-readable line
    print(f"Predicted modality: {result['modality']} "
          f"(confidence={result['confidence']:.4f})")


if __name__ == "__main__":
    main()