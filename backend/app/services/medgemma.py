"""
Med-Gemma vision-language model service for ophthalmic image cross-checking.

Provides a lightweight wrapper to run the google/medgemma-4b-it model and
return structured ophthalmology-style findings.
"""

import asyncio
import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from app.config import settings

_model = None
_processor = None
_device = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _decode_base64_image(image_data: str) -> Image.Image:
    """Decode base64 image data into a PIL Image."""
    clean_base64 = re.sub(
        r"^data:image/(png|jpeg|jpg|webp|bmp);base64,", "", image_data
    )
    image_bytes = base64.b64decode(clean_base64)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def is_model_loaded() -> bool:
    return _model is not None and _processor is not None


def load_model() -> None:
    """Load Med-Gemma model and processor."""
    global _model, _processor, _device
    if _model is not None:
        return

    from transformers import AutoModelForImageTextToText, AutoProcessor

    _device = _get_device()
    model_kwargs: Dict[str, Any] = {}

    if _device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    elif _device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    _model = AutoModelForImageTextToText.from_pretrained(
        settings.MEDGEMMA_MODEL_NAME, **model_kwargs
    )
    if _device != "cuda":
        _model = _model.to(_device)

    _processor = AutoProcessor.from_pretrained(settings.MEDGEMMA_MODEL_NAME)
    print(f"Med-Gemma model loaded on device: {_device}")


def _parse_model_response(response_text: str) -> Dict[str, Any]:
    """Parse Med-Gemma text output into structured JSON-like dict."""
    response_text = response_text.strip()

    try:
        candidate = json.loads(response_text)
        return _validate_result(candidate)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            candidate = json.loads(json_match.group())
            return _validate_result(candidate)
        except json.JSONDecodeError:
            pass

    return {
        "classification": "Analysis Complete",
        "confidence": 60,
        "findings": [response_text[:500] if len(response_text) > 500 else response_text],
        "recommendation": "Please review with a specialist",
        "explanation": response_text,
    }


def _validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    required_defaults = {
        "classification": "Unknown",
        "confidence": 50,
        "findings": [],
        "recommendation": "Further review recommended",
        "explanation": "No detailed explanation provided",
    }
    for field, default in required_defaults.items():
        if field not in result:
            result[field] = default

    try:
        result["confidence"] = float(result.get("confidence", 50))
        result["confidence"] = max(0, min(100, result["confidence"]))
    except (TypeError, ValueError):
        result["confidence"] = 50

    findings = result.get("findings", [])
    if not isinstance(findings, list):
        findings = [findings]
    result["findings"] = [str(item) for item in findings]
    return result


def _build_prompt() -> str:
    return "\n".join(
        [
            "You are an expert ophthalmology assistant specialized in retinal imaging.",
            "Analyze the provided image and respond ONLY with JSON:",
            '{"classification": "string", "confidence": number, "findings": ["string"], "recommendation": "string", "explanation": "string"}',
        ]
    )


def _infer_sync(image_data: str) -> Dict[str, Any]:
    if not is_model_loaded():
        load_model()

    image = _decode_base64_image(image_data)
    prompt = _build_prompt()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert ophthalmology assistant."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        gen_tokens = generation[0][input_len:]

    decoded = _processor.decode(gen_tokens, skip_special_tokens=True)
    parsed = _parse_model_response(decoded)
    parsed["model"] = settings.MEDGEMMA_MODEL_NAME
    return parsed


async def analyze_eye_image(image_data: str) -> Dict[str, Any]:
    """Public async wrapper for Med-Gemma inference."""
    return await asyncio.to_thread(_infer_sync, image_data)
