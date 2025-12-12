"""
OctoMed-7B Vision-Language Model Service for Eye/Medical Image Analysis.

This service uses the OctoMed/OctoMed-7B model from Hugging Face for
analyzing medical images with high accuracy.
"""

import torch
import json
import re
import base64
import io
from PIL import Image
from app.config import settings

# Global model instances (loaded once at startup)
_model = None
_processor = None
_device = None


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """
    Load the OctoMed-7B model and processor.
    Should be called once at application startup.
    """
    global _model, _processor, _device
    
    if _model is not None:
        return  # Already loaded
    
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    print("🔄 Loading OctoMed-7B model...")
    
    _device = get_device()
    print(f"   Using device: {_device}")
    
    # Load model with appropriate settings based on device
    model_kwargs = {
        "device_map": "auto" if _device == "cuda" else None,
    }
    
    if _device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "OctoMed/OctoMed-7B",
        **model_kwargs
    )
    
    # Move to device if not using device_map
    if _device != "cuda":
        _model = _model.to(_device)
    
    # Configure processor with optimized pixel settings
    min_pixels = 262144
    max_pixels = 262144
    _processor = AutoProcessor.from_pretrained(
        "OctoMed/OctoMed-7B",
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    print("✅ OctoMed-7B model loaded successfully")


def is_model_loaded() -> bool:
    """Check if the model is loaded."""
    return _model is not None and _processor is not None


def unload_model():
    """Unload the model to free memory."""
    global _model, _processor, _device
    
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("🗑️ OctoMed model unloaded")


def decode_base64_image(image_data: str) -> Image.Image:
    """
    Decode a base64 image string to PIL Image.
    
    Args:
        image_data: Base64 encoded image (with or without data URL prefix)
        
    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    clean_base64 = re.sub(r'^data:image/(png|jpeg|jpg|webp|bmp);base64,', '', image_data)
    
    # Decode base64
    image_bytes = base64.b64decode(clean_base64)
    
    # Open as PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


async def analyze_eye_image(image_data: str, classifier_type: str = None) -> dict:
    """
    Analyze an eye/medical image using OctoMed-7B model.
    
    Args:
        image_data: Base64 encoded image data (with or without data URL prefix)
        classifier_type: "oct", "fundus", or None for auto-detection using both
    
    Returns:
        Dictionary with analysis results matching AIAnalysisResult schema
    """
    global _model, _processor, _device
    
    if not is_model_loaded():
        raise RuntimeError("OctoMed model is not loaded. Call load_model() first.")
    
    from qwen_vl_utils import process_vision_info
    
    # Decode the image
    image = decode_base64_image(image_data)

    # Run auxiliary classifiers for context (best-effort).
    classifier_result = None
    classifier_context = ""
    gradcam_image = None
    gradcam_insights = None
    
    if settings.ENABLE_CLASSIFIER:
        try:
            from app.services.classifier import (
                classify_image, 
                format_classifier_context,
            )
            from app.services.gradcam import (
                generate_gradcam_for_image, 
                get_gradcam_insights
            )

            # classify_image now auto-detects modality if classifier_type is None
            # This uses the ResNet18 modality detector to choose OCT vs Fundus
            classifier_result = classify_image(image, classifier_type, auto_detect=True)
            classifier_context = format_classifier_context(classifier_result)
            
            # Get the detected/used classifier type for Grad-CAM
            used_classifier_type = classifier_result.get("classifier_type") if classifier_result else None
            
            # Generate Grad-CAM for the classifier that was used
            if used_classifier_type:
                try:
                    gradcam_image, heatmap = generate_gradcam_for_image(
                        image, 
                        classifier_type=used_classifier_type
                    )
                    gradcam_insights = get_gradcam_insights(heatmap)
                    print(f"✅ Grad-CAM generated ({used_classifier_type}): {gradcam_insights['focus_region']}")
                except Exception as gc_error:
                    print(f"⚠️ Grad-CAM generation failed ({used_classifier_type}): {gc_error}")
                
        except Exception as cls_error:
            classifier_context = (
                "Auxiliary classifier could not run for this image; "
                f"proceed with visual analysis only. Error: {cls_error}"
            )
            print(f"Classifier inference error: {cls_error}")
    
    # Save temporarily to create proper message format
    # OctoMed expects image path or URL, so we'll use PIL image directly
    
    # Build Grad-CAM context for the prompt
    gradcam_context = ""
    if gradcam_insights:
        gradcam_context = (
            f"\nGrad-CAM Visual Attention Analysis:\n"
            f"- Focus Region: {gradcam_insights['focus_region']}\n"
            f"- High Activation Area: {gradcam_insights['high_activation_percentage']}%\n"
            f"- Interpretation: {gradcam_insights['interpretation']}\n"
        )
    
    # Add modality detection info to context
    modality_context = ""
    if classifier_result and classifier_result.get("detected_modality"):
        det = classifier_result["detected_modality"]
        modality_context = f"\nImage Modality: {det['modality'].upper()} (auto-detected with {det['confidence']:.1f}% confidence)\n"
    
    # Create the analysis prompt
    prompt_parts = [
        "You are an expert AI ophthalmology assistant specialized in analyzing eye and retinal images.",
        modality_context,
        "Auxiliary context from CNN classifier (use as hints but verify visually):",
        classifier_context or "No auxiliary classifier output is available for this image.",
        gradcam_context,
        "",
        "Analyze this medical eye scan/image and provide a detailed structured report.",
        "Consider the Grad-CAM attention map analysis which highlights regions the classifier focused on.",
        "",
        "Respond with a JSON object containing:",
        '1. "classification": The primary diagnosis or condition (e.g., "Diabetic Retinopathy", "Glaucoma Suspect", "Normal", etc.)',
        '2. "confidence": A number from 0 to 100 indicating certainty',
        '3. "findings": An array of specific observations from the image',
        '4. "recommendation": A clear recommendation for the ophthalmologist',
        '5. "explanation": A detailed explanation on how you arrived at the classification and key findings.',
        "",
        "Respond ONLY with valid JSON in this exact format:",
        '{"classification": "string", "confidence": number, "findings": ["string"], "recommendation": "string", "explanation": "string"}',
    ]
    prompt = "\n".join(prompt_parts)

    # Create message in OctoMed format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    try:
        # Prepare inputs
        text = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = _processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        
        # Move inputs to device
        inputs = inputs.to(device=_device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = _model.generate(**inputs, max_new_tokens=2048)
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON response
        result = parse_model_response(output_text)

        # Attach classifier context to response for transparency
        if classifier_result:
            result["classifier"] = classifier_result
        
        # Attach Grad-CAM visualization
        if gradcam_image:
            result["gradcamImage"] = gradcam_image
        if gradcam_insights:
            result["gradcamInsights"] = gradcam_insights
        
        return result
        
    except Exception as e:
        print(f"OctoMed analysis error: {e}")
        raise


def parse_model_response(response_text: str) -> dict:
    """
    Parse the model's response text into a structured dictionary.
    
    Args:
        response_text: Raw text output from the model
        
    Returns:
        Dictionary with analysis results
    """
    # Clean up response text
    response_text = response_text.strip()
    
    # Try to extract JSON from the response
    # First, try direct JSON parsing
    try:
        result = json.loads(response_text)
        return validate_result(result)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in the response
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return validate_result(result)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON with nested structures
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return validate_result(result)
        except json.JSONDecodeError:
            pass
    
    # If no JSON found, create structured response from text
    return {
        "classification": "Analysis Complete",
        "confidence": 70,
        "findings": [response_text[:500] if len(response_text) > 500 else response_text],
        "recommendation": "Please review the findings with a specialist",
        "explanation": response_text
    }


def validate_result(result: dict) -> dict:
    """
    Validate and normalize the analysis result.
    
    Args:
        result: Raw parsed result dictionary
        
    Returns:
        Validated and normalized result
    """
    # Ensure all required fields exist
    required_fields = {
        "classification": "Unknown",
        "confidence": 50,
        "findings": [],
        "recommendation": "Further review recommended",
        "explanation": "No detailed explanation provided"
    }
    
    for field, default in required_fields.items():
        if field not in result:
            result[field] = default
    
    # Normalize confidence to float
    try:
        result["confidence"] = float(result["confidence"])
        result["confidence"] = max(0, min(100, result["confidence"]))
    except (ValueError, TypeError):
        result["confidence"] = 50
    
    # Ensure findings is a list
    if not isinstance(result["findings"], list):
        result["findings"] = [str(result["findings"])]
    
    # Ensure all findings are strings
    result["findings"] = [str(f) for f in result["findings"]]
    
    return result
