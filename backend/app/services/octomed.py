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
from typing import Any, Dict, List, Optional
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
    
    print("Loading OctoMed-7B model...")
    
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
    
    print("OctoMed-7B model loaded successfully")


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
    
    print("OctoMed model unloaded")


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


def _strip_think_tokens(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model outputs."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _generate_octomed_response(
    messages: List[Dict[str, Any]], max_new_tokens: int = 512
) -> str:
    """
    Run the loaded OctoMed model against a chat-style prompt.
    
    Supports both multimodal (image+text) and text-only messages, depending on
    the provided message content.
    """
    if not is_model_loaded():
        raise RuntimeError("OctoMed model is not loaded. Call load_model() first.")

    from qwen_vl_utils import process_vision_info

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
    inputs = inputs.to(device=_device)

    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return _strip_think_tokens(output_text)


def _summarize_gradcam(gradcam_insights: Optional[Dict[str, Any]]) -> str:
    """Convert Grad-CAM insight dict to a short, human-readable sentence."""
    if not gradcam_insights:
        return "Grad-CAM focus map not available."

    focus_region = gradcam_insights.get("focus_region", "an unspecified region")
    center = gradcam_insights.get("focus_center", {})
    coverage = gradcam_insights.get("high_activation_percentage")
    concentration = gradcam_insights.get("concentration_score")
    interpretation = gradcam_insights.get("interpretation")

    center_text = ""
    if isinstance(center, dict) and "x" in center and "y" in center:
        center_text = f" near coordinates ({center.get('x')}, {center.get('y')})"

    extras = []
    if coverage is not None:
        extras.append(f"hot area covers ~{coverage}% of the image")
    if concentration is not None:
        extras.append(f"attention concentration score {concentration}")

    parts = [f"Grad-CAM highlights the {focus_region}{center_text}."]
    if extras:
        parts.append("; ".join(extras) + ".")
    if interpretation:
        parts.append(str(interpretation))
    return " ".join(parts)


def _format_vote_for_prompt(vote: Dict[str, Any]) -> str:
    """Compact a vote dict into a single prompt-friendly line."""
    findings = vote.get("findings") or []
    findings_text = ""
    if isinstance(findings, list):
        findings_text = ", ".join(str(item) for item in findings if item)
    else:
        findings_text = str(findings)

    explanation = str(vote.get("explanation", "")).replace("\n", " ").strip()

    return (
        f"{vote.get('model', 'LLM')} -> classification: "
        f"{vote.get('classification', 'Unknown')} "
        f"({vote.get('confidence', 'n/a')}% confidence); "
        f"findings: {findings_text or 'n/a'}; "
        f"recommendation: {vote.get('recommendation', 'n/a')}; "
        f"explanation: {explanation or 'n/a'}"
    )


async def _run_llm_council(
    image_data: str,
    initial_result: Dict[str, Any],
    classifier_context: str,
    modality_text: str,
    gradcam_insights: Optional[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Convene a small council of LLMs (OctoMed + optional Gemini + OctoMed reviewer)
    to refine the final explanation.
    """
    votes: List[Dict[str, Any]] = [
        {
            "model": "OctoMed-7B",
            "classification": initial_result.get("classification"),
            "confidence": initial_result.get("confidence"),
            "findings": initial_result.get("findings"),
            "recommendation": initial_result.get("recommendation"),
            "explanation": initial_result.get("explanation"),
            "source": "vision-primary",
        }
    ]

    # Optional cross-check with Med-Gemma if enabled
    if getattr(settings, "ENABLE_MEDGEMMA_CROSSCHECK", False):
        try:
            from app.services.medgemma import analyze_eye_image as analyze_with_medgemma

            med_vote = await analyze_with_medgemma(image_data)
            med_vote["model"] = med_vote.get("model", settings.MEDGEMMA_MODEL_NAME)
            med_vote["source"] = "cross-check"
            votes.append(med_vote)
        except Exception as exc:
            print(f"[council] Med-Gemma vote failed: {exc}")

    reviewer_prompt_parts = [
        "You are an independent ophthalmology LLM reviewing another model's report.",
        f"Proposed classification: {initial_result.get('classification')} "
        f"({initial_result.get('confidence')}% confidence).",
        f"Findings: {', '.join(initial_result.get('findings', [])) or 'n/a'}",
        f"Recommendation: {initial_result.get('recommendation')}",
        f"Modality cue: {modality_text}",
        f"Auxiliary classifier hint: {classifier_context}",
    ]
    if gradcam_insights:
        reviewer_prompt_parts.append(_summarize_gradcam(gradcam_insights))
    reviewer_prompt_parts.append(
        "Provide a short second-opinion explanation (<=120 words) noting agreement "
        "or disagreements and any safety flags."
    )

    reviewer_messages = [
        {"role": "user", "content": [{"type": "text", "text": "\n".join(reviewer_prompt_parts)}]}
    ]
    try:
        reviewer_explanation = _generate_octomed_response(reviewer_messages, max_new_tokens=384)
        votes.append(
            {
                "model": "OctoMed-Reviewer",
                "classification": initial_result.get("classification"),
                "confidence": initial_result.get("confidence"),
                "findings": initial_result.get("findings"),
                "recommendation": initial_result.get("recommendation"),
                "explanation": reviewer_explanation,
                "source": "text-review",
            }
        )
    except Exception as exc:
        print(f"[council] Reviewer generation failed: {exc}")

    vote_lines = "\n".join(_format_vote_for_prompt(vote) for vote in votes)
    council_prompt_parts = [
        "You are coordinating a council of ophthalmology LLMs.",
        "Combine their perspectives into one clear final explanation for a clinician.",
        modality_text,
        f"Auxiliary classifier context: {classifier_context}",
        _summarize_gradcam(gradcam_insights),
        "Council opinions:",
        vote_lines,
        (
            "Produce one concise explanation (<=150 words) describing the reasoning, where the "
            "council agrees or diverges, and why the final stance was chosen. Avoid JSON; "
            "return prose only."
        ),
    ]
    council_messages = [
        {"role": "user", "content": [{"type": "text", "text": "\n".join(council_prompt_parts)}]}
    ]

    try:
        consensus = _generate_octomed_response(council_messages, max_new_tokens=448)
        final_explanation = _strip_think_tokens(consensus) or initial_result.get(
            "explanation", ""
        )
    except Exception as exc:
        print(f"[council] Consensus generation failed: {exc}")
        final_explanation = initial_result.get("explanation", "")

    return final_explanation.strip(), votes


async def analyze_eye_image(image_data: str) -> dict:
    """
    Analyze an eye/medical image using OctoMed-7B model.
    
    Args:
        image_data: Base64 encoded image data (with or without data URL prefix)
    
    Returns:
        Dictionary with analysis results matching AIAnalysisResult schema
    """
    global _model, _processor, _device
    
    if not is_model_loaded():
        raise RuntimeError("OctoMed model is not loaded. Call load_model() first.")

    # Decode the image
    image = decode_base64_image(image_data)

    # Run auxiliary VGG16 classifier for context (best-effort).
    classifier_result = None
    classifier_context = ""
    gradcam_image = None
    gradcam_insights = None
    modality_result = None

    try:
        from app.services.classifier import (
            format_classifier_context,
            run_classifier_with_gradcam,
        )

        classifier_bundle = run_classifier_with_gradcam(image)
        classifier_result = classifier_bundle.get("classifier")
        classifier_context = format_classifier_context(classifier_result)
        gradcam_image = classifier_bundle.get("gradcam_image")
        gradcam_insights = classifier_bundle.get("gradcam_insights")
        modality_result = classifier_bundle.get("modality")
    except Exception as cls_error:
        classifier_context = (
            "Auxiliary classifier could not run for this image; "
            f"proceed with visual analysis only. Error: {cls_error}"
        )
        print(f"Classifier inference error: {cls_error}")

    modality_text = (
        f"Imaging modality detected: {modality_result.get('modality')} "
        f"({modality_result.get('confidence', 0):.1f}% confidence)."
        if modality_result
        else "Imaging modality not detected automatically; interpret using standard retinal imaging best practices."
    )

    # Create the analysis prompt
    prompt_parts = [
        "You are an expert AI ophthalmology assistant specialized in analyzing eye and retinal images.",
        modality_text,
        "Auxiliary context from a separate image classifier (use as a hint but verify visually):",
        classifier_context or "No auxiliary classifier output is available for this image.",
        "",
        "Analyze this ophthalmic scan (OCT B-scan, OCTA, or fundus photo) and provide a detailed structured report.",
        "Respond with a JSON object containing:",
        '1. "classification": The primary diagnosis or condition (e.g., "Diabetic Retinopathy", "Glaucoma Suspect", "Normal", etc.). Please also consider the results from the auxiliary classifier above.',
        '2. "confidence": Highest probability you infer from the classifier context or visual evidence.',
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
        output_text = _generate_octomed_response(messages, max_new_tokens=2048)
        result = parse_model_response(output_text)

        # Attach classifier context to response for transparency
        if classifier_result:
            result["classifier"] = classifier_result
        if modality_result:
            result["modality"] = modality_result
        if gradcam_image:
            result["gradcam_image"] = gradcam_image
        if gradcam_insights:
            result["gradcam_insights"] = gradcam_insights

        # Council of LLMs to refine the explanation (optional)
        if settings.ENABLE_LLM_COUNCIL:
            try:
                explanation, votes = await _run_llm_council(
                    image_data=image_data,
                    initial_result=result,
                    classifier_context=classifier_context,
                    modality_text=modality_text,
                    gradcam_insights=gradcam_insights,
                )
                result["explanation"] = explanation
                result["council_votes"] = votes
            except Exception as council_error:
                print(f"[council] Deliberation failed: {council_error}")

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
    response_text = _strip_think_tokens(response_text)
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
