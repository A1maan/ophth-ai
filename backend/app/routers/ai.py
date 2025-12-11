from fastapi import APIRouter, HTTPException, UploadFile, File, status
from app.schemas.schemas import AIAnalysisRequest, AIAnalysisResult
from app.services.octomed import analyze_eye_image, is_model_loaded
import base64

router = APIRouter()


@router.post("/analyze", response_model=AIAnalysisResult)
async def analyze_image(request: AIAnalysisRequest):
    """
    Analyze an eye/medical image using OctoMed-7B AI model.
    
    Accepts a base64-encoded image and returns structured analysis results.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not loaded. Please wait for model initialization."
        )
    
    try:
        result = await analyze_eye_image(request.image_base64)
        return AIAnalysisResult(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {str(e)}"
        )


@router.post("/analyze-upload", response_model=AIAnalysisResult)
async def analyze_uploaded_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded eye/medical image using OctoMed-7B AI model.
    
    Accepts an uploaded image file and returns structured analysis results.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not loaded. Please wait for model initialization."
        )
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read and encode file
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")
        
        result = await analyze_eye_image(image_base64)
        return AIAnalysisResult(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {str(e)}"
        )


@router.get("/health")
async def ai_health_check():
    """Check if AI service is available."""
    from app.services.octomed import is_model_loaded, get_device
    
    if not is_model_loaded():
        return {
            "status": "unavailable",
            "message": "OctoMed model is not loaded yet",
            "device": get_device()
        }
    
    return {
        "status": "available",
        "message": "OctoMed-7B AI service is ready",
        "device": get_device()
    }


@router.post("/load-model")
async def load_ai_model():
    """Manually trigger model loading (useful if LOAD_MODEL_ON_STARTUP=false)."""
    from app.services.octomed import load_model, is_model_loaded, get_device
    
    if is_model_loaded():
        return {
            "status": "already_loaded",
            "message": "OctoMed model is already loaded",
            "device": get_device()
        }
    
    try:
        load_model()
        return {
            "status": "loaded",
            "message": "OctoMed-7B model loaded successfully",
            "device": get_device()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/unload-model")
async def unload_ai_model():
    """Unload the model to free memory."""
    from app.services.octomed import unload_model, is_model_loaded
    
    if not is_model_loaded():
        return {
            "status": "not_loaded",
            "message": "Model is not currently loaded"
        }
    
    try:
        unload_model()
        return {
            "status": "unloaded",
            "message": "OctoMed model unloaded successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )
