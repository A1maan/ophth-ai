import google.generativeai as genai
from app.config import settings
import json
import re

# Configure Gemini API
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


async def analyze_eye_image(image_data: str) -> dict:
    """
    Analyze an eye/medical image using Google Gemini AI.
    
    Args:
        image_data: Base64 encoded image data (with or without data URL prefix)
    
    Returns:
        Dictionary with analysis results matching AIAnalysisResult schema
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")
    
    # Remove data URL prefix if present
    clean_base64 = re.sub(r'^data:image/(png|jpeg|jpg|webp);base64,', '', image_data)
    
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create image part for the API
    image_part = {
        "mime_type": "image/png",
        "data": clean_base64
    }
    
    prompt = """You are an expert AI ophthalmology assistant specialized in analyzing eye and retinal images.
    
Analyze this medical eye scan/image and provide a structured report with:

1. **Classification**: The primary diagnosis or condition observed (e.g., "Diabetic Retinopathy", "Glaucoma Suspect", "Age-related Macular Degeneration", "Normal", "Cataracts", etc.)

2. **Confidence**: A confidence score from 0 to 100 indicating how certain you are about the classification

3. **Findings**: A list of specific observations from the image (e.g., "Hemorrhages detected in the macula", "Elevated cup-to-disc ratio", "Drusen deposits visible")

4. **Recommendation**: A clear recommendation for the ophthalmologist (e.g., "Refer to retinal specialist", "Schedule follow-up in 3 months", "Immediate intervention recommended")

5. **Explanation**: A detailed explanation of what you observe and why you made this classification

IMPORTANT: 
- If this is not an eye/medical image, indicate that in the classification as "Invalid Image - Not an Eye Scan"
- Be specific and clinical in your findings
- This is for demonstration purposes only and should not replace professional medical diagnosis

Respond ONLY with a valid JSON object in this exact format:
{
    "classification": "string",
    "confidence": number,
    "findings": ["string", "string"],
    "recommendation": "string",
    "explanation": "string"
}"""
    
    try:
        response = model.generate_content([prompt, image_part])
        
        if response.text:
            # Clean up response text and parse JSON
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["classification", "confidence", "findings", "recommendation", "explanation"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure confidence is a number
            result["confidence"] = float(result["confidence"])
            
            # Ensure findings is a list
            if not isinstance(result["findings"], list):
                result["findings"] = [result["findings"]]
            
            return result
        
        raise ValueError("No response text from Gemini")
        
    except json.JSONDecodeError as e:
        # Return a fallback response if JSON parsing fails
        return {
            "classification": "Analysis Error",
            "confidence": 0,
            "findings": ["Unable to parse AI response"],
            "recommendation": "Please try again or consult with a specialist",
            "explanation": f"The AI model response could not be parsed: {str(e)}"
        }
    except Exception as e:
        raise RuntimeError(f"Gemini analysis failed: {str(e)}")


async def analyze_oculomics(oculomics_data: dict) -> dict:
    """
    Analyze oculomics data and provide risk assessment.
    
    Args:
        oculomics_data: Dictionary containing oculomics measurements
    
    Returns:
        Dictionary with risk assessment
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""You are an expert ophthalmologist. Analyze these oculomics measurements and provide a risk assessment:

Measurements:
- Intraocular Pressure (IOP): {oculomics_data.get('intraocularPressure', 'N/A')} mmHg
- Cup-to-Disc Ratio: {oculomics_data.get('cupToDiscRatio', 'N/A')}
- RNFL Thickness: {oculomics_data.get('rnflThickness', 'N/A')} µm
- Vessel Density: {oculomics_data.get('vesselDensity', 'N/A')} %

Normal ranges for reference:
- IOP: 10-21 mmHg
- Cup-to-Disc Ratio: <0.5
- RNFL Thickness: >80 µm
- Vessel Density: >45%

Provide a risk assessment in JSON format:
{{
    "risk_level": "Low/Medium/High",
    "concerns": ["list of concerns"],
    "recommendations": ["list of recommendations"],
    "summary": "brief summary"
}}"""
    
    try:
        response = model.generate_content(prompt)
        
        if response.text:
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        
        raise ValueError("No response from Gemini")
        
    except Exception as e:
        return {
            "risk_level": "Unknown",
            "concerns": ["Unable to analyze"],
            "recommendations": ["Consult with specialist"],
            "summary": f"Analysis failed: {str(e)}"
        }
