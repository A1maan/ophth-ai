import { AIAnalysisResult } from "../types";

// Backend API URL - adjust if your backend runs on a different port
const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

/**
 * Check if the AI service is available and model is loaded
 */
export const checkAIHealth = async (): Promise<{
  status: string;
  message: string;
  device?: string;
}> => {
  const response = await fetch(`${API_BASE_URL}/api/ai/health`);
  if (!response.ok) {
    throw new Error("Failed to check AI health");
  }
  return response.json();
};

/**
 * Analyze a medical image using the OctoMed backend API
 * @param base64Image - Base64 encoded image (with or without data URL prefix)
 * @returns Analysis result with classification, confidence, findings, etc.
 */
export const analyzeMedicalImage = async (base64Image: string): Promise<AIAnalysisResult> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/ai/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image_base64: base64Image,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      
      if (response.status === 503) {
        throw new Error(
          "AI model is not loaded yet. Please wait for the server to initialize or call /api/ai/load-model"
        );
      }
      
      throw new Error(errorData.detail || `Analysis failed with status ${response.status}`);
    }

    const result = await response.json();
    return result as AIAnalysisResult;
  } catch (error) {
    console.error("OctoMed Analysis Error:", error);
    throw error;
  }
};

/**
 * Analyze an uploaded image file using the OctoMed backend API
 * @param file - Image file to analyze
 * @returns Analysis result
 */
export const analyzeUploadedImage = async (file: File): Promise<AIAnalysisResult> => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/api/ai/analyze-upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      
      if (response.status === 503) {
        throw new Error(
          "AI model is not loaded yet. Please wait for the server to initialize."
        );
      }
      
      throw new Error(errorData.detail || `Analysis failed with status ${response.status}`);
    }

    const result = await response.json();
    return result as AIAnalysisResult;
  } catch (error) {
    console.error("OctoMed Upload Analysis Error:", error);
    throw error;
  }
};

/**
 * Manually trigger model loading on the backend
 */
export const loadAIModel = async (): Promise<{
  status: string;
  message: string;
  device?: string;
}> => {
  const response = await fetch(`${API_BASE_URL}/api/ai/load-model`, {
    method: "POST",
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to load AI model");
  }
  
  return response.json();
};

/**
 * Unload the AI model to free memory
 */
export const unloadAIModel = async (): Promise<{
  status: string;
  message: string;
}> => {
  const response = await fetch(`${API_BASE_URL}/api/ai/unload-model`, {
    method: "POST",
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to unload AI model");
  }
  
  return response.json();
};
