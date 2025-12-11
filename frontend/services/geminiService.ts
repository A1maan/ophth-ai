import { GoogleGenAI, Type } from "@google/genai";
import { AIAnalysisResult } from "../types";

const apiKey = process.env.API_KEY || "";

// Lazily initialize the AI client to avoid errors on page load
let ai: GoogleGenAI | null = null;

const getAIClient = (): GoogleGenAI => {
  if (!apiKey || apiKey === "your_api_key_here") {
    throw new Error(
      "Gemini API Key not configured. Please set GEMINI_API_KEY in frontend/.env file.\n" +
      "Get your API key from: https://aistudio.google.com/apikey"
    );
  }
  if (!ai) {
    ai = new GoogleGenAI({ apiKey });
  }
  return ai;
};

export const analyzeMedicalImage = async (base64Image: string): Promise<AIAnalysisResult> => {
  // Remove data URL prefix if present
  const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, "");

  try {
    const client = getAIClient();
    const response = await client.models.generateContent({
      model: "gemini-2.5-flash",
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/png", // Assuming PNG for simplicity in this demo context
              data: cleanBase64,
            },
          },
          {
            text: `You are an expert AI medical assistant. Analyze this medical scan/image. 
            Provide a structured report with a classification (e.g., Pneumonia, Fracture, Normal, etc.), 
            a confidence score (0-100), key findings, a recommendation for the doctor, and a detailed explanation of what you see.
            
            IMPORTANT: This is for a demonstration. If the image is not a medical image, state that it is invalid.`
          },
        ],
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            classification: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
            findings: { 
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            recommendation: { type: Type.STRING },
            explanation: { type: Type.STRING },
          },
          required: ["classification", "confidence", "findings", "recommendation", "explanation"],
        },
      },
    });

    if (response.text) {
      return JSON.parse(response.text) as AIAnalysisResult;
    }
    throw new Error("No response text from Gemini");
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    throw error;
  }
};
