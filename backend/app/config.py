from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    APP_NAME: str = "Ophth-AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./ophth_ai.db"
    
    # AI Model Settings
    MODEL_NAME: str = "OctoMed/OctoMed-7B"
    LOAD_MODEL_ON_STARTUP: bool = True  # Set to False to lazy load
    ENABLE_CLASSIFIER: bool = True
    CLASSIFIER_WEIGHTS_PATH: str = "ml/best_weights_classifier.h5"
    CLASSIFIER_LABELS: list[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
    FUNDUS_CLASSIFIER_WEIGHTS_PATH: str = "ml/MobileNetV2_custom.h5"
    FUNDUS_CLASSIFIER_LABELS: list[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
    ENABLE_MODALITY_DETECTOR: bool = True
    MODALITY_CLASSIFIER_PATH: str = "ml/modality_classifier.pth"
    OCT_GRADCAM_LAYER: str = "block5_conv3"
    FUNDUS_GRADCAM_LAYER: str = "Conv_1"
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
