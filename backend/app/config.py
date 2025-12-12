from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    APP_NAME: str = "Ophth-AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Demo Mode - notifications not persisted, allows re-analysis
    DEMO_MODE: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./ophth_ai.db"
    
    # AI Model Settings
    MODEL_NAME: str = "OctoMed/OctoMed-7B"
    LOAD_MODEL_ON_STARTUP: bool = True  # Set to False to lazy load
    ENABLE_CLASSIFIER: bool = True
    
    # VGG16 OCT Classifier (4 classes)
    OCT_CLASSIFIER_WEIGHTS_PATH: str = "ml/best_weights_classifier.h5"
    OCT_CLASSIFIER_LABELS: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
    OCT_GRADCAM_LAYER: str = "block5_conv3"
    
    # MobileNetV2 Fundus Classifier (8 classes)
    FUNDUS_CLASSIFIER_WEIGHTS_PATH: str = "ml/MobileNetV2_custom.h5"
    FUNDUS_CLASSIFIER_LABELS: List[str] = ["AMD", "Cataract", "Diabetes", "Glaucoma", "Hypertension", "Myopia", "Normal", "Other"]
    FUNDUS_GRADCAM_LAYER: str = "block_16_project"
    
    # Default classifier (can be "oct" or "fundus")
    DEFAULT_CLASSIFIER: str = "oct"
    
    # Modality Detector (ResNet18) - auto-detects OCT vs Fundus
    MODALITY_CLASSIFIER_PATH: str = "ml/modality_classifier.pth"
    ENABLE_MODALITY_DETECTION: bool = True  # Auto-detect image type
    MODALITY_CONFIDENCE_THRESHOLD: float = 70.0  # Min confidence to trust detection
    
    # Legacy aliases for backward compatibility
    CLASSIFIER_WEIGHTS_PATH: str = "ml/best_weights_classifier.h5"
    CLASSIFIER_LABELS: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # Scanner/Cron Job Settings
    SCANNER_ENABLED: bool = True
    SCANNER_INTERVAL_SECONDS: int = 30  # How often to scan for pending images
    
    # Abnormality Detection Thresholds
    CRITICAL_CONFIDENCE_THRESHOLD: float = 80.0
    WARNING_CONFIDENCE_THRESHOLD: float = 60.0
    ABNORMAL_CLASSIFICATIONS: List[str] = [
        "CNV", "DME", "DRUSEN",
        "Diabetic Retinopathy", "Glaucoma", "Macular Degeneration",
        "Choroidal Neovascularization", "Diabetic Macular Edema"
    ]
    
    # CORS
    CORS_ORIGINS: List[str] = [
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
