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
    CLASSIFIER_WEIGHTS_PATH: str = "ml/best_weights_classifier.h5"
    CLASSIFIER_LABELS: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # Scanner/Cron Job Settings
    SCANNER_ENABLED: bool = True
    SCANNER_INTERVAL_SECONDS: int = 30  # How often to scan for pending images
    
    # Folder watcher for demo ingestion
    WATCH_FOLDER_ENABLED: bool = True
    WATCH_FOLDER_PATH: str = "./incoming_scans"
    WATCH_FOLDER_POLL_SECONDS: int = 15
    WATCH_FOLDER_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "bmp", "webp"]
    
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
