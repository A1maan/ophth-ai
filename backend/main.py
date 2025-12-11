from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db
from app.routers import patients, notifications, ai


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Initialize database
    print("🚀 Starting Ophth-AI Backend...")
    init_db()
    print("✅ Database initialized")
    
    # Load AI model on startup if configured
    if settings.LOAD_MODEL_ON_STARTUP:
        print("🧠 Loading OctoMed-7B AI model (this may take a while)...")
        try:
            from app.services.octomed import load_model
            load_model()
        except Exception as e:
            print(f"⚠️ Failed to load AI model: {e}")
            print("   AI analysis will be unavailable until model is loaded.")
    else:
        print("ℹ️ AI model will be loaded on first request (LOAD_MODEL_ON_STARTUP=False)")
    
    yield
    
    # Shutdown: Cleanup
    print("🧹 Cleaning up resources...")
    try:
        from app.services.octomed import unload_model
        unload_model()
    except Exception:
        pass
    print("👋 Shutting down Ophth-AI Backend...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API for Ophth-AI - Ophthalmology AI Assistant",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    patients.router,
    prefix="/api/patients",
    tags=["Patients"]
)

app.include_router(
    notifications.router,
    prefix="/api/notifications",
    tags=["Notifications"]
)

app.include_router(
    ai.router,
    prefix="/api/ai",
    tags=["AI Analysis"]
)


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "patients": "/api/patients",
            "notifications": "/api/notifications",
            "ai": "/api/ai",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
