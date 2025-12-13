import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import patients, notifications, ai
from app.services.realtime import start_classifier_watcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Initialize database
    print("dYs? Starting Ophth-AI Backend...")
    init_db()
    print("ƒo. Database initialized")
    
    # Load AI model on startup if configured
    if settings.LOAD_MODEL_ON_STARTUP:
        print("dYÿ Loading OctoMed-7B AI model (this may take a while)...")
        try:
            from app.services.octomed import load_model
            load_model()
        except Exception as e:
            print(f"ƒsÿ‹,? Failed to load AI model: {e}")
            print("   AI analysis will be unavailable until model is loaded.")
    else:
        print("ƒ,1‹,? AI model will be loaded on first request (LOAD_MODEL_ON_STARTUP=False)")

    watcher_task = None
    try:
        watcher_task = asyncio.create_task(start_classifier_watcher())
    except Exception as exc:
        print(f"[realtime] Could not start classifier watcher: {exc}")
    
    yield
    
    # Shutdown: Cleanup
    print("dY1 Cleaning up resources...")
    try:
        from app.services.octomed import unload_model
        unload_model()
    except Exception:
        pass

    if watcher_task:
        watcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await watcher_task

    print("dY`< Shutting down Ophth-AI Backend...")


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
