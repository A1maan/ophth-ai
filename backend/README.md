# Ophth-AI Backend

FastAPI backend for the Ophth-AI ophthalmology AI assistant.

## Features

- 🏥 **Patient Management**: CRUD operations for patient records
- 🔔 **Notifications**: Real-time notification system
- 🤖 **AI Analysis**: Integration with OctoMed-7B for eye image analysis
- 📊 **Oculomics Data**: Track and analyze eye health metrics
- 🔄 **Workflow Management**: Patient care workflow tracking
- 📂 **Demo Folder Watcher**: Drop OCT/OCTA images into a watch folder and let the system auto-create dummy patients, run AI analysis, and push WebSocket alerts in real time.

## Tech Stack

- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: SQL toolkit and ORM
- **Pydantic**: Data validation using Python type annotations
- **OctoMed-7B**: Vision-Language AI model for medical image analysis
- **PyTorch**: Deep learning framework
- **SQLite/PostgreSQL**: Database (SQLite for dev, PostgreSQL for production)

## Hardware Requirements

⚠️ **Important**: OctoMed-7B is a 7B parameter model and requires significant resources:

- **GPU (Recommended)**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, etc.)
- **CPU Fallback**: 32GB+ RAM (slower inference)
- **Disk Space**: ~15GB for model weights

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch first (with CUDA support if available)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure model loading:

```env
# Set to false if you want to load model on first request instead
LOAD_MODEL_ON_STARTUP=true
```

### 4. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --port 8000

# Or run directly
python main.py
```

The API will be available at `http://localhost:8000`

**Note**: First startup will download the OctoMed-7B model (~14GB) from Hugging Face.

### 5. Demo Folder Watcher

The backend now watches `backend/incoming_scans` (configurable via `WATCH_FOLDER_PATH`) whenever `WATCH_FOLDER_ENABLED=true`.

1. Place any OCT/OCTA image (png/jpg/webp/bmp) inside that folder.
2. The watcher automatically:
   - Generates a dummy patient record with randomized metadata
   - Converts the image to base64 and attaches it to the patient
   - Triggers the AI pipeline and streams the result through WebSockets
3. Processed files are moved to `backend/incoming_scans/processed` so you can keep track of what has been analyzed.

You can tweak polling cadence, allowed extensions, or disable the watcher entirely by editing the related settings in `app/config.py` or your `.env`.

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Patients

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/patients` | List all patients |
| GET | `/api/patients/{id}` | Get patient by ID |
| POST | `/api/patients` | Create new patient |
| PATCH | `/api/patients/{id}` | Update patient |
| DELETE | `/api/patients/{id}` | Delete patient |
| POST | `/api/patients/{id}/analyze` | Trigger AI analysis |
| POST | `/api/patients/{id}/validate` | Validate AI analysis |
| POST | `/api/patients/{id}/schedule` | Schedule appointment |

### Notifications

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/notifications` | List all notifications |
| GET | `/api/notifications/unread-count` | Get unread count |
| POST | `/api/notifications` | Create notification |
| PATCH | `/api/notifications/{id}/read` | Mark as read |
| PATCH | `/api/notifications/mark-all-read` | Mark all as read |
| DELETE | `/api/notifications/{id}` | Delete notification |

### AI Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ai/analyze` | Analyze image (base64) |
| POST | `/api/ai/analyze-upload` | Analyze uploaded image |
| GET | `/api/ai/health` | Check AI service status |
| POST | `/api/ai/load-model` | Manually load AI model |
| POST | `/api/ai/unload-model` | Unload model to free memory |

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .env.example           # Environment template
└── app/
    ├── __init__.py
    ├── config.py          # Settings and configuration
    ├── database.py        # Database connection
    ├── models/
    │   ├── __init__.py
    │   └── models.py      # SQLAlchemy models
    ├── schemas/
    │   ├── __init__.py
    │   └── schemas.py     # Pydantic schemas
    ├── routers/
    │   ├── __init__.py
    │   ├── patients.py    # Patient endpoints
    │   ├── notifications.py # Notification endpoints
    │   └── ai.py          # AI analysis endpoints
    └── services/
        ├── __init__.py
        └── gemini.py      # Gemini AI service
```

## Connecting to Frontend

Update your frontend to use the API:

```typescript
const API_BASE = 'http://localhost:8000/api';

// Fetch patients
const patients = await fetch(`${API_BASE}/patients`).then(r => r.json());

// Create patient
await fetch(`${API_BASE}/patients`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(patientData)
});

// Analyze image
const result = await fetch(`${API_BASE}/ai/analyze`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image_base64: base64Image })
});
```

## Production Deployment

For production, update `.env`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/ophth_ai
DEBUG=false
```

Run with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## License

MIT
