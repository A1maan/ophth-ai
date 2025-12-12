"""
Folder watcher service that ingests demo images and streams notifications.

The watcher scans a configured directory for new image files, creates dummy
patient records, runs AI analysis, and forwards the structured result via the
existing notification WebSocket service.
"""

import asyncio
import base64
import mimetypes
import random
from pathlib import Path
from typing import Optional, Set, List

from app.config import settings
from app.database import SessionLocal
from app.models.models import Patient, CaseStatus
from app.services.scanner import process_pending_patient

# Watcher state
_watcher_running = False
_watcher_task: Optional[asyncio.Task] = None
_processed_files: Set[str] = set()

# Demo patient metadata
_NAMES = [
    "Alice Walker",
    "Michael Chang",
    "Priya Nair",
    "David Kim",
    "Sarah Connor",
    "Omar Hassan",
    "Lena Hofmann",
    "Lucas Silva",
    "Elena Garcia",
]
_SYMPTOMS = [
    ["Blurred vision", "Eye strain"],
    ["Floaters", "Light sensitivity"],
    ["Headache", "Eye pressure"],
    ["Sudden vision loss"],
    ["Distorted central vision"],
]
_HISTORIES = [
    "Long-standing myopia with regular check-ups.",
    "Type 2 diabetes for 8 years, mild NPDR previously.",
    "Family history of glaucoma; patient under observation.",
    "No significant medical history, routine screening.",
    "Hypertension managed with medication.",
]


def _ensure_watch_dir() -> Path:
    """Create the watch directory if it does not exist."""
    watch_dir = Path(settings.WATCH_FOLDER_PATH).resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    return watch_dir


def _archive_path(watch_dir: Path) -> Path:
    """Directory where processed images are moved."""
    processed_dir = watch_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    return processed_dir


def _allowed_extensions() -> Set[str]:
    """Normalized set of extensions the watcher accepts."""
    return {f".{ext.lower().lstrip('.')}" for ext in settings.WATCH_FOLDER_EXTENSIONS}


def _list_unprocessed_images(watch_dir: Path) -> List[Path]:
    """Return image files that have not been processed yet."""
    allowed = _allowed_extensions()
    files: List[Path] = []
    for path in watch_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        if str(path) in _processed_files:
            continue
        files.append(path)
    return sorted(files)


def _encode_image(file_path: Path) -> str:
    """Encode the image file into a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(file_path.name)
    mime = mime_type or "image/png"
    with file_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _random_patient() -> dict:
    """Build a dummy patient payload."""
    name = random.choice(_NAMES)
    gender = random.choice(["Male", "Female"])
    age = random.randint(25, 75)
    symptoms = random.choice(_SYMPTOMS)
    history = random.choice(_HISTORIES)
    return {
        "name": name,
        "gender": gender,
        "age": age,
        "symptoms": symptoms,
        "history": history,
    }


async def _process_image(file_path: Path):
    """Create a dummy patient for the given image and run analysis."""
    db = SessionLocal()
    try:
        payload = _random_patient()
        patient = Patient(
            name=payload["name"],
            age=payload["age"],
            gender=payload["gender"],
            symptoms=payload["symptoms"],
            history=payload["history"],
            status=CaseStatus.PENDING,
            workflow_step=1,
            image_url=_encode_image(file_path),
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

        print(f"📁 New image ingested from watcher: {file_path.name} -> Patient {patient.name}")
        await process_pending_patient(patient, db)
    except Exception as exc:
        print(f"❌ Failed to process watched image {file_path}: {exc}")
    finally:
        db.close()


async def _watch_loop():
    """Background loop that polls the folder for new images."""
    global _watcher_running

    watch_dir = _ensure_watch_dir()
    archive_dir = _archive_path(watch_dir)
    allowed = ", ".join(sorted(_allowed_extensions()))

    print(f"📁 Folder watcher started on {watch_dir} (extensions: {allowed})")

    while _watcher_running:
        try:
            images = _list_unprocessed_images(watch_dir)
            if images:
                print(f"📁 Found {len(images)} new image(s) to process...")
            for image_path in images:
                await _process_image(image_path)
                _processed_files.add(str(image_path))
                try:
                    destination = archive_dir / image_path.name
                    image_path.rename(destination)
                except Exception as move_error:
                    print(f"⚠️ Could not archive processed file {image_path}: {move_error}")
        except Exception as loop_error:
            print(f"❌ Folder watcher error: {loop_error}")

        await asyncio.sleep(settings.WATCH_FOLDER_POLL_SECONDS)

    print("📁 Folder watcher stopped")


def start_folder_watcher():
    """Launch the folder watcher task."""
    global _watcher_running, _watcher_task

    if not settings.WATCH_FOLDER_ENABLED:
        print("ℹ️ Folder watcher disabled via settings")
        return

    if _watcher_running:
        print("⚠️ Folder watcher already running")
        return

    _watcher_running = True
    _watcher_task = asyncio.create_task(_watch_loop())


def stop_folder_watcher():
    """Stop the folder watcher task."""
    global _watcher_running, _watcher_task

    _watcher_running = False
    if _watcher_task:
        _watcher_task.cancel()
        _watcher_task = None
