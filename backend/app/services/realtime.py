"""
Realtime utilities for broadcasting notification events to connected clients.

This provides a lightweight WebSocket connection manager and a background
classifier watcher that scans the local `db` directory for OCT scans. When an
abnormal result is detected it broadcasts a notification payload to all
connected clients for the hackathon demo.
"""

import asyncio
import base64
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

from fastapi import WebSocket
from PIL import Image

from app.config import settings


class NotificationManager:
    """Track active WebSocket connections and broadcast JSON payloads."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        """Broadcast a JSON message to all connected clients."""
        stale: Set[WebSocket] = set()
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                stale.add(connection)

        for connection in stale:
            self.disconnect(connection)


notification_manager = NotificationManager()


async def classifier_watch_loop(poll_interval: float = 5.0) -> None:
    """
    Simple loop that watches the local db/ folder for OCT scans.

    Any scan classified as non-NORMAL triggers a broadcast notification. The
    classifier logic is left intact; this merely hooks into it to emit events.
    """
    from app.services.classifier import classify_image

    scans_dir = Path(__file__).resolve().parents[2] / "db"
    if not scans_dir.exists():
        print(f"[realtime] No db/ folder found at {scans_dir}, skipping watcher.")
        return

    processed: Dict[str, float] = {}
    print(f"[realtime] Watching {scans_dir} for new scans...")

    while True:
        try:
            for image_path in scans_dir.glob("*"):
                suffix = image_path.suffix.lower()
                if suffix not in {".bmp", ".png", ".jpg", ".jpeg"}:
                    continue

                mtime = image_path.stat().st_mtime
                if processed.get(image_path.name) == mtime:
                    continue  # Skip files we've already handled

                processed[image_path.name] = mtime

                if not settings.ENABLE_CLASSIFIER:
                    print("[realtime] Classifier disabled; skipping notification workflow")
                    continue

                with Image.open(image_path) as img:
                    # Run classifier in a thread to avoid blocking the loop.
                    # Use auto type selection to route fundus vs OCT.
                    result = await asyncio.to_thread(
                        classify_image, img.copy(), "auto"
                    )

                if not result:
                    print(f"[realtime] No classifier result for {image_path.name}")
                    continue

                label = str(result.get("label", "")).lower()
                is_normal = label == "normal"
                print(
                    f"[realtime] Classified {image_path.name} -> {result.get('label')} "
                    f"({result.get('confidence', 0):.1f}%, type={result.get('classifier_type')})"
                )

                if not is_normal:
                    timestamp = datetime.utcnow().isoformat()
                    raw_bytes = image_path.read_bytes()
                    image_data_url = f"data:image/bmp;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"
                    payload = {
                        "id": f"cls-{image_path.stem}-{int(mtime)}",
                        "patientId": image_path.stem,  # File stem acts as identifier for demo
                        "message": (
                            f"Classifier flagged {image_path.name}: "
                            f"{result.get('label')} ({result.get('confidence', 0):.1f}% confidence)"
                        ),
                        "label": result.get("label"),
                        "confidence": result.get("confidence"),
                        "probabilities": result.get("probabilities"),
                        "imageData": image_data_url,
                        "timestamp": timestamp,
                        "read": False,
                        "type": "critical",
                        "source": "classifier",
                    }
                    await notification_manager.broadcast(
                        {"event": "abnormal_scan", "data": payload}
                    )
                    print(f"[realtime] Broadcast abnormal scan alert for {image_path.name}")
                else:
                    print(f"[realtime] {image_path.name} predicted NORMAL; no notification sent")

        except asyncio.CancelledError:
            print("[realtime] Classifier watcher cancelled, shutting down.")
            break
        except Exception as exc:  # pragma: no cover - defensive logging for demo
            print(f"[realtime] Watcher error: {exc}")

        await asyncio.sleep(poll_interval)


async def start_classifier_watcher() -> None:
    """
    Start the classifier watcher and keep it running until cancelled.

    This is separated so it can be started in the FastAPI lifespan hook.
    """
    with suppress(asyncio.CancelledError):
        await classifier_watch_loop()
