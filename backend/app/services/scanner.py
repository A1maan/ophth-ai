"""
Background Scanner Service for detecting abnormalities in pending patient images.

This service runs as a background task that periodically scans the database
for patients with pending images and runs AI analysis on them.
"""

import asyncio
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models.models import Patient, CaseStatus, Notification, NotificationType
from app.services.websocket_manager import manager


# Scanner state
_scanner_running = False
_scanner_task: Optional[asyncio.Task] = None


def determine_severity(classification: str, confidence: float) -> str:
    """
    Determine notification severity based on AI classification results.
    
    Args:
        classification: The AI classification (e.g., "DME", "Normal")
        confidence: Confidence percentage (0-100)
    
    Returns:
        Severity level: "critical", "warning", "info", or "success"
    """
    # Check if classification indicates abnormality
    is_abnormal = any(
        abnormal.lower() in classification.lower()
        for abnormal in settings.ABNORMAL_CLASSIFICATIONS
    )
    
    # Also check if NOT normal
    is_normal = "normal" in classification.lower() or "healthy" in classification.lower()
    
    if is_abnormal or not is_normal:
        if confidence >= settings.CRITICAL_CONFIDENCE_THRESHOLD:
            return "critical"
        elif confidence >= settings.WARNING_CONFIDENCE_THRESHOLD:
            return "warning"
        else:
            return "info"
    else:
        return "success"


async def analyze_patient_image(patient: Patient, db: Session) -> Optional[dict]:
    """
    Run AI analysis on a patient's image.
    
    Args:
        patient: Patient model instance
        db: Database session
    
    Returns:
        Analysis result dict or None if failed
    """
    from app.services.octomed import analyze_eye_image, is_model_loaded
    
    if not patient.image_url:
        print(f"⚠️ Patient {patient.id} has no image URL")
        return None
    
    if not is_model_loaded():
        print("⚠️ AI model not loaded, skipping analysis")
        return None
    
    try:
        print(f"🔬 Analyzing image for patient: {patient.name} ({patient.id})")
        result = await analyze_eye_image(patient.image_url)
        return result
    except Exception as e:
        print(f"❌ Analysis failed for patient {patient.id}: {e}")
        return None


async def process_pending_patient(patient: Patient, db: Session):
    """
    Process a single pending patient: run analysis and broadcast notification.
    """
    # Run AI analysis
    result = await analyze_patient_image(patient, db)
    
    if not result:
        return
    
    classification = result.get("classification", "Unknown")
    confidence = result.get("confidence", 0)
    findings = result.get("findings", [])
    recommendation = result.get("recommendation", "")
    
    # Determine severity
    severity = determine_severity(classification, confidence)
    
    # Update patient record
    patient.ai_analysis = result
    patient.workflow_step = 2  # Move to validation step
    patient.updated_at = datetime.utcnow()
    
    # Update status based on severity
    if severity == "critical":
        patient.status = CaseStatus.CRITICAL
    elif severity in ["warning", "info"] and patient.status == CaseStatus.PENDING:
        patient.status = CaseStatus.STABLE
    
    db.commit()
    db.refresh(patient)
    
    print(f"✅ Analysis complete for {patient.name}: {classification} ({confidence:.1f}%) - {severity}")
    
    # Create notification in DB only if NOT in demo mode
    if not settings.DEMO_MODE:
        notification = Notification(
            patient_id=patient.id,
            message=f"AI Analysis: {classification} detected for {patient.name} ({confidence:.1f}% confidence)",
            type=NotificationType.CRITICAL if severity == "critical" else (
                NotificationType.SUCCESS if severity == "success" else NotificationType.INFO
            )
        )
        db.add(notification)
        db.commit()
    
    # Broadcast via WebSocket (always, regardless of demo mode)
    notification_type = "abnormality_detected" if severity in ["critical", "warning"] else "analysis_complete"
    
    await manager.broadcast_notification(
        notification_type=notification_type,
        patient_id=patient.id,
        patient_name=patient.name,
        classification=classification,
        confidence=confidence,
        findings=findings,
        recommendation=recommendation,
        severity=severity
    )


async def scan_pending_images():
    """
    Main scanner function that checks for pending patients and processes them.
    """
    db = SessionLocal()
    try:
        # Query for patients that need analysis
        # In demo mode: check workflow_step == 1 (allows re-running)
        # In normal mode: check workflow_step == 1 AND no ai_analysis
        
        query = db.query(Patient).filter(
            Patient.image_url.isnot(None),
            Patient.image_url != "",
            Patient.workflow_step == 1
        )
        
        if not settings.DEMO_MODE:
            # In production, skip patients that already have analysis
            query = query.filter(Patient.ai_analysis.is_(None))
        
        pending_patients = query.all()
        
        if not pending_patients:
            return
        
        print(f"🔍 Scanner found {len(pending_patients)} pending patient(s)")
        
        # Broadcast scanner status
        await manager.broadcast_scanner_status("running", len(pending_patients))
        
        # Process each pending patient
        for patient in pending_patients:
            try:
                await process_pending_patient(patient, db)
            except Exception as e:
                print(f"❌ Error processing patient {patient.id}: {e}")
                continue
        
        # Broadcast scanner idle
        await manager.broadcast_scanner_status("idle", 0)
        
    except Exception as e:
        print(f"❌ Scanner error: {e}")
    finally:
        db.close()


async def scanner_loop():
    """
    Background loop that runs the scanner at configured intervals.
    """
    global _scanner_running
    
    print(f"🚀 Scanner started (interval: {settings.SCANNER_INTERVAL_SECONDS}s)")
    
    while _scanner_running:
        try:
            await scan_pending_images()
        except Exception as e:
            print(f"❌ Scanner loop error: {e}")
        
        # Wait for next scan interval
        await asyncio.sleep(settings.SCANNER_INTERVAL_SECONDS)
    
    print("🛑 Scanner stopped")


def start_scanner():
    """Start the background scanner task."""
    global _scanner_running, _scanner_task
    
    if not settings.SCANNER_ENABLED:
        print("ℹ️ Scanner is disabled in settings")
        return
    
    if _scanner_running:
        print("⚠️ Scanner is already running")
        return
    
    _scanner_running = True
    _scanner_task = asyncio.create_task(scanner_loop())
    print("✅ Scanner task created")


def stop_scanner():
    """Stop the background scanner task."""
    global _scanner_running, _scanner_task
    
    _scanner_running = False
    
    if _scanner_task:
        _scanner_task.cancel()
        _scanner_task = None
    
    print("🛑 Scanner stop requested")


def is_scanner_running() -> bool:
    """Check if scanner is currently running."""
    return _scanner_running


async def trigger_manual_scan():
    """Manually trigger a single scan (useful for testing)."""
    print("🔍 Manual scan triggered")
    await scan_pending_images()
