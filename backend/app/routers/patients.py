from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app.database import get_db
from app.models.models import Patient, CaseStatus, Notification, NotificationType
from app.schemas.schemas import (
    PatientCreate, 
    PatientUpdate, 
    PatientResponseCamel,
    MessageResponse
)

router = APIRouter()


def patient_to_camel(patient: Patient) -> dict:
    """Convert SQLAlchemy Patient to camelCase dict for frontend."""
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "admissionDate": patient.admission_date.isoformat() if patient.admission_date else None,
        "symptoms": patient.symptoms or [],
        "status": patient.status.value if patient.status else CaseStatus.PENDING.value,
        "imageUrl": patient.image_url,
        "history": patient.history or "",
        "oculomics": patient.oculomics,
        "aiAnalysis": patient.ai_analysis,
        "validationStatus": patient.validation_status.value if patient.validation_status else None,
        "appointmentDate": patient.appointment_date.isoformat() if patient.appointment_date else None,
        "workflowStep": patient.workflow_step or 1,
    }


@router.get("/", response_model=List[PatientResponseCamel])
async def get_patients(
    skip: int = 0, 
    limit: int = 100, 
    status: str = None,
    db: Session = Depends(get_db)
):
    """Get all patients with optional filtering."""
    query = db.query(Patient)
    
    if status:
        query = query.filter(Patient.status == status)
    
    patients = query.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()
    return [patient_to_camel(p) for p in patients]


@router.get("/{patient_id}", response_model=PatientResponseCamel)
async def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Get a specific patient by ID."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    return patient_to_camel(patient)


@router.post("/", response_model=PatientResponseCamel, status_code=status.HTTP_201_CREATED)
async def create_patient(patient_data: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient."""
    new_patient = Patient(
        name=patient_data.name,
        age=patient_data.age,
        gender=patient_data.gender,
        symptoms=patient_data.symptoms,
        status=CaseStatus(patient_data.status.value),
        history=patient_data.history,
        image_url=patient_data.image_url,
        oculomics=patient_data.oculomics.model_dump() if patient_data.oculomics else None,
        workflow_step=1,
    )
    
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    
    # Create notification for new patient
    notification = Notification(
        patient_id=new_patient.id,
        message=f"New case admitted: {new_patient.name} ({new_patient.status.value})",
        type=NotificationType.CRITICAL if new_patient.status == CaseStatus.CRITICAL else NotificationType.INFO
    )
    db.add(notification)
    db.commit()
    
    return patient_to_camel(new_patient)


@router.patch("/{patient_id}", response_model=PatientResponseCamel)
async def update_patient(
    patient_id: str, 
    patient_data: PatientUpdate, 
    db: Session = Depends(get_db)
):
    """Update a patient's information."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    
    update_data = patient_data.model_dump(exclude_unset=True)
    
    # Handle nested objects
    if "oculomics" in update_data and update_data["oculomics"]:
        update_data["oculomics"] = update_data["oculomics"]
    if "ai_analysis" in update_data and update_data["ai_analysis"]:
        update_data["ai_analysis"] = update_data["ai_analysis"]
    
    for field, value in update_data.items():
        if hasattr(patient, field):
            setattr(patient, field, value)
    
    patient.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(patient)
    
    return patient_to_camel(patient)


@router.delete("/{patient_id}", response_model=MessageResponse)
async def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    """Delete a patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    
    db.delete(patient)
    db.commit()
    
    return MessageResponse(message=f"Patient {patient_id} deleted successfully")


@router.post("/{patient_id}/analyze", response_model=PatientResponseCamel)
async def trigger_ai_analysis(patient_id: str, db: Session = Depends(get_db)):
    """Trigger AI analysis for a patient's image."""
    from app.services.gemini import analyze_eye_image
    
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    
    if not patient.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient has no image to analyze"
        )
    
    try:
        analysis_result = await analyze_eye_image(patient.image_url)
        patient.ai_analysis = analysis_result
        patient.workflow_step = 2  # Move to validation step
        patient.updated_at = datetime.utcnow()
        
        # Create notification
        notification = Notification(
            patient_id=patient.id,
            message=f"AI analysis completed for {patient.name}: {analysis_result.get('classification', 'Unknown')}",
            type=NotificationType.SUCCESS
        )
        db.add(notification)
        db.commit()
        db.refresh(patient)
        
        return patient_to_camel(patient)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {str(e)}"
        )


@router.post("/{patient_id}/validate")
async def validate_analysis(
    patient_id: str, 
    approved: bool,
    db: Session = Depends(get_db)
):
    """Validate or reject AI analysis."""
    from app.models.models import ValidationStatus
    
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    
    patient.validation_status = ValidationStatus.APPROVED if approved else ValidationStatus.REJECTED
    patient.workflow_step = 3 if approved else 1  # Move to scheduling or back to analysis
    patient.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(patient)
    
    return patient_to_camel(patient)


@router.post("/{patient_id}/schedule")
async def schedule_appointment(
    patient_id: str,
    appointment_date: datetime,
    db: Session = Depends(get_db)
):
    """Schedule a follow-up appointment."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    
    patient.appointment_date = appointment_date
    patient.workflow_step = 4  # Complete
    patient.updated_at = datetime.utcnow()
    
    # Create notification
    notification = Notification(
        patient_id=patient.id,
        message=f"Appointment scheduled for {patient.name} on {appointment_date.strftime('%Y-%m-%d %H:%M')}",
        type=NotificationType.SUCCESS
    )
    db.add(notification)
    db.commit()
    db.refresh(patient)
    
    return patient_to_camel(patient)


@router.post("/{patient_id}/reset-analysis", response_model=PatientResponseCamel)
async def reset_patient_analysis(patient_id: str, db: Session = Depends(get_db)):
    """
    Reset a patient's analysis state for re-scanning (Demo Mode).
    This clears the AI analysis and resets workflow_step to 1.
    """
    from app.config import settings
    
    if not settings.DEMO_MODE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reset analysis is only available in demo mode"
        )
    
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with id {patient_id} not found"
        )
    
    # Reset analysis state
    patient.ai_analysis = None
    patient.validation_status = None
    patient.workflow_step = 1
    patient.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(patient)
    
    return patient_to_camel(patient)


@router.post("/reset-all-analysis", response_model=MessageResponse)
async def reset_all_analysis(db: Session = Depends(get_db)):
    """
    Reset all patients' analysis state for re-scanning (Demo Mode).
    This clears AI analysis and resets workflow_step to 1 for all patients.
    """
    from app.config import settings
    
    if not settings.DEMO_MODE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reset analysis is only available in demo mode"
        )
    
    # Reset all patients
    patients = db.query(Patient).all()
    count = 0
    for patient in patients:
        if patient.workflow_step != 1 or patient.ai_analysis:
            patient.ai_analysis = None
            patient.validation_status = None
            patient.workflow_step = 1
            patient.updated_at = datetime.utcnow()
            count += 1
    
    db.commit()
    
    return MessageResponse(message=f"Reset {count} patient(s) for re-analysis")
