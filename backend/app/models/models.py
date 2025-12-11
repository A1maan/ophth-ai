from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from app.database import Base


class CaseStatus(str, enum.Enum):
    CRITICAL = "Critical"
    STABLE = "Stable"
    PENDING = "Pending Analysis"
    DISCHARGED = "Discharged"


class NotificationType(str, enum.Enum):
    INFO = "info"
    CRITICAL = "critical"
    SUCCESS = "success"


class ValidationStatus(str, enum.Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


def generate_uuid():
    return str(uuid.uuid4())


class Patient(Base):
    """Patient database model."""
    __tablename__ = "patients"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    admission_date = Column(DateTime, default=datetime.utcnow)
    symptoms = Column(JSON, default=list)  # List of symptoms
    status = Column(SQLEnum(CaseStatus), default=CaseStatus.PENDING)
    image_url = Column(Text, nullable=True)  # Base64 or URL
    history = Column(Text, default="")
    
    # Oculomics data (stored as JSON for flexibility)
    oculomics = Column(JSON, nullable=True)
    
    # AI Analysis result (stored as JSON)
    ai_analysis = Column(JSON, nullable=True)
    
    # Workflow fields
    validation_status = Column(SQLEnum(ValidationStatus), nullable=True)
    appointment_date = Column(DateTime, nullable=True)
    workflow_step = Column(Integer, default=1)  # 1: Analysis, 2: Validation, 3: Scheduling, 4: Complete
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    notifications = relationship("Notification", back_populates="patient", cascade="all, delete-orphan")


class Notification(Base):
    """Notification database model."""
    __tablename__ = "notifications"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    read = Column(Boolean, default=False)
    type = Column(SQLEnum(NotificationType), default=NotificationType.INFO)
    
    # Relationships
    patient = relationship("Patient", back_populates="notifications")


class OculomicsHistory(Base):
    """Oculomics history for tracking over time."""
    __tablename__ = "oculomics_history"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    intraocular_pressure = Column(Float)  # mmHg
    cup_to_disc_ratio = Column(Float)
    rnfl_thickness = Column(Float)  # µm
    vessel_density = Column(Float)  # %
