from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


# Enums matching frontend
class CaseStatus(str, Enum):
    CRITICAL = "Critical"
    STABLE = "Stable"
    PENDING = "Pending Analysis"
    DISCHARGED = "Discharged"


class NotificationType(str, Enum):
    INFO = "info"
    CRITICAL = "critical"
    SUCCESS = "success"


class ValidationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


# Oculomics Schemas
class OculomicsHistoryEntry(BaseModel):
    date: str
    iop: float
    density: float


class OculomicsData(BaseModel):
    intraocularPressure: float = Field(..., description="mmHg")
    cupToDiscRatio: float
    rnflThickness: float = Field(..., description="µm")
    vesselDensity: float = Field(..., description="%")
    history: List[OculomicsHistoryEntry] = []


# AI Analysis Schemas
class ClassifierResult(BaseModel):
    label: str
    confidence: float = Field(..., ge=0, le=100)
    probabilities: Dict[str, float]
    model: Optional[str] = None
    classifier_type: Optional[str] = None


class ModalityResult(BaseModel):
    modality: str
    confidence: float = Field(..., ge=0, le=100)
    raw_label: Optional[str] = None


class GradCAMInsights(BaseModel):
    focus_region: str
    focus_center: Dict[str, int]
    high_activation_percentage: float
    concentration_score: float
    interpretation: str


class AIAnalysisResult(BaseModel):
    classification: str
    confidence: float = Field(..., ge=0, le=100)
    findings: List[str]
    recommendation: str
    explanation: str
    classifier: Optional[ClassifierResult] = None
    modality: Optional[ModalityResult] = None
    gradcam_image: Optional[str] = None
    gradcam_insights: Optional[GradCAMInsights] = None


class AIAnalysisRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")
    patient_id: Optional[str] = None


# Patient Schemas
class PatientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str
    symptoms: List[str] = []
    history: str = ""
    image_url: Optional[str] = None


class PatientCreate(PatientBase):
    status: CaseStatus = CaseStatus.PENDING
    oculomics: Optional[OculomicsData] = None


class PatientUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = None
    symptoms: Optional[List[str]] = None
    status: Optional[CaseStatus] = None
    history: Optional[str] = None
    image_url: Optional[str] = None
    oculomics: Optional[OculomicsData] = None
    ai_analysis: Optional[AIAnalysisResult] = None
    validation_status: Optional[ValidationStatus] = None
    appointment_date: Optional[datetime] = None
    workflow_step: Optional[int] = Field(None, ge=1, le=4)


class PatientResponse(PatientBase):
    id: str
    admission_date: datetime
    status: CaseStatus
    oculomics: Optional[OculomicsData] = None
    ai_analysis: Optional[AIAnalysisResult] = None
    validation_status: Optional[ValidationStatus] = None
    appointment_date: Optional[datetime] = None
    workflow_step: int = 1
    created_at: datetime
    updated_at: datetime
    
    # Alias for frontend compatibility
    @property
    def admissionDate(self) -> str:
        return self.admission_date.isoformat()
    
    @property  
    def imageUrl(self) -> Optional[str]:
        return self.image_url
    
    @property
    def aiAnalysis(self) -> Optional[AIAnalysisResult]:
        return self.ai_analysis
    
    @property
    def validationStatus(self) -> Optional[ValidationStatus]:
        return self.validation_status
    
    @property
    def appointmentDate(self) -> Optional[str]:
        return self.appointment_date.isoformat() if self.appointment_date else None
    
    @property
    def workflowStep(self) -> int:
        return self.workflow_step

    class Config:
        from_attributes = True


# Frontend-compatible Patient response (camelCase)
class PatientResponseCamel(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    admissionDate: str
    symptoms: List[str]
    status: CaseStatus
    imageUrl: Optional[str] = None
    history: str
    oculomics: Optional[OculomicsData] = None
    aiAnalysis: Optional[AIAnalysisResult] = None
    validationStatus: Optional[str] = None
    appointmentDate: Optional[str] = None
    workflowStep: int = 1

    class Config:
        from_attributes = True


# Notification Schemas
class NotificationBase(BaseModel):
    message: str
    type: NotificationType = NotificationType.INFO


class NotificationCreate(NotificationBase):
    patient_id: str


class NotificationResponse(NotificationBase):
    id: str
    patient_id: str
    timestamp: datetime
    read: bool
    
    class Config:
        from_attributes = True


class NotificationResponseCamel(BaseModel):
    id: str
    patientId: str
    message: str
    timestamp: datetime
    read: bool
    type: NotificationType

    class Config:
        from_attributes = True


# Utility Schemas
class HealthCheck(BaseModel):
    status: str = "healthy"
    version: str


class MessageResponse(BaseModel):
    message: str
    success: bool = True
