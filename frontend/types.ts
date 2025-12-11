export enum CaseStatus {
  CRITICAL = 'Critical',
  STABLE = 'Stable',
  PENDING = 'Pending Analysis',
  DISCHARGED = 'Discharged'
}

export interface AIAnalysisResult {
  classification: string;
  confidence: number;
  findings: string[];
  recommendation: string;
  explanation: string;
}

export interface OculomicsData {
  intraocularPressure: number; // mmHg
  cupToDiscRatio: number;
  rnflThickness: number; // µm
  vesselDensity: number; // %
  history: Array<{ date: string; iop: number; density: number }>;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  admissionDate: string;
  symptoms: string[];
  status: CaseStatus;
  imageUrl?: string; // Base64 or URL
  history: string;
  // Oculomics
  oculomics?: OculomicsData;
  // Workflow fields
  aiAnalysis?: AIAnalysisResult;
  validationStatus?: 'approved' | 'rejected';
  appointmentDate?: string;
  workflowStep?: number; // 1: Analysis, 2: Validation, 3: Scheduling, 4: Complete
}

export interface Notification {
  id: string;
  patientId: string;
  message: string;
  timestamp: Date;
  read: boolean;
  type: 'info' | 'critical' | 'success';
}