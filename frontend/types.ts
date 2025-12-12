export enum CaseStatus {
  CRITICAL = 'Critical',
  STABLE = 'Stable',
  PENDING = 'Pending Analysis',
  DISCHARGED = 'Discharged'
}

export interface ClassifierResult {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model?: string;
  classifier_type?: string;
}

export interface ModalityResult {
  modality: string;
  confidence: number;
  raw_label?: string;
}

export interface GradCAMInsights {
  focus_region: string;
  focus_center: { x: number; y: number };
  high_activation_percentage: number;
  concentration_score: number;
  interpretation: string;
}

export interface CouncilVote {
  model: string;
  classification?: string;
  confidence?: number;
  findings?: string[];
  recommendation?: string;
  explanation?: string;
  source?: string;
}

export interface AIAnalysisResult {
  classification: string;
  confidence: number;
  findings: string[];
  recommendation: string;
  explanation: string;
  classifier?: ClassifierResult;
  modality?: ModalityResult;
  gradcam_image?: string;
  gradcam_insights?: GradCAMInsights;
  council_votes?: CouncilVote[];
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
  source?: string;
  label?: string;
  confidence?: number;
  probabilities?: Record<string, number>;
  imageData?: string;
}
