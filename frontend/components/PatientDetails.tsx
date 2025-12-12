import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDashboard } from '../contexts/DashboardContext';
import { 
  ArrowLeft, 
  ScanLine, 
  FileText, 
  AlertTriangle, 
  CheckCircle2, 
  Microscope,
  Calendar,
  ClipboardCheck,
  XCircle,
  Clock,
  Check,
  Eye,
  Activity,
  Layers
} from 'lucide-react';
import { analyzeMedicalImage } from '../services/apiService';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

const PatientDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { getPatientById, updatePatient } = useDashboard();
  
  // Local state for analysis process
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meetingDate, setMeetingDate] = useState('');
  const [viewMode, setViewMode] = useState<'original' | 'heatmap'>('original');

  // Get patient from context. If context updates (via updatePatient), this component re-renders.
  const patient = getPatientById(id || '');
  const [imagePreview, setImagePreview] = useState<string | null>(patient?.imageUrl || null);

  // Sync image preview if patient loads late
  useEffect(() => {
    if (patient?.imageUrl && !imagePreview) {
      setImagePreview(patient.imageUrl);
    }
  }, [patient, imagePreview]);

  // Auto-switch to heatmap when analysis is done
  useEffect(() => {
    if (patient?.aiAnalysis && viewMode === 'original') {
      setViewMode('heatmap');
    }
  }, [patient?.aiAnalysis]);

  if (!patient) return <div className="p-8 text-center text-slate-500">Patient not found</div>;

  const currentStep = patient.workflowStep || 1;
  const analysisResult = patient.aiAnalysis;
  const oculomics = patient.oculomics;

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setImagePreview(result);
        // Reset workflow when image changes
        updatePatient(patient.id, { 
          imageUrl: result, 
          aiAnalysis: undefined, 
          workflowStep: 1,
          validationStatus: undefined,
          appointmentDate: undefined
        });
        setViewMode('original');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!imagePreview) return;
    setAnalyzing(true);
    setError(null);
    try {
      let imageToAnalyze = imagePreview;
      if (imagePreview.startsWith('http')) {
        const response = await fetch(imagePreview);
        const blob = await response.blob();
        imageToAnalyze = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        });
      }

      const result = await analyzeMedicalImage(imageToAnalyze);
      
      // Update patient with results and move to step 2
      updatePatient(patient.id, { 
        aiAnalysis: result,
        workflowStep: 2
      });
      
    } catch (err: any) {
      const errorMessage = err?.message || "Failed to analyze image.";
      setError(`Analysis failed: ${errorMessage}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleValidation = (status: 'approved' | 'rejected') => {
    updatePatient(patient.id, {
      validationStatus: status,
      workflowStep: 3
    });
  };

  const handleSchedule = () => {
    if (!meetingDate) return;
    updatePatient(patient.id, {
      appointmentDate: meetingDate,
      workflowStep: 4 // Complete
    });
  };

  // --- Render Helpers ---

  const renderStepper = () => (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4 mb-6">
      <div className="flex items-center justify-between relative">
        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-full h-1 bg-slate-100 -z-0"></div>
        <div className="absolute left-0 top-1/2 transform -translate-y-1/2 h-1 bg-indigo-500 -z-0 transition-all duration-500" 
             style={{ width: `${((currentStep - 1) / 3) * 100}%` }}></div>

        {[
          { step: 1, label: 'Analysis', icon: Microscope },
          { step: 2, label: 'Validation', icon: ClipboardCheck },
          { step: 3, label: 'Schedule', icon: Calendar },
          { step: 4, label: 'Done', icon: Check }
        ].map((item) => {
          const isActive = currentStep >= item.step;
          const isCurrent = currentStep === item.step;
          return (
            <div key={item.step} className="relative z-10 flex flex-col items-center gap-2 bg-white px-2">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 border-2
                ${isActive ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-200 text-slate-400'}
                ${isCurrent ? 'ring-4 ring-indigo-100' : ''}
              `}>
                <item.icon className="w-5 h-5" />
              </div>
              <span className={`text-xs font-medium ${isActive ? 'text-indigo-700' : 'text-slate-400'}`}>
                {item.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button 
          onClick={() => navigate('/')}
          className="p-2 hover:bg-white rounded-full transition-colors text-slate-500"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-slate-800">{patient.name}</h1>
          <p className="text-sm text-slate-500">Patient ID: {patient.id}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Patient Info, Image, Oculomics */}
        <div className="lg:col-span-1 space-y-6">
          {/* Info Card */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6">
            <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-500" />
              Patient Information
            </h3>
            <div className="space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-slate-500">Age</p>
                  <p className="font-medium text-slate-800">{patient.age} yrs</p>
                </div>
                <div>
                  <p className="text-slate-500">Gender</p>
                  <p className="font-medium text-slate-800">{patient.gender}</p>
                </div>
              </div>
              <div className="pt-4 border-t border-slate-100">
                <p className="text-slate-500 mb-1">Symptoms</p>
                <div className="flex flex-wrap gap-2">
                  {patient.symptoms.map(s => (
                    <span key={s} className="px-2 py-1 bg-slate-100 text-slate-600 rounded text-xs">{s}</span>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-slate-500 mb-1">Medical History</p>
                <p className="text-slate-700 leading-relaxed">{patient.history}</p>
              </div>
            </div>
          </div>

          {/* Oculomics Data Card */}
          {oculomics && (
            <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6">
              <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <Eye className="w-4 h-4 text-indigo-500" />
                Oculomics & Biomarkers
              </h3>
              
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">IOP (mmHg)</p>
                  <div className="flex items-end gap-2">
                    <span className="text-xl font-bold text-slate-800">{oculomics.intraocularPressure}</span>
                    <span className={`text-xs mb-1 font-medium ${oculomics.intraocularPressure > 21 ? 'text-red-500' : 'text-emerald-500'}`}>
                      {oculomics.intraocularPressure > 21 ? 'High' : 'Normal'}
                    </span>
                  </div>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">C/D Ratio</p>
                  <div className="flex items-end gap-2">
                    <span className="text-xl font-bold text-slate-800">{oculomics.cupToDiscRatio}</span>
                     <span className={`text-xs mb-1 font-medium ${oculomics.cupToDiscRatio > 0.5 ? 'text-amber-500' : 'text-emerald-500'}`}>
                      {oculomics.cupToDiscRatio > 0.5 ? 'Elevated' : 'Normal'}
                    </span>
                  </div>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">RNFL (µm)</p>
                  <span className="text-xl font-bold text-slate-800">{oculomics.rnflThickness}</span>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <p className="text-xs text-slate-500 mb-1">Vessel Density</p>
                  <span className="text-xl font-bold text-slate-800">{oculomics.vesselDensity}%</span>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-slate-600 mb-3 flex items-center gap-1">
                  <Activity className="w-3 h-3" />
                  IOP History (6 Months)
                </h4>
                <div className="h-32 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={oculomics.history}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                      <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{fontSize: 10, fill: '#94a3b8'}} />
                      <YAxis domain={['dataMin - 5', 'dataMax + 5']} axisLine={false} tickLine={false} tick={{fontSize: 10, fill: '#94a3b8'}} width={20} />
                      <Tooltip 
                        contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '12px'}}
                      />
                      <Line type="monotone" dataKey="iop" stroke="#6366f1" strokeWidth={2} dot={{r: 2}} activeDot={{r: 4}} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Image Upload/Preview Card */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                <ScanLine className="w-4 h-4 text-indigo-500" />
                Medical Imaging
              </h3>
              {analysisResult && (
                <div className="flex bg-slate-100 p-0.5 rounded-lg">
                  <button
                    onClick={() => setViewMode('original')}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                      viewMode === 'original' ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    Original
                  </button>
                  <button
                    onClick={() => setViewMode('heatmap')}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1 ${
                      viewMode === 'heatmap' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    <Layers className="w-3 h-3" />
                    AI Attention
                  </button>
                </div>
              )}
            </div>
            
            <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden relative group mb-4">
              {imagePreview ? (
                <>
                  {/* Show original or Grad-CAM overlay based on view mode */}
                  {viewMode === 'heatmap' && analysisResult?.gradcamImage ? (
                    <img 
                      src={analysisResult.gradcamImage} 
                      alt="Grad-CAM Analysis" 
                      className="w-full h-full object-fill transition-opacity animate-in fade-in duration-500" 
                    />
                  ) : (
                    <img src={imagePreview} alt="Scan" className="w-full h-full object-fill transition-opacity" />
                  )}
                  
                  {/* Fallback simulated Grad-CAM overlay if no real one available */}
                  {viewMode === 'heatmap' && !analysisResult?.gradcamImage && analysisResult && (
                    <div 
                      className="absolute inset-0 opacity-60 mix-blend-overlay pointer-events-none animate-in fade-in duration-700"
                      style={{
                        background: 'radial-gradient(circle at 40% 40%, rgba(255, 0, 0, 0.8) 0%, rgba(255, 255, 0, 0.6) 25%, rgba(0, 255, 0, 0.3) 50%, rgba(0, 0, 255, 0.1) 80%, transparent 100%)',
                        filter: 'blur(10px) contrast(1.2)'
                      }}
                    />
                  )}
                  
                  {/* Heatmap Legend */}
                  {viewMode === 'heatmap' && analysisResult && (
                     <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-md rounded-lg p-3 text-white pointer-events-none animate-in slide-in-from-bottom-2">
                       <p className="text-xs font-semibold mb-1 flex items-center gap-1.5">
                         <Activity className="w-3 h-3 text-red-400" />
                         {analysisResult.gradcamInsights 
                           ? `AI Focus: ${analysisResult.gradcamInsights.focus_region.replace('-', ' ')}`
                           : 'AI Focus Areas'
                         }
                       </p>
                       <div className="h-1.5 w-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 to-red-500 rounded-full"></div>
                       <div className="flex justify-between text-[10px] mt-1 text-slate-300">
                         <span>Low Attention</span>
                         <span>High Attention</span>
                       </div>
                       {analysisResult.gradcamInsights && (
                         <p className="text-[10px] mt-2 text-slate-300 leading-relaxed">
                           {analysisResult.gradcamInsights.interpretation}
                         </p>
                       )}
                     </div>
                  )}

                  <div className={`absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity ${viewMode === 'heatmap' ? 'hidden' : ''}`}>
                    <label className="cursor-pointer bg-white/20 backdrop-blur-sm hover:bg-white/30 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                      Change Image
                      <input type="file" className="hidden" onChange={handleImageUpload} accept="image/*" />
                    </label>
                  </div>
                </>
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-slate-500">
                  <ScanLine className="w-12 h-12 mb-2 opacity-20" />
                  <p className="text-xs">No image uploaded</p>
                </div>
              )}
            </div>
            
            {/* Step 1 Action: Analyze */}
            {currentStep === 1 && (
              <button 
                onClick={handleAnalyze}
                disabled={analyzing || !imagePreview}
                className={`w-full py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 font-medium transition-all
                  ${analyzing ? 'bg-indigo-100 text-indigo-400 cursor-wait' : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-200'}`}
              >
                {analyzing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Microscope className="w-4 h-4" />
                    Run AI Analysis
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Workflow & Results */}
        <div className="lg:col-span-2 space-y-6">
          {renderStepper()}

          {error && (
             <div className="bg-red-50 text-red-700 p-4 rounded-lg flex items-center gap-3 border border-red-100">
               <AlertTriangle className="w-5 h-5 flex-shrink-0" />
               <p>{error}</p>
             </div>
          )}

          {/* Workflow Actions Areas */}
          
          {/* Step 2: Validation */}
          {currentStep === 2 && analysisResult && (
            <div className="bg-white rounded-xl shadow-md border border-indigo-100 p-6 animate-in fade-in slide-in-from-top-4">
              <h3 className="text-lg font-semibold text-slate-800 mb-2 flex items-center gap-2">
                <ClipboardCheck className="w-5 h-5 text-indigo-600" />
                Validate Findings
              </h3>
              <p className="text-slate-600 text-sm mb-6">
                Please review the AI-generated report below. Do you approve the classification of <span className="font-bold text-slate-800">{analysisResult.classification}</span>?
              </p>
              <div className="flex gap-4">
                <button 
                  onClick={() => handleValidation('approved')}
                  className="flex-1 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <CheckCircle2 className="w-5 h-5" />
                  Approve Findings
                </button>
                <button 
                  onClick={() => handleValidation('rejected')}
                  className="flex-1 py-3 bg-white border border-slate-200 hover:bg-red-50 text-slate-700 hover:text-red-600 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <XCircle className="w-5 h-5" />
                  Reject / Flag
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Scheduling */}
          {currentStep === 3 && (
            <div className="bg-white rounded-xl shadow-md border border-indigo-100 p-6 animate-in fade-in slide-in-from-top-4">
              <h3 className="text-lg font-semibold text-slate-800 mb-2 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-indigo-600" />
                Schedule Follow-up
              </h3>
              <div className="mb-4">
                <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-4
                  ${patient.validationStatus === 'approved' ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'}`}>
                  {patient.validationStatus === 'approved' ? <CheckCircle2 className="w-3 h-3"/> : <XCircle className="w-3 h-3"/>}
                  Diagnosis {patient.validationStatus === 'approved' ? 'Approved' : 'Rejected'}
                </div>
                <p className="text-slate-600 text-sm">
                  Please schedule a meeting with the patient to discuss the findings and next steps.
                </p>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-4 items-end">
                <div className="flex-1 w-full">
                  <label className="block text-xs font-medium text-slate-700 mb-1">Meeting Date & Time</label>
                  <input 
                    type="datetime-local" 
                    className="w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                    value={meetingDate}
                    onChange={(e) => setMeetingDate(e.target.value)}
                  />
                </div>
                <button 
                  onClick={handleSchedule}
                  disabled={!meetingDate}
                  className="w-full sm:w-auto py-2.5 px-6 bg-indigo-600 disabled:bg-slate-300 disabled:cursor-not-allowed hover:bg-indigo-700 text-white rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  Confirm Meeting
                </button>
              </div>
            </div>
          )}

          {/* Step 4: Complete */}
          {currentStep === 4 && (
            <div className="bg-emerald-50 rounded-xl border border-emerald-100 p-6 flex items-start gap-4 animate-in fade-in">
              <div className="bg-emerald-100 p-2 rounded-full">
                <Check className="w-6 h-6 text-emerald-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-emerald-900 mb-1">Workflow Complete</h3>
                <p className="text-emerald-800 text-sm mb-2">All stages have been successfully completed.</p>
                <div className="flex flex-wrap gap-4 text-sm mt-3">
                  <div className="flex items-center gap-2 text-emerald-700 bg-white/50 px-3 py-1.5 rounded-md">
                    <ClipboardCheck className="w-4 h-4" />
                    Status: <span className="font-medium capitalize">{patient.validationStatus}</span>
                  </div>
                  <div className="flex items-center gap-2 text-emerald-700 bg-white/50 px-3 py-1.5 rounded-md">
                    <Clock className="w-4 h-4" />
                    Meeting: <span className="font-medium">{new Date(patient.appointmentDate!).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* AI Analysis Display */}
          {!analysisResult && currentStep === 1 && (
            <div className="h-64 bg-slate-50 rounded-xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center text-slate-400 text-center p-8">
              <Microscope className="w-8 h-8 mb-3 opacity-50" />
              <p>Waiting for analysis...</p>
              <p className="text-xs mt-1">Upload an image and click "Run AI Analysis" to begin.</p>
            </div>
          )}

          {analysisResult && (
            <div className={`space-y-6 transition-all duration-500 ${currentStep > 2 ? 'opacity-80' : ''}`}>
              <div className="bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden">
                <div className={`px-6 py-4 border-b border-slate-100 flex items-center justify-between ${
                  analysisResult.classification.toLowerCase().includes('normal') ? 'bg-green-50/50' : 'bg-amber-50/50'
                }`}>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Classification</p>
                    <h2 className="text-xl font-bold text-slate-800">{analysisResult.classification}</h2>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">AI Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-indigo-600 rounded-full" style={{ width: `${analysisResult.confidence}%` }}></div>
                      </div>
                      <span className="text-lg font-bold text-indigo-700">{analysisResult.confidence}%</span>
                    </div>
                  </div>
                </div>

                <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
                   <div>
                     <h4 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                       <ScanLine className="w-4 h-4 text-indigo-500" />
                       Key Findings
                     </h4>
                     <ul className="space-y-2">
                       {analysisResult.findings.map((finding, idx) => (
                         <li key={idx} className="flex items-start gap-2 text-sm text-slate-600">
                           <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0"></div>
                           {finding}
                         </li>
                       ))}
                     </ul>
                   </div>
                   
                   <div>
                     <h4 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                       <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                       Recommendation
                     </h4>
                     <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-4">
                       <p className="text-sm text-emerald-800 font-medium leading-relaxed">
                         {analysisResult.recommendation}
                       </p>
                     </div>
                   </div>
                </div>

                <div className="px-6 pb-6 pt-2">
                  <h4 className="font-semibold text-slate-800 mb-3">Detailed Explanation</h4>
                  <p className="text-sm text-slate-600 leading-relaxed bg-slate-50 p-4 rounded-lg border border-slate-100">
                    {analysisResult.explanation}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatientDetails;