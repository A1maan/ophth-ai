import React, { createContext, useContext, useState, useEffect, useMemo, ReactNode } from 'react';
import { Patient, Notification, CaseStatus, OculomicsData } from '../types';

type RealtimeStatus = 'connecting' | 'connected' | 'disconnected';

interface DashboardContextType {
  patients: Patient[];
  notifications: Notification[];
  unreadCount: number;
  realtimeStatus: RealtimeStatus;
  latestRealtime: Notification | null;
  autoAnalyzePatientId: string | null;
  addPatient: (patient: Patient) => void;
  updatePatient: (id: string, updates: Partial<Patient>) => void;
  markNotificationRead: (id: string) => void;
  requestAutoAnalyze: (patientId: string) => void;
  clearAutoAnalyze: () => void;
  simulateNewCase: () => void;
  getPatientById: (id: string) => Patient | undefined;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_URL = `${API_BASE_URL.replace(/^http/i, 'ws')}/api/notifications/ws`;

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

const NAMES = ["James Smith", "Maria Garcia", "Robert Johnson", "Lisa Chen", "David Kim", "Sarah Connor"];
const SYMPTOMS_LIST = [
  ["Cough", "Fever"], 
  ["Chest Pain", "Shortness of breath"], 
  ["Headache", "Dizziness"], 
  ["Abdominal Pain"], 
  ["Fracture", "Swelling"]
];

const generateOculomicsData = (isCritical: boolean): OculomicsData => {
  const history = [];
  const today = new Date();
  for (let i = 5; i >= 0; i--) {
    const d = new Date(today);
    d.setMonth(d.getMonth() - i);
    // Random fluctuation
    const baseIop = isCritical ? 24 : 16;
    history.push({
      date: d.toLocaleDateString('en-US', { month: 'short' }),
      iop: Math.floor(baseIop + (Math.random() * 4 - 2)),
      density: Math.floor(isCritical ? 40 : 50 + (Math.random() * 5))
    });
  }

  return {
    intraocularPressure: history[history.length - 1].iop,
    cupToDiscRatio: isCritical ? 0.65 : 0.35,
    rnflThickness: isCritical ? 75 : 95,
    vesselDensity: isCritical ? 42 : 52,
    history
  };
};

const INITIAL_PATIENTS: Patient[] = [
  {
    id: 'p-101',
    name: 'Alice Walker',
    age: 45,
    gender: 'Female',
    admissionDate: new Date().toISOString(),
    symptoms: ['Persistent Cough', 'Fatigue'],
    status: CaseStatus.PENDING,
    history: 'No prior major surgeries. Smoker for 5 years.',
    imageUrl: 'https://picsum.photos/400/400?grayscale',
    workflowStep: 1,
    oculomics: generateOculomicsData(false)
  },
  {
    id: 'p-102',
    name: 'Michael Chang',
    age: 62,
    gender: 'Male',
    admissionDate: new Date(Date.now() - 86400000).toISOString(),
    symptoms: ['Chest Pressure', 'Nausea'],
    status: CaseStatus.CRITICAL,
    history: 'Hypertension, Diabetes Type 2.',
    imageUrl: 'https://picsum.photos/400/400?grayscale&blur=1',
    workflowStep: 1,
    oculomics: generateOculomicsData(true)
  }
];

interface DashboardProviderProps {
  children: ReactNode;
}

export const DashboardProvider = ({ children }: DashboardProviderProps) => {
  const [patients, setPatients] = useState<Patient[]>(INITIAL_PATIENTS);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>('disconnected');
  const [latestRealtime, setLatestRealtime] = useState<Notification | null>(null);
  const [autoAnalyzePatientId, setAutoAnalyzePatientId] = useState<string | null>(null);

  const unreadCount = useMemo(
    () => notifications.filter(n => !n.read).length,
    [notifications]
  );

  const normalizeNotification = (payload: any): Notification => {
    const probabilities = payload.probabilities
      ? Object.fromEntries(
          Object.entries(payload.probabilities).map(([k, v]) => [k, Number(v)])
        )
      : undefined;

    return {
      id: payload.id ?? `notif-${Date.now()}`,
      patientId: payload.patientId ?? payload.patient_id ?? 'unassigned',
      message: payload.message ?? 'New notification',
      timestamp: payload.timestamp ? new Date(payload.timestamp) : new Date(),
      read: Boolean(payload.read),
      type: (payload.type as Notification['type']) || 'info',
      source: payload.source,
      label: payload.label,
      confidence: payload.confidence ? Number(payload.confidence) : undefined,
      probabilities,
      imageData: payload.imageData || payload.image_data,
    };
  };

  const pushNotification = (notification: Notification) => {
    setNotifications(prev => {
      if (prev.some(n => n.id === notification.id)) return prev;
      return [notification, ...prev];
    });
  };

  const buildIncomingPatient = (notification: Notification): Patient => {
    const name = NAMES[Math.floor(Math.random() * NAMES.length)];
    const symptoms = SYMPTOMS_LIST[Math.floor(Math.random() * SYMPTOMS_LIST.length)];
    const isCritical = (notification.label && notification.label.toLowerCase() !== 'normal') || notification.type === 'critical';
    const gender = Math.random() > 0.5 ? 'Male' : 'Female';

    return {
      id: notification.patientId || `p-${Date.now().toString().slice(-4)}`,
      name,
      age: 20 + Math.floor(Math.random() * 60),
      gender,
      admissionDate: new Date().toISOString(),
      symptoms,
      status: isCritical ? CaseStatus.CRITICAL : CaseStatus.PENDING,
      history: 'Auto-created from incoming scan. Run AI analysis to confirm.',
      imageUrl: notification.imageData || `https://picsum.photos/400/400?grayscale&random=${Date.now()}`,
      workflowStep: 1,
      oculomics: generateOculomicsData(isCritical),
    };
  };

  const ensurePatientFromNotification = (notification: Notification) => {
    if (!notification.patientId) return;
    setPatients(prev => {
      if (prev.some(p => p.id === notification.patientId)) return prev;

      const placeholder = buildIncomingPatient(notification);
      return [placeholder, ...prev];
    });
  };

  useEffect(() => {
    // Prime UI with existing notifications from the backend
    const fetchExisting = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/notifications?limit=20`);
        if (!response.ok) return;
        const data = await response.json();
        const hydrated = data.map((item: any) => normalizeNotification(item));
        setNotifications(hydrated);
      } catch (err) {
        console.warn('Unable to preload notifications', err);
      }
    };

    fetchExisting();
  }, []);

  useEffect(() => {
    // Connect to backend WebSocket for live classifier alerts
    let socket: WebSocket | null = null;
    setRealtimeStatus('connecting');

    try {
      socket = new WebSocket(WS_URL);
    } catch (err) {
      console.warn('Unable to open WebSocket', err);
      setRealtimeStatus('disconnected');
      return;
    }

    socket.onopen = () => setRealtimeStatus('connected');
    socket.onclose = () => setRealtimeStatus('disconnected');
    socket.onerror = () => setRealtimeStatus('disconnected');

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const payload = message?.data ?? message;

        if (message?.event === 'abnormal_scan' || message?.event === 'notification') {
          const incoming = normalizeNotification(payload);
          pushNotification(incoming);
          ensurePatientFromNotification(incoming);
          setLatestRealtime(incoming);
        }
      } catch (err) {
        console.warn('Failed to parse websocket message', err);
      }
    };

    return () => socket?.close();
  }, []);

  useEffect(() => {
    if (!latestRealtime) return;
    const timer = setTimeout(() => setLatestRealtime(null), 6000);
    return () => clearTimeout(timer);
  }, [latestRealtime]);

  const addPatient = (patient: Patient) => {
    setPatients(prev => [patient, ...prev]);
    
    const newNotification: Notification = {
      id: `notif-${Date.now()}`,
      patientId: patient.id,
      message: `New case admitted: ${patient.name} (${patient.status})`,
      timestamp: new Date(),
      read: false,
      type: patient.status === CaseStatus.CRITICAL ? 'critical' : 'info'
    };

    pushNotification(newNotification);
  };

  const updatePatient = (id: string, updates: Partial<Patient>) => {
    setPatients(prev => prev.map(p => p.id === id ? { ...p, ...updates } : p));
  };

  const markNotificationRead = (id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const requestAutoAnalyze = (patientId: string) => setAutoAnalyzePatientId(patientId);
  const clearAutoAnalyze = () => setAutoAnalyzePatientId(null);

  const getPatientById = (id: string) => patients.find(p => p.id === id);

  const simulateNewCase = () => {
    const id = `p-${Date.now().toString().slice(-4)}`;
    const name = NAMES[Math.floor(Math.random() * NAMES.length)];
    const symptoms = SYMPTOMS_LIST[Math.floor(Math.random() * SYMPTOMS_LIST.length)];
    const isCritical = Math.random() > 0.7;
    
    const newPatient: Patient = {
      id,
      name,
      age: 20 + Math.floor(Math.random() * 60),
      gender: Math.random() > 0.5 ? 'Male' : 'Female',
      admissionDate: new Date().toISOString(),
      symptoms,
      status: isCritical ? CaseStatus.CRITICAL : CaseStatus.PENDING,
      history: 'No significant history available yet.',
      imageUrl: 'https://picsum.photos/400/400?grayscale&random=' + id,
      workflowStep: 1,
      oculomics: generateOculomicsData(isCritical)
    };

    addPatient(newPatient);
  };

  // Simulate a new case every 60 seconds automatically
  useEffect(() => {
    const interval = setInterval(() => {
      // simulateNewCase(); // Uncomment to enable auto-simulation
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <DashboardContext.Provider value={{ 
      patients, 
      notifications, 
      unreadCount,
      realtimeStatus,
      latestRealtime,
      autoAnalyzePatientId,
      addPatient, 
      updatePatient, 
      markNotificationRead,
      requestAutoAnalyze,
      clearAutoAnalyze,
      simulateNewCase,
      getPatientById
    }}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (!context) throw new Error("useDashboard must be used within DashboardProvider");
  return context;
};
