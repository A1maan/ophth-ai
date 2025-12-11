import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Patient, Notification, CaseStatus, OculomicsData } from '../types';

interface DashboardContextType {
  patients: Patient[];
  notifications: Notification[];
  unreadCount: number;
  addPatient: (patient: Patient) => void;
  updatePatient: (id: string, updates: Partial<Patient>) => void;
  markNotificationRead: (id: string) => void;
  simulateNewCase: () => void;
  getPatientById: (id: string) => Patient | undefined;
}

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

  const unreadCount = notifications.filter(n => !n.read).length;

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

    setNotifications(prev => [newNotification, ...prev]);
  };

  const updatePatient = (id: string, updates: Partial<Patient>) => {
    setPatients(prev => prev.map(p => p.id === id ? { ...p, ...updates } : p));
  };

  const markNotificationRead = (id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

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
      addPatient, 
      updatePatient, 
      markNotificationRead,
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