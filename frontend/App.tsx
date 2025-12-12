import React from 'react';
import { HashRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { DashboardProvider, useDashboard } from './contexts/DashboardContext';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import PatientDetails from './components/PatientDetails';
import { ToastContainer } from './components/ToastNotification';

// Separate component for Toast handling with navigation
const ToastManager: React.FC = () => {
  const { toasts, dismissToast } = useDashboard();
  const navigate = useNavigate();

  const handlePatientClick = (patientId: string) => {
    navigate(`/patient/${patientId}`);
  };

  return (
    <ToastContainer
      toasts={toasts}
      onDismiss={dismissToast}
      onPatientClick={handlePatientClick}
    />
  );
};

const AppContent: React.FC = () => {
  return (
    <>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="patient/:id" element={<PatientDetails />} />
        </Route>
      </Routes>
      <ToastManager />
    </>
  );
};

const App: React.FC = () => {
  return (
    <DashboardProvider>
      <Router>
        <AppContent />
      </Router>
    </DashboardProvider>
  );
};

export default App;
