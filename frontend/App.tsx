import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { DashboardProvider } from './contexts/DashboardContext';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import PatientDetails from './components/PatientDetails';

const App: React.FC = () => {
  return (
    <DashboardProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="patient/:id" element={<PatientDetails />} />
          </Route>
        </Routes>
      </Router>
    </DashboardProvider>
  );
};

export default App;
