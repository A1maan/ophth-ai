import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Users, 
  Settings, 
  LogOut, 
  Activity, 
  Bell, 
  Menu,
  Plus,
  Wifi,
  WifiOff,
  AlertTriangle
} from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import NotificationDropdown from './NotificationDropdown';

const Layout = () => {
  const { unreadCount, simulateNewCase, realtimeStatus, latestRealtime } = useDashboard();
  const location = useLocation();
  const [isNotifOpen, setIsNotifOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="flex h-screen bg-slate-50 text-slate-800">
      {/* Sidebar */}
      <aside 
        className={`${isSidebarOpen ? 'w-64' : 'w-20'} bg-white border-r border-slate-200 transition-all duration-300 flex flex-col fixed h-full z-20 md:relative`}
      >
        <div className="h-16 flex items-center justify-center border-b border-slate-100">
          <Activity className="w-8 h-8 text-indigo-600" />
          {isSidebarOpen && <span className="ml-3 font-bold text-xl text-slate-800 tracking-tight">OpthAI</span>}
        </div>

        <nav className="flex-1 py-6 px-3 space-y-1">
          <Link to="/" className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${isActive('/') ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-50'}`}>
            <LayoutDashboard className="w-5 h-5" />
            {isSidebarOpen && <span className="ml-3">Dashboard</span>}
          </Link>
          <div className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg text-slate-600 hover:bg-slate-50 cursor-pointer`}>
            <Users className="w-5 h-5" />
            {isSidebarOpen && <span className="ml-3">Patients</span>}
          </div>
          <div className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg text-slate-600 hover:bg-slate-50 cursor-pointer`}>
            <Settings className="w-5 h-5" />
            {isSidebarOpen && <span className="ml-3">Settings</span>}
          </div>
        </nav>

        <div className="p-4 border-t border-slate-100">
          <div className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg text-slate-500 hover:text-red-600 cursor-pointer transition-colors`}>
            <LogOut className="w-5 h-5" />
            {isSidebarOpen && <span className="ml-3">Logout</span>}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Top Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 z-10">
          <div className="flex items-center">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 rounded-md hover:bg-slate-100 text-slate-500 mr-4">
              <Menu className="w-5 h-5" />
            </button>
            <h2 className="text-lg font-semibold text-slate-700">
              {isActive('/') ? 'Doctor Dashboard' : 'Patient Details'}
            </h2>
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={simulateNewCase}
              className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-indigo-50 text-indigo-700 rounded-full text-xs font-medium hover:bg-indigo-100 transition-colors"
            >
              <Plus className="w-3 h-3" />
              Simulate Incoming Case
            </button>

            <div className="relative">
              <button 
                onClick={() => setIsNotifOpen(!isNotifOpen)}
                className="p-2 relative rounded-full hover:bg-slate-100 text-slate-500 transition-colors"
              >
                <Bell className="w-5 h-5" />
                {unreadCount > 0 && (
                  <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 rounded-full ring-2 ring-white"></span>
                )}
              </button>
              <NotificationDropdown isOpen={isNotifOpen} onClose={() => setIsNotifOpen(false)} />
            </div>
            
            <div className={`hidden md:flex items-center gap-2 text-xs font-semibold px-3 py-1 rounded-full border ${realtimeStatus === 'connected' ? 'border-emerald-200 text-emerald-700 bg-emerald-50' : 'border-amber-200 text-amber-700 bg-amber-50'}`}>
              {realtimeStatus === 'connected' ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              <span>{realtimeStatus === 'connected' ? 'Live feed' : 'Waiting for feed'}</span>
            </div>

            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-bold ring-2 ring-indigo-100">
              DR
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto bg-slate-50/50 p-6">
          <Outlet />
        </main>
      </div>

      {latestRealtime && (
        <div className="fixed bottom-6 right-6 w-80 bg-white shadow-2xl border border-red-100 rounded-xl p-4 z-40 animate-pulse">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-red-50 text-red-600">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div>
              <p className="text-xs uppercase font-bold text-red-500">Live Alert</p>
              <p className="text-sm font-semibold text-slate-800">{latestRealtime.message}</p>
              {latestRealtime.label && latestRealtime.confidence !== undefined && (
                <p className="text-xs text-slate-500 mt-1">
                  {latestRealtime.label} • {latestRealtime.confidence.toFixed(1)}% confidence
                </p>
              )}
              {latestRealtime.probabilities && (
                <p className="text-[11px] text-slate-500 mt-1">
                  {Object.entries(latestRealtime.probabilities)
                    .sort((a, b) => Number(b[1]) - Number(a[1]))
                    .slice(0, 3)
                    .map(([k, v]) => `${k}:${Number(v).toFixed(1)}%`)
                    .join(' · ')}
                </p>
              )}
              <p className="text-xs text-slate-500 mt-1">
                {latestRealtime.patientId} • {new Date(latestRealtime.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Layout;
