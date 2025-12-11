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
  Plus
} from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import NotificationDropdown from './NotificationDropdown';

const Layout = () => {
  const { unreadCount, simulateNewCase } = useDashboard();
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
          {isSidebarOpen && <span className="ml-3 font-bold text-xl text-slate-800 tracking-tight">MediScan</span>}
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
    </div>
  );
};

export default Layout;
