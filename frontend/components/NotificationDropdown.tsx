import React from 'react';
import { Bell, AlertCircle, Info, CheckCircle } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import { useNavigate } from 'react-router-dom';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

const NotificationDropdown: React.FC<Props> = ({ isOpen, onClose }) => {
  const { notifications, markNotificationRead, requestAutoAnalyze } = useDashboard();
  const navigate = useNavigate();

  if (!isOpen) return null;

  const handleClick = (notif: any) => {
    markNotificationRead(notif.id);
    requestAutoAnalyze(notif.patientId);
    navigate(`/patient/${notif.patientId}`);
    onClose();
  };

  return (
    <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-xl border border-slate-100 z-50 overflow-hidden">
      <div className="bg-slate-50 px-4 py-3 border-b border-slate-100 flex justify-between items-center">
        <h3 className="text-sm font-semibold text-slate-700">Notifications</h3>
        <span className="text-xs text-slate-500">{notifications.length} Total</span>
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="p-6 text-center text-slate-400 text-sm">
            No new notifications
          </div>
        ) : (
          notifications.map((notif) => (
            <div 
              key={notif.id}
              onClick={() => handleClick(notif)}
              className={`p-4 border-b border-slate-50 cursor-pointer hover:bg-slate-50 transition-colors ${!notif.read ? 'bg-blue-50/30' : ''}`}
            >
              <div className="flex items-start gap-3">
                <div className="mt-1">
                  {notif.type === 'critical' ? (
                    <AlertCircle className="w-4 h-4 text-red-500" />
                  ) : notif.type === 'success' ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : (
                    <Info className="w-4 h-4 text-blue-500" />
                  )}
                </div>
                <div>
                  <p className={`text-sm ${!notif.read ? 'font-semibold text-slate-800' : 'text-slate-600'}`}>
                    {notif.message}
                  </p>
                  {notif.confidence !== undefined && notif.label && (
                    <p className="text-xs text-slate-500 mt-1">
                      {notif.label} • {notif.confidence.toFixed(1)}% confidence
                    </p>
                  )}
                  {notif.probabilities && (
                    <p className="text-[11px] text-slate-500 mt-1">
                      {Object.entries(notif.probabilities)
                        .sort((a, b) => Number(b[1]) - Number(a[1]))
                        .slice(0, 3)
                        .map(([k, v]) => `${k}:${Number(v).toFixed(1)}%`)
                        .join(' · ')}
                    </p>
                  )}
                  <p className="text-xs text-slate-400 mt-1">
                    {new Date(notif.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default NotificationDropdown;
