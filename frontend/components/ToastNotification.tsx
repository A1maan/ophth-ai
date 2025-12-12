import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface Toast {
  id: string;
  type: 'critical' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  patientId?: string;
  patientName?: string;
  duration?: number; // auto-dismiss duration in ms (0 = no auto-dismiss)
}

interface ToastNotificationProps {
  toasts: Toast[];
  onDismiss: (id: string) => void;
  onPatientClick?: (patientId: string) => void;
}

const severityConfig = {
  critical: {
    bg: 'bg-red-50',
    border: 'border-red-400',
    icon: '🚨',
    iconBg: 'bg-red-100',
    titleColor: 'text-red-800',
    textColor: 'text-red-700',
    buttonBg: 'bg-red-100 hover:bg-red-200',
    buttonText: 'text-red-800',
  },
  warning: {
    bg: 'bg-yellow-50',
    border: 'border-yellow-400',
    icon: '⚠️',
    iconBg: 'bg-yellow-100',
    titleColor: 'text-yellow-800',
    textColor: 'text-yellow-700',
    buttonBg: 'bg-yellow-100 hover:bg-yellow-200',
    buttonText: 'text-yellow-800',
  },
  info: {
    bg: 'bg-blue-50',
    border: 'border-blue-400',
    icon: 'ℹ️',
    iconBg: 'bg-blue-100',
    titleColor: 'text-blue-800',
    textColor: 'text-blue-700',
    buttonBg: 'bg-blue-100 hover:bg-blue-200',
    buttonText: 'text-blue-800',
  },
  success: {
    bg: 'bg-green-50',
    border: 'border-green-400',
    icon: '✅',
    iconBg: 'bg-green-100',
    titleColor: 'text-green-800',
    textColor: 'text-green-700',
    buttonBg: 'bg-green-100 hover:bg-green-200',
    buttonText: 'text-green-800',
  },
};

const ToastItem: React.FC<{
  toast: Toast;
  onDismiss: (id: string) => void;
  onPatientClick?: (patientId: string) => void;
}> = ({ toast, onDismiss, onPatientClick }) => {
  const config = severityConfig[toast.type];

  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => {
        onDismiss(toast.id);
      }, toast.duration);

      return () => clearTimeout(timer);
    }
  }, [toast.id, toast.duration, onDismiss]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 100, scale: 0.8 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 100, scale: 0.8 }}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      className={`${config.bg} ${config.border} border-l-4 rounded-lg shadow-lg p-4 max-w-md w-full pointer-events-auto`}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={`${config.iconBg} rounded-full p-2 flex-shrink-0`}>
          <span className="text-xl">{config.icon}</span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className={`${config.titleColor} font-semibold text-sm`}>
              {toast.title}
            </h4>
            <button
              onClick={() => onDismiss(toast.id)}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Dismiss notification"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </div>

          <p className={`${config.textColor} text-sm mt-1`}>{toast.message}</p>

          {toast.patientId && toast.patientName && onPatientClick && (
            <button
              onClick={() => onPatientClick(toast.patientId!)}
              className={`${config.buttonBg} ${config.buttonText} text-xs font-medium px-3 py-1.5 rounded-md mt-2 transition-colors`}
            >
              View {toast.patientName}'s Case
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export const ToastContainer: React.FC<ToastNotificationProps> = ({
  toasts,
  onDismiss,
  onPatientClick,
}) => {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3 pointer-events-none">
      <AnimatePresence mode="popLayout">
        {toasts.map((toast) => (
          <ToastItem
            key={toast.id}
            toast={toast}
            onDismiss={onDismiss}
            onPatientClick={onPatientClick}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

// Hook for managing toasts
export const useToasts = () => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const newToast: Toast = {
      ...toast,
      id,
      duration: toast.duration ?? (toast.type === 'critical' ? 0 : 8000), // Critical stays, others auto-dismiss
    };
    
    setToasts((prev) => [...prev, newToast]);
    return id;
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const clearAllToasts = useCallback(() => {
    setToasts([]);
  }, []);

  return {
    toasts,
    addToast,
    dismissToast,
    clearAllToasts,
  };
};

export default ToastContainer;
