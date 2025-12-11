import React from 'react';
import { useDashboard } from '../contexts/DashboardContext';
import { 
  Users, 
  Activity, 
  AlertTriangle, 
  Clock, 
  Search,
  ChevronRight
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { CaseStatus } from '../types';

const data = [
  { name: 'Mon', cases: 4 },
  { name: 'Tue', cases: 3 },
  { name: 'Wed', cases: 7 },
  { name: 'Thu', cases: 5 },
  { name: 'Fri', cases: 8 },
  { name: 'Sat', cases: 6 },
  { name: 'Sun', cases: 9 },
];

const StatCard = ({ title, value, icon: Icon, color, trend }: any) => (
  <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm flex items-start justify-between">
    <div>
      <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
      <h3 className="text-2xl font-bold text-slate-800">{value}</h3>
      <p className={`text-xs mt-2 font-medium ${trend === 'up' ? 'text-green-600' : 'text-slate-400'}`}>
        {trend === 'up' ? '+12.5% from last week' : 'Stable'}
      </p>
    </div>
    <div className={`p-3 rounded-lg ${color}`}>
      <Icon className="w-6 h-6 text-white" />
    </div>
  </div>
);

const Dashboard = () => {
  const { patients } = useDashboard();
  const navigate = useNavigate();

  const criticalCount = patients.filter(p => p.status === CaseStatus.CRITICAL).length;
  const pendingCount = patients.filter(p => p.status === CaseStatus.PENDING).length;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Total Patients" 
          value={patients.length} 
          icon={Users} 
          color="bg-indigo-500" 
          trend="up"
        />
        <StatCard 
          title="Critical Cases" 
          value={criticalCount} 
          icon={AlertTriangle} 
          color="bg-red-500" 
          trend="down"
        />
        <StatCard 
          title="Pending Analysis" 
          value={pendingCount} 
          icon={Clock} 
          color="bg-amber-500" 
          trend="stable"
        />
        <StatCard 
          title="Recovery Rate" 
          value="94%" 
          icon={Activity} 
          color="bg-emerald-500" 
          trend="up"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Table Section */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-100 shadow-sm flex flex-col">
          <div className="p-6 border-b border-slate-100 flex justify-between items-center">
            <h3 className="font-semibold text-slate-800">Recent Cases</h3>
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
              <input 
                type="text" 
                placeholder="Search patients..." 
                className="pl-9 pr-4 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 w-64"
              />
            </div>
          </div>
          
          <div className="overflow-x-auto flex-1">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-slate-500">
                <tr>
                  <th className="px-6 py-4 font-medium">Patient</th>
                  <th className="px-6 py-4 font-medium">Admission Date</th>
                  <th className="px-6 py-4 font-medium">Status</th>
                  <th className="px-6 py-4 font-medium">Symptoms</th>
                  <th className="px-6 py-4 font-medium"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {patients.map((patient) => (
                  <tr 
                    key={patient.id} 
                    className="hover:bg-slate-50/50 transition-colors cursor-pointer"
                    onClick={() => navigate(`/patient/${patient.id}`)}
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-xs font-bold text-slate-600">
                          {patient.name.charAt(0)}
                        </div>
                        <div>
                          <p className="font-medium text-slate-800">{patient.name}</p>
                          <p className="text-xs text-slate-500">ID: {patient.id}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {new Date(patient.admissionDate).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                        ${patient.status === CaseStatus.CRITICAL ? 'bg-red-100 text-red-800' : 
                          patient.status === CaseStatus.STABLE ? 'bg-green-100 text-green-800' : 
                          'bg-amber-100 text-amber-800'}`}>
                        {patient.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-600 max-w-xs truncate">
                      {patient.symptoms.join(', ')}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <ChevronRight className="w-4 h-4 text-slate-400" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Analytics Chart Section */}
        <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-6 flex flex-col">
          <h3 className="font-semibold text-slate-800 mb-6">Weekly Admissions</h3>
          <div className="flex-1 min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorCases" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} />
                <Tooltip 
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                />
                <Area type="monotone" dataKey="cases" stroke="#6366f1" strokeWidth={3} fillOpacity={1} fill="url(#colorCases)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
