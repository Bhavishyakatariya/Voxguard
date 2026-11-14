import React from 'react';
import {
  ShieldAlert,
  Activity,
  Eye,
  History,
  Info,
  Cpu,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, backendStatus }) {
  const navItems = [
    { id: 'detection', label: 'Audio Detection', icon: Activity },
    { id: 'explainability', label: 'Explainable AI', icon: Eye },
    { id: 'history', label: 'Analysis History', icon: History },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  return (
    <aside className="w-64 bg-[#0F172A]/80 border-r border-slate-800 flex flex-col justify-between p-5 backdrop-blur-xl select-none">
      <div>
        {/* Brand Header */}
        <div className="flex items-center gap-3 px-2 py-4 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
              VoxGuard
              <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                AI
              </span>
            </h1>
            <p className="text-xs text-slate-400 font-medium">Deepfake Voice Defense</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${isActive
                    ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md shadow-indigo-600/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Model Status Card */}
      <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800/80 shadow-inner">
        <div className="flex items-center gap-2 mb-2">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <span className="text-xs font-semibold text-slate-300">Model Engine</span>
        </div>
        <div className="flex items-center gap-2">
          {backendStatus?.status === 'online' ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-xs text-emerald-400 font-medium">VoxGuardCNN Active</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              <span className="text-xs text-amber-400 font-medium">Backend Offline</span>
            </>
          )}
        </div>
        <div className="mt-2 pt-2 border-t border-slate-800/60 text-[11px] text-slate-400 flex justify-between">
          <span>Grad-CAM:</span>
          <span className="text-indigo-400 font-mono">Hooked (conv4)</span>
        </div>
      </div>
    </aside>
  );
}


