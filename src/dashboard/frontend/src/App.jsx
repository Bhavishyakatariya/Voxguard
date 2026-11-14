import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar.jsx';
import AudioUploadSection from './components/AudioUploadSection.jsx';
import PredictionResult from './components/PredictionResult.jsx';
import XAIVisualization from './components/XAIVisualization.jsx';
import AnalysisHistory from './components/AnalysisHistory.jsx';
import AboutProject from './components/AboutProject.jsx';
import { 
  Activity, 
  ShieldCheck, 
  AlertCircle, 
  RefreshCw, 
  CheckCircle2, 
  XCircle,
  FileAudio
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('detection');
  const [backendStatus, setBackendStatus] = useState(null);
  const [availableSamples, setAvailableSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);

  // Poll backend health & samples on load
  const fetchBackendData = async () => {
    try {
      const resHealth = await fetch('/api/health');
      if (resHealth.ok) {
        const dataHealth = await resHealth.json();
        setBackendStatus(dataHealth);
      } else {
        setBackendStatus({ status: 'offline' });
      }

      const resSamples = await fetch('/api/samples');
      if (resSamples.ok) {
        const dataSamples = await resSamples.json();
        setAvailableSamples(dataSamples.samples || []);
      }

      const resHist = await fetch('/api/history');
      if (resHist.ok) {
        const dataHist = await resHist.json();
        setHistory(dataHist.history || []);
      }
    } catch (err) {
      setBackendStatus({ status: 'offline' });
    }
  };

  useEffect(() => {
    fetchBackendData();
    const interval = setInterval(fetchBackendData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async (source) => {
    setIsLoading(true);
    setErrorMessage(null);
    setCurrentResult(null);

    try {
      let response;
      if (source.type === 'upload') {
        const formData = new FormData();
        formData.append('file', source.file);
        response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData,
        });
      } else {
        response = await fetch('/api/analyze/sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: source.filename }),
        });
      }

      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.message || 'Inference pipeline failure');
      }

      setCurrentResult(data);
      // Refresh history
      const resHist = await fetch('/api/history');
      if (resHist.ok) {
        const dataHist = await resHist.json();
        setHistory(dataHist.history || []);
      }
    } catch (err) {
      setErrorMessage(err.message || 'Failed to communicate with VoxGuard inference service.');
    } finally {
      setIsLoading(false);
    }
  };

  const realCount = history.filter(h => h.verdict === 'REAL').length;
  const fakeCount = history.filter(h => h.verdict === 'FAKE').length;

  return (
    <div className="flex h-screen bg-[#0B0F17] overflow-hidden select-none">
      {/* Sidebar Navigation */}
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        backendStatus={backendStatus} 
      />

      {/* Main Content Viewport */}
      <main className="flex-1 overflow-y-auto p-8 relative">
        {/* Top Header & Session Telemetry */}
        <header className="flex flex-col md:flex-row md:items-center justify-between pb-6 mb-6 border-b border-slate-800 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
                Major Project Presentation Build
              </span>
            </div>
            <h1 className="text-3xl font-black text-white tracking-tight">
              VoxGuard Forensic Dashboard
            </h1>
            <p className="text-sm text-slate-400 mt-0.5">
              Explainable deepfake voice detection powered by trained 2D CNN & Grad-CAM attribution
            </p>
          </div>

          {/* Session Overview Stats */}
          <div className="flex items-center gap-3">
            <div className="px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-[10px] text-slate-400 uppercase font-semibold block">Total Trials</span>
              <span className="text-lg font-bold text-white font-mono">{history.length}</span>
            </div>
            <div className="px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-[10px] text-emerald-400 uppercase font-semibold block">Real Voices</span>
              <span className="text-lg font-bold text-emerald-400 font-mono">{realCount}</span>
            </div>
            <div className="px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <span className="text-[10px] text-rose-400 uppercase font-semibold block">Fake Clones</span>
              <span className="text-lg font-bold text-rose-400 font-mono">{fakeCount}</span>
            </div>
          </div>
        </header>

        {/* Global Error Banner */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded-xl bg-rose-950/40 border border-rose-500/50 flex items-start gap-3 text-rose-300 text-sm">
            <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="block font-semibold">Inference Error</strong>
              <span>{errorMessage}</span>
            </div>
          </div>
        )}

        {/* Tab 1: Audio Detection */}
        {activeTab === 'detection' && (
          <div>
            <AudioUploadSection
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
              availableSamples={availableSamples}
              selectedSample={selectedSample}
              setSelectedSample={setSelectedSample}
            />

            {/* Display Prediction Result */}
            <PredictionResult result={currentResult} />

            {/* Display Grad-CAM Explainability */}
            <XAIVisualization result={currentResult} />
          </div>
        )}

        {/* Tab 2: Explainable AI Focus */}
        {activeTab === 'explainability' && (
          <div>
            {currentResult ? (
              <XAIVisualization result={currentResult} />
            ) : (
              <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-10 text-center text-slate-400">
                <Activity className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                <h3 className="text-base font-semibold text-slate-200">No Analysis Available Yet</h3>
                <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                  Run an audio analysis in the Audio Detection tab to generate and inspect the Grad-CAM attribution heatmap here.
                </p>
                <button
                  onClick={() => setActiveTab('detection')}
                  className="mt-4 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold"
                >
                  Go to Audio Detection
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Analysis History */}
        {activeTab === 'history' && (
          <AnalysisHistory 
            history={history} 
            onSelectRecord={(rec) => {
              setCurrentResult(rec);
              setActiveTab('detection');
            }} 
          />
        )}

        {/* Tab 4: About Project */}
        {activeTab === 'about' && (
          <AboutProject />
        )}
      </main>
    </div>
  );
}

