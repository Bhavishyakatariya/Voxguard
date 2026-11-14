import React from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  Gauge, 
  Clock, 
  FileCheck2, 
  Sparkles,
  Layers,
  HelpCircle 
} from 'lucide-react';

export default function PredictionResult({ result }) {
  if (!result) return null;

  const isFake = result.verdict === 'FAKE';
  const confidence = result.confidence_pct || (result.confidence * 100);

  return (
    <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-xl mb-8 animate-in fade-in duration-300">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div>
          <span className="text-xs uppercase font-semibold tracking-wider text-indigo-400 block mb-0.5">
            Forensic Verdict
          </span>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            Classification & Confidence Assessment
          </h2>
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-400 font-mono block">Inference Latency</span>
          <span className="text-xs font-semibold text-slate-300 font-mono">
            {result.latency_seconds ? `${result.latency_seconds}s` : '< 0.1s (CPU)'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
        {/* Main Verdict Badge */}
        <div className={`p-6 rounded-2xl border flex flex-col items-center justify-center text-center shadow-lg ${
          isFake 
            ? 'bg-rose-950/20 border-rose-500/40 text-rose-400 shadow-rose-500/10'
            : 'bg-emerald-950/20 border-emerald-500/40 text-emerald-400 shadow-emerald-500/10'
        }`}>
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-3 ${
            isFake ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'
          }`}>
            {isFake ? <ShieldAlert className="w-9 h-9" /> : <ShieldCheck className="w-9 h-9" />}
          </div>
          <span className="text-2xl font-black tracking-tight mb-1">
            {isFake ? 'FAKE VOICE' : 'REAL VOICE'}
          </span>
          <span className="text-xs font-medium text-slate-300">
            {isFake ? 'AI-Synthesized Voice Clone' : 'Authentic Human Speech'}
          </span>
        </div>

        {/* Confidence Meter */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Gauge className="w-4 h-4 text-cyan-400" />
                Detection Confidence
              </span>
              <span className={`text-2xl font-extrabold font-mono ${
                isFake ? 'text-rose-400' : 'text-emerald-400'
              }`}>
                {confidence.toFixed(1)}%
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden mb-3">
              <div 
                className={`h-full rounded-full transition-all duration-700 ${
                  isFake 
                    ? 'bg-gradient-to-r from-rose-500 to-amber-500 shadow-rose-500/50 shadow-md' 
                    : 'bg-gradient-to-r from-emerald-500 to-cyan-400 shadow-emerald-500/50 shadow-md'
                }`}
                style={{ width: `${Math.min(100, Math.max(5, confidence))}%` }}
              ></div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800/80 text-center">
            <div>
              <span className="text-[11px] text-slate-400 block">Class Output</span>
              <span className="text-xs font-mono font-semibold text-slate-200">
                {isFake ? 'Label: 1 (Spoof)' : 'Label: 0 (Real)'}
              </span>
            </div>
            <div>
              <span className="text-[11px] text-slate-400 block">Fake Probability</span>
              <span className="text-xs font-mono font-semibold text-indigo-400">
                P = {result.fake_probability !== undefined ? result.fake_probability : (isFake ? (confidence/100).toFixed(4) : (1 - confidence/100).toFixed(4))}
              </span>
            </div>
            <div>
              <span className="text-[11px] text-slate-400 block">Primary Anomaly Band</span>
              <span className="text-xs font-semibold capitalize text-cyan-400">
                {result.frequency_band || 'Mid-Range'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Forensic Disclaimer */}
      <div className="mt-4 p-3 bg-slate-900/40 rounded-xl border border-slate-800/60 flex items-start gap-2.5">
        <HelpCircle className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
        <p className="text-[11px] text-slate-400 leading-relaxed">
          <strong>Evidentiary Note:</strong> Confidence reflects the neural network's mathematical sigmoid output probability for the predicted class based on spectro-temporal features, not an absolute guarantee. Refer to the Grad-CAM attribution below for verifiable evidence.
        </p>
      </div>
    </div>
  );
}

