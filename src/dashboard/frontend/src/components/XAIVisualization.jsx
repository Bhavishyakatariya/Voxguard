import React from 'react';
import { 
  Eye, 
  Sparkles, 
  Activity, 
  Compass, 
  Sliders, 
  ExternalLink,
  MessageSquareQuote
} from 'lucide-react';

export default function XAIVisualization({ result }) {
  if (!result || !result.heatmap_url) return null;

  return (
    <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-xl mb-8 animate-in fade-in duration-300">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div>
          <span className="text-xs uppercase font-semibold tracking-wider text-cyan-400 block mb-0.5">
            Explainable AI (XAI)
          </span>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Eye className="w-5 h-5 text-indigo-400" />
            AI Explanation — Grad-CAM Heatmap
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700/60 font-mono">
            Layer: conv4 (128×256)
          </span>
        </div>
      </div>

      {/* Generated High-Resolution Heatmap Image */}
      <div className="mb-6 rounded-xl overflow-hidden border border-slate-800 bg-[#0F141C] shadow-2xl relative group">
        <img 
          src={result.heatmap_url} 
          alt="VoxGuard Grad-CAM Heatmap"
          className="w-full h-auto object-contain max-h-[460px] mx-auto transition-transform duration-300"
          onError={(e) => {
            // Fallback if image path needs full host
            if (!e.target.src.startsWith('http://127.0.0.1:8000')) {
              e.target.src = `http://127.0.0.1:8000${result.heatmap_url}`;
            }
          }}
        />
        <div className="p-3 bg-slate-900/90 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-cyan-400" />
            <strong>Interpretation:</strong> Red/Yellow regions identify acoustic anomalies that exerted primary decision influence.
          </span>
          <a 
            href={result.heatmap_url.startsWith('http') ? result.heatmap_url : `http://127.0.0.1:8000${result.heatmap_url}`} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
          >
            <span>Open 300-DPI</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>

      {/* Automated Forensic Caption */}
      <div className="p-5 rounded-xl bg-gradient-to-r from-indigo-950/30 via-slate-900/70 to-cyan-950/20 border border-indigo-500/30">
        <div className="flex items-center gap-2 mb-2 text-indigo-400 font-semibold text-sm">
          <MessageSquareQuote className="w-4 h-4" />
          <h3>Why did VoxGuard make this prediction?</h3>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed font-sans pl-6 border-l-2 border-indigo-500/50">
          "{result.caption}"
        </p>

        {/* Coordinate Metadata Pills */}
        <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap gap-2 text-xs">
          {result.time_interval_seconds && (
            <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700/60 font-mono">
              ⏱️ Time: {result.time_interval_seconds[0]}s – {result.time_interval_seconds[1]}s
            </span>
          )}
          {result.frequency_range_hz && (
            <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700/60 font-mono">
              📊 Freq: {(result.frequency_range_hz[0]/1000).toFixed(1)} kHz – {(result.frequency_range_hz[1]/1000).toFixed(1)} kHz
            </span>
          )}
          {result.attribution_strength && (
            <span className="px-2.5 py-1 rounded bg-slate-800 text-cyan-300 border border-slate-700/60 font-mono">
              🔥 Attribution: {(result.attribution_strength * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

