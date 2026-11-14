import React from 'react';
import { 
  ShieldCheck, 
  Cpu, 
  Layers, 
  Code2, 
  Award, 
  Sparkles,
  BarChart3,
  Server
} from 'lucide-react';

export default function AboutProject() {
  return (
    <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-7 shadow-xl backdrop-blur-xl mb-8 space-y-6">
      <div>
        <span className="text-xs uppercase font-semibold tracking-wider text-indigo-400 block mb-1">
          Academic Major Project
        </span>
        <h2 className="text-2xl font-black text-white flex items-center gap-2">
          VoxGuard: Explainable Deepfake Voice Detection & Forensic Attribution
        </h2>
        <p className="text-sm text-slate-300 mt-2 leading-relaxed">
          VoxGuard is an interpretable deep learning forensic system designed to differentiate between authentic human voices and AI-synthesized clones. Operating across a dual-stream architecture, it pairs high-accuracy neural classification with mathematical Grad-CAM spectro-temporal attribution.
        </p>
      </div>

      {/* 4 Architectural Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm mb-1">
            <Layers className="w-4 h-4" />
            <h3>1. Standardized Acoustic Ingestion</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Resamples audio to 16 kHz Mono, applies silence trimming, peak normalization, and computes 128×128 log-magnitude Mel-filterbank spectrograms conforming to a strict interface contract.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-2 text-cyan-400 font-semibold text-sm mb-1">
            <Cpu className="w-4 h-4" />
            <h3>2. 4-Block Deep 2D CNN Backbone</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Constructed with 4 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool) scaling from 32 to 256 channels, followed by global average pooling and dropout to prevent overfitting.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm mb-1">
            <Sparkles className="w-4 h-4" />
            <h3>3. Grad-CAM Explainability (conv4)</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Intercepts forward activations and backward gradients at the final convolutional layer (`conv4`), computing neuron importance weights to produce bounded time-frequency visual heatmaps.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center gap-2 text-rose-400 font-semibold text-sm mb-1">
            <BarChart3 className="w-4 h-4" />
            <h3>4. Semantic Forensic Captioning</h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Heuristic bounding-box coordinate extractor translates peak attribution clusters into plain-language acoustic reports spanning low (pitch), mid (formant), and high (phase) frequencies.
          </p>
        </div>
      </div>

      {/* Verified Academic Benchmarks */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-950/20 to-slate-900/80 border border-indigo-500/20">
        <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">
          Verified Test Partition Benchmark Results
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] text-slate-400 block">In-The-Wild Accuracy</span>
            <span className="text-base font-bold text-emerald-400 font-mono">96.94%</span>
          </div>
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] text-slate-400 block">In-The-Wild AUC</span>
            <span className="text-base font-bold text-cyan-400 font-mono">0.9951</span>
          </div>
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] text-slate-400 block">In-The-Wild EER</span>
            <span className="text-base font-bold text-indigo-400 font-mono">2.11%</span>
          </div>
          <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Calibrated Test Acc</span>
            <span className="text-base font-bold text-white font-mono">93.03%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

