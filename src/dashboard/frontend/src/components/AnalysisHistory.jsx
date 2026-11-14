import React from 'react';
import { 
  History, 
  FileAudio, 
  CheckCircle2, 
  XCircle, 
  ExternalLink,
  Clock
} from 'lucide-react';

export default function AnalysisHistory({ history, onSelectRecord }) {
  if (!history || history.length === 0) {
    return (
      <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-6 text-center text-slate-400">
        <History className="w-8 h-8 text-slate-600 mx-auto mb-2" />
        <p className="text-sm">No analysis history recorded in this session yet.</p>
        <p className="text-xs text-slate-500 mt-1">Upload an audio recording to populate the audit log.</p>
      </div>
    );
  }

  return (
    <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-xl mb-8">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <History className="w-5 h-5 text-indigo-400" />
            Audit Trail & Analysis History
          </h2>
          <p className="text-xs text-slate-400">
            Chronological records of all voice trials analyzed during this session
          </p>
        </div>
        <span className="text-xs text-slate-400 bg-slate-800 px-2.5 py-1 rounded-full font-mono">
          {history.length} Trials
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
              <th className="py-3 px-3">File Name</th>
              <th className="py-3 px-3">Verdict</th>
              <th className="py-3 px-3">Confidence</th>
              <th className="py-3 px-3">Band Focus</th>
              <th className="py-3 px-3">Timestamp</th>
              <th className="py-3 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-sans">
            {history.map((record, index) => {
              const isFake = record.verdict === 'FAKE';
              return (
                <tr key={index} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3 px-3 font-medium text-slate-200 flex items-center gap-2">
                    <FileAudio className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                    <span className="truncate max-w-xs">{record.filename}</span>
                  </td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-bold tracking-tight inline-flex items-center gap-1 ${
                      isFake 
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' 
                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    }`}>
                      {isFake ? <XCircle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                      {record.verdict}
                    </span>
                  </td>
                  <td className="py-3 px-3 font-mono font-semibold text-slate-300">
                    {(record.confidence_pct || record.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 capitalize text-slate-400">
                    {record.frequency_band || 'Mid-range'}
                  </td>
                  <td className="py-3 px-3 text-slate-500 font-mono">
                    {record.timestamp || 'Recent'}
                  </td>
                  <td className="py-3 px-3 text-right">
                    <button
                      onClick={() => onSelectRecord(record)}
                      className="text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center gap-1 hover:underline"
                    >
                      <span>View</span>
                      <ExternalLink className="w-3 h-3" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

