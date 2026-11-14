import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileAudio, 
  Play, 
  Pause, 
  Trash2, 
  Sparkles, 
  ArrowRight,
  ShieldCheck,
  Radio
} from 'lucide-react';

export default function AudioUploadSection({ 
  onAnalyze, 
  isLoading, 
  availableSamples, 
  selectedSample, 
  setSelectedSample 
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioDuration, setAudioDuration] = useState(null);
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setSelectedSample(null);
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setIsPlaying(false);
    }
  };

  const handleSampleSelect = (sampleFilename) => {
    if (!sampleFilename) return;
    setSelectedSample(sampleFilename);
    setSelectedFile(null);
    setAudioUrl(`/api/audio/sample/${sampleFilename}`);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setSelectedSample(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setIsPlaying(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onAnalyze({ type: 'upload', file: selectedFile });
    } else if (selectedSample) {
      onAnalyze({ type: 'sample', filename: selectedSample });
    }
  };

  return (
    <div className="bg-[#111827]/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-xl mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Radio className="w-5 h-5 text-indigo-400" />
            Audio Source Selection
          </h2>
          <p className="text-xs text-slate-400">
            Upload voice audio (.wav, .flac, .mp3) or choose from genuine test benchmarks
          </p>
        </div>
        {(selectedFile || selectedSample) && (
          <button
            onClick={handleClear}
            className="text-xs text-slate-400 hover:text-red-400 flex items-center gap-1 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear selection
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Option A: Drag & Drop File Upload */}
        <div 
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
            selectedFile
              ? 'border-indigo-500 bg-indigo-950/20 shadow-md shadow-indigo-500/10'
              : 'border-slate-700/80 hover:border-indigo-500/60 bg-slate-900/50 hover:bg-slate-900/80'
          }`}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".wav,.flac,.mp3,.ogg" 
            className="hidden" 
          />
          <div className="w-12 h-12 rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-3">
            <UploadCloud className="w-6 h-6" />
          </div>
          <span className="text-sm font-semibold text-slate-200 mb-1">
            {selectedFile ? selectedFile.name : 'Upload Audio File'}
          </span>
          <span className="text-xs text-slate-400">
            {selectedFile 
              ? `${(selectedFile.size / 1024).toFixed(1)} KB (Ready for forensic analysis)` 
              : 'Drag & drop or click to browse (.wav, .flac, .mp3)'}
          </span>
        </div>

        {/* Option B: Pre-loaded Benchmark Samples */}
        <div className="border border-slate-800 rounded-xl p-5 bg-slate-900/50 flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block mb-2">
              Or Select Verified Sample
            </span>
            <p className="text-xs text-slate-400 mb-4">
              Inspect authentic human speech or AI-synthesized clones directly from the test partition
            </p>
          </div>

          <div className="space-y-2">
            {availableSamples && availableSamples.length > 0 ? (
              availableSamples.map((sample) => (
                <button
                  key={sample.filename}
                  onClick={() => handleSampleSelect(sample.filename)}
                  className={`w-full text-left px-3.5 py-2.5 rounded-lg text-xs font-medium flex items-center justify-between transition-all ${
                    selectedSample === sample.filename
                      ? 'bg-indigo-600 text-white font-semibold shadow-md'
                      : 'bg-slate-800/60 hover:bg-slate-800 text-slate-300 border border-slate-700/60'
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <FileAudio className="w-4 h-4 text-cyan-400" />
                    {sample.display_name} ({sample.filename})
                  </span>
                  <span className="text-[10px] text-slate-400 uppercase font-mono">16 kHz</span>
                </button>
              ))
            ) : (
              <div className="text-xs text-slate-500 italic p-3 bg-slate-800/30 rounded-lg">
                No sample audios found in data/sample_audios/
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Audio Player Bar (if audio file loaded) */}
      {audioUrl && (
        <div className="p-3.5 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-between mb-6">
          <audio 
            ref={audioRef} 
            src={audioUrl} 
            onEnded={() => setIsPlaying(false)}
            onLoadedMetadata={(e) => setAudioDuration(e.target.duration)}
          />
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlayback}
              className="w-9 h-9 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center transition-colors shadow-md"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div>
              <span className="text-xs font-medium text-slate-200 block truncate max-w-xs">
                {selectedFile?.name || selectedSample}
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                {audioDuration ? `${audioDuration.toFixed(2)}s duration` : 'Audio loaded'}
              </span>
            </div>
          </div>
          <span className="text-xs text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20 font-medium">
            Format: Standard PCM
          </span>
        </div>
      )}

      {/* Primary Action Button */}
      <button
        disabled={isLoading || (!selectedFile && !selectedSample)}
        onClick={handleSubmit}
        className={`w-full py-3.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 shadow-lg transition-all duration-200 ${
          isLoading || (!selectedFile && !selectedSample)
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
            : 'bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white shadow-indigo-500/25 cursor-pointer transform hover:-translate-y-0.5'
        }`}
      >
        {isLoading ? (
          <>
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
            <span>Extracting Mel-Spectrogram & Executing Grad-CAM...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4 text-cyan-200" />
            <span>Analyze Voice Recording</span>
            <ArrowRight className="w-4 h-4" />
          </>
        )}
      </button>
    </div>
  );
}

