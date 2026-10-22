import React, { useState, useEffect } from 'react';
import {
  Satellite,
  BarChart3,
  LineChart,
  Layers,
  Sparkles,
  Info,
  Compass,
  Search,
  Filter,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
  ChevronRight,
  ShieldCheck,
  Cpu,
  Eye,
  Grid
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [resultsData, setResultsData] = useState(null);
  const [curvesData, setCurvesData] = useState(null);
  const [patchData, setPatchData] = useState(null);
  const [metaData, setMetaData] = useState(null);
  const [gradcamData, setGradcamData] = useState(null);
  const [pipelineData, setPipelineData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Overview Tab State
  const [splitMode, setSplitMode] = useState('primary'); // 'primary' | 'clean'

  // Curves Tab State
  const [selectedModelRun, setSelectedModelRun] = useState('CustomCNN_seed2024');

  // Patches Tab State
  const [patchFilterLabel, setPatchFilterLabel] = useState('all'); // 'all' | '1' | '0'
  const [patchFilterStrip, setPatchFilterStrip] = useState('all'); // 'all' | '6' | '10' | '14'
  const [patchFilterAmbiguity, setPatchFilterAmbiguity] = useState('all'); // 'all' | 'clean' | 'ambiguous'
  const [patchFilterOutcome, setPatchFilterOutcome] = useState('all'); // 'all' | 'TP' | 'FP' | 'FN' | 'TN'
  const [patchSelectedModel, setPatchSelectedModel] = useState('CustomCNN_seed2024');
  const [selectedPatch, setSelectedPatch] = useState(null);

  // Grad-CAM Tab State
  const [gradcamCategory, setGradcamCategory] = useState('failures'); // 'failures' | 'tp' | 'fp' | 'fn' | 'tn'

  // Pipeline Tab State
  const [selectedStage, setSelectedStage] = useState(1);

  // Metadata Modal
  const [showMetaModal, setShowMetaModal] = useState(false);

  useEffect(() => {
    async function loadAllData() {
      try {
        setLoading(true);
        const [resRes, resCurv, resPatches, resMeta, resGrad, resPipe] = await Promise.all([
          fetch('/data/results.json').then((r) => r.json()),
          fetch('/data/training_curves.json').then((r) => r.json()),
          fetch('/data/per_patch_scores.json').then((r) => r.json()),
          fetch('/data/meta.json').then((r) => r.json()),
          fetch('/data/gradcam/manifest.json').then((r) => r.json()),
          fetch('/data/pipeline_stages/manifest.json').then((r) => r.json()),
        ]);
        setResultsData(resRes);
        setCurvesData(resCurv);
        setPatchData(resPatches);
        setMetaData(resMeta);
        setGradcamData(resGrad);
        setPipelineData(resPipe);
        if (resPatches?.patches?.length > 0) {
          setSelectedPatch(resPatches.patches[217] || resPatches.patches[0]);
        }
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadAllData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-space-900 text-slate-200">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-400 mb-4"></div>
        <p className="text-lg font-medium text-cyan-300">Loading IceShadowNet Radar Benchmark Data...</p>
        <p className="text-xs text-slate-400 mt-2">Ingesting Chandrayaan-2 DFSAR L1A Quad-Pol SLI records</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-space-900 text-red-400 p-6">
        <AlertTriangle className="h-12 w-12 text-rose-500 mb-3" />
        <h2 className="text-xl font-bold">Failed to load benchmark data</h2>
        <p className="text-sm text-slate-400 mt-1">{error}</p>
        <p className="text-xs text-slate-500 mt-4">Make sure data is exported via <code>python -m scripts.analysis.export_dashboard_data</code></p>
      </div>
    );
  }

  const runs = resultsData?.runs || [];
  const summaryByConfig = resultsData?.summary_by_config || {};
  const baselines = resultsData?.baselines || {};
  const trivial = resultsData?.trivial_baselines || {};

  return (
    <div className="min-h-screen flex flex-col bg-space-900 text-slate-100">
      {/* Top Header */}
      <header className="border-b border-space-700 bg-space-800/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-cyan-950/70 border border-cyan-500/40 rounded-lg text-cyan-400 shadow-inner">
              <Satellite className="h-6 w-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-tight">IceShadowNet</span>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-full">
                  Chandrayaan-2 DFSAR Real Benchmark
                </span>
              </div>
              <p className="text-xs text-slate-400">Lunar South Polar Subsurface Ice Characterization via Quad-Pol Radar</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowMetaModal(true)}
              className="flex items-center space-x-1 px-3 py-1.5 text-xs font-medium rounded-lg bg-space-700 hover:bg-space-600 border border-space-600 text-slate-200 transition"
            >
              <Info className="h-4 w-4 text-cyan-400" />
              <span>Methodology & Baselines</span>
            </button>
            <div className="hidden sm:flex items-center space-x-2 px-3 py-1 bg-space-900 border border-space-700 rounded-md text-xs font-mono text-emerald-400">
              <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></div>
              <span>N=338 Spatial Test Split</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 overflow-x-auto scrollbar-none border-t border-space-700/60 text-xs font-medium">
          <button
            onClick={() => setActiveTab('overview')}
            className={`py-3 px-4 border-b-2 flex items-center space-x-2 whitespace-nowrap transition ${
              activeTab === 'overview'
                ? 'border-cyan-400 text-cyan-400 font-semibold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="h-4 w-4" />
            <span>Benchmark Overview</span>
          </button>

          <button
            onClick={() => setActiveTab('curves')}
            className={`py-3 px-4 border-b-2 flex items-center space-x-2 whitespace-nowrap transition ${
              activeTab === 'curves'
                ? 'border-cyan-400 text-cyan-400 font-semibold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <LineChart className="h-4 w-4" />
            <span>Training Curves</span>
          </button>

          <button
            onClick={() => setActiveTab('patches')}
            className={`py-3 px-4 border-b-2 flex items-center space-x-2 whitespace-nowrap transition ${
              activeTab === 'patches'
                ? 'border-cyan-400 text-cyan-400 font-semibold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Grid className="h-4 w-4" />
            <span>Patch Explorer (338 Patches)</span>
          </button>

          <button
            onClick={() => setActiveTab('gradcam')}
            className={`py-3 px-4 border-b-2 flex items-center space-x-2 whitespace-nowrap transition ${
              activeTab === 'gradcam'
                ? 'border-cyan-400 text-cyan-400 font-semibold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="h-4 w-4" />
            <span>Grad-CAM Interpretability</span>
          </button>

          <button
            onClick={() => setActiveTab('pipeline')}
            className={`py-3 px-4 border-b-2 flex items-center space-x-2 whitespace-nowrap transition ${
              activeTab === 'pipeline'
                ? 'border-cyan-400 text-cyan-400 font-semibold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-4 w-4" />
            <span>Radar Pipeline Stages</span>
          </button>

          <button
            onClick={() => setActiveTab('per_strip')}
            className={`py-3 px-4 border-b-2 flex items-center space-x-2 whitespace-nowrap transition ${
              activeTab === 'per_strip'
                ? 'border-cyan-400 text-cyan-400 font-semibold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="h-4 w-4" />
            <span>Per-Strip Analysis</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {activeTab === 'overview' && (
          <OverviewTab
            runs={runs}
            summaryByConfig={summaryByConfig}
            baselines={baselines}
            trivial={trivial}
            splitMode={splitMode}
            setSplitMode={setSplitMode}
          />
        )}

        {activeTab === 'curves' && (
          <CurvesTab
            curvesData={curvesData}
            selectedModelRun={selectedModelRun}
            setSelectedModelRun={setSelectedModelRun}
          />
        )}

        {activeTab === 'patches' && (
          <PatchesTab
            patchData={patchData}
            patchFilterLabel={patchFilterLabel}
            setPatchFilterLabel={setPatchFilterLabel}
            patchFilterStrip={patchFilterStrip}
            setPatchFilterStrip={setPatchFilterStrip}
            patchFilterAmbiguity={patchFilterAmbiguity}
            setPatchFilterAmbiguity={setPatchFilterAmbiguity}
            patchFilterOutcome={patchFilterOutcome}
            setPatchFilterOutcome={setPatchFilterOutcome}
            patchSelectedModel={patchSelectedModel}
            setPatchSelectedModel={setPatchSelectedModel}
            selectedPatch={selectedPatch}
            setSelectedPatch={setSelectedPatch}
          />
        )}

        {activeTab === 'gradcam' && (
          <GradCamTab
            gradcamData={gradcamData}
            gradcamCategory={gradcamCategory}
            setGradcamCategory={setGradcamCategory}
          />
        )}

        {activeTab === 'pipeline' && (
          <PipelineTab
            pipelineData={pipelineData}
            selectedStage={selectedStage}
            setSelectedStage={setSelectedStage}
          />
        )}

        {activeTab === 'per_strip' && (
          <PerStripTab
            resultsData={resultsData}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-space-800 bg-space-900 py-6 text-center text-xs text-slate-500">
        <p>IceShadowNet: Chandrayaan-2 Dual-Frequency Synthetic Aperture Radar (DFSAR) Polarimetric Benchmark</p>
        <p className="mt-1 text-slate-600">ISRO Chandrayaan-2 L1A SLI Scene (lines 100,000–140,000). Model inputs strictly raw log-intensities [HH, HV, VV].</p>
      </footer>

      {/* Metadata Modal */}
      {showMetaModal && (
        <MetaModal
          metaData={metaData}
          onClose={() => setShowMetaModal(false)}
        />
      )}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 1. OVERVIEW TAB
// ─────────────────────────────────────────────────────────────────────────────
function OverviewTab({ runs, summaryByConfig, baselines, trivial, splitMode, setSplitMode }) {
  const isPrimary = splitMode === 'primary';
  const splitInfo = isPrimary ? trivial?.full_test : trivial?.clean_test;

  return (
    <div className="space-y-6">
      {/* Headline Insight Banner */}
      <div className="p-5 rounded-xl border border-cyan-500/30 bg-gradient-to-r from-space-800 via-space-800 to-cyan-950/40 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">
              <ShieldCheck className="h-4 w-4" />
              <span>Headline Research Finding (§3.1)</span>
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              On Pooled Test Metrics, No Deep Model Outperforms Classical Logistic Regression
            </h2>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
              CustomCNN achieves ROC-AUC <strong className="text-cyan-300">0.756 ± 0.013</strong> vs Logistic Regression’s <strong className="text-cyan-300">0.731</strong> (gap &lt; 2 seed-std).
              On along-track strips 6 & 10, deep model F1 remains tethered to the trivial all-positive baseline, demonstrating that 3-channel backscatter contains limited discriminative capacity beyond bulk radar brightness.
            </p>
          </div>

          {/* Split Mode Selector */}
          <div className="flex bg-space-900/90 border border-space-600 p-1 rounded-lg self-start md:self-auto shrink-0">
            <button
              onClick={() => setSplitMode('primary')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
                isPrimary
                  ? 'bg-cyan-500 text-space-900 shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Full Test Split (N=338, Prev=29.9%)
            </button>
            <button
              onClick={() => setSplitMode('clean')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
                !isPrimary
                  ? 'bg-cyan-500 text-space-900 shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Ambiguity-Removed (N=205, Prev=49.3%)
            </button>
          </div>
        </div>
      </div>

      {/* Model Family Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(summaryByConfig).map(([cfg, data]) => {
          const stats = isPrimary ? data.primary : data.clean;
          const isWinner = cfg === 'CustomCNN';
          return (
            <div
              key={cfg}
              className={`p-4 rounded-xl border bg-space-800 transition ${
                isWinner ? 'border-cyan-500/50 shadow-md shadow-cyan-950/20' : 'border-space-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold font-mono text-slate-200 truncate">{cfg}</span>
                {isWinner && (
                  <span className="text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded">
                    Top Val AUC
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-space-700/60">
                <div>
                  <div className="text-[10px] text-slate-400 uppercase">ROC-AUC</div>
                  <div className="text-lg font-bold text-white">
                    {stats.roc_auc.mean.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-500">± {stats.roc_auc.std.toFixed(4)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400 uppercase">F1 Score</div>
                  <div className="text-lg font-bold text-white">
                    {stats.f1.mean.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-500">± {stats.f1.std.toFixed(4)}</div>
                </div>
              </div>
            </div>
          );
        })}

        {/* Logistic Regression Card */}
        <div className="p-4 rounded-xl border border-space-700 bg-space-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold font-mono text-slate-200">Logistic Regression</span>
            <span className="text-[10px] bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded">Baseline</span>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-space-700/60">
            <div>
              <div className="text-[10px] text-slate-400 uppercase">ROC-AUC</div>
              <div className="text-lg font-bold text-amber-300">
                {isPrimary ? '0.7314' : '0.8512'}
              </div>
              <div className="text-[10px] text-slate-500">Balanced</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase">F1 Score</div>
              <div className="text-lg font-bold text-amber-300">
                {isPrimary ? '0.5145' : '0.6813'}
              </div>
              <div className="text-[10px] text-slate-500">Th=0.50</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Benchmark Results Table */}
      <div className="rounded-xl border border-space-700 bg-space-800 overflow-hidden shadow">
        <div className="p-4 border-b border-space-700 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              {isPrimary ? '1. Primary Benchmark (Full Spatial Test Set: N=338)' : '2. Ambiguity-Removed Benchmark (Clean Test Set: N=205)'}
            </h3>
            <p className="text-xs text-slate-400">
              3 along-track strips (6, 10, 14), 10 distinct resamples, 1000 block-bootstrap rounds.
            </p>
          </div>
          <div className="text-xs text-slate-400 font-mono bg-space-900 px-3 py-1.5 rounded border border-space-700">
            Trivial All-Positive F1: <strong className="text-cyan-400">{splitInfo?.all_positive_f1.toFixed(4)}</strong> (Prev: {(splitInfo?.prevalence * 100).toFixed(1)}%)
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-space-900/70 text-slate-400 uppercase font-mono border-b border-space-700">
              <tr>
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-3">Seed</th>
                <th className="py-3 px-3">Best Ep</th>
                <th className="py-3 px-3">Val AUC</th>
                <th className="py-3 px-4">Test ROC-AUC (95% CI)</th>
                <th className="py-3 px-4">Test PR-AUC (95% CI)</th>
                <th className="py-3 px-4">Test F1 (Default 95% CI)</th>
                <th className="py-3 px-3">Tuned Th</th>
                <th className="py-3 px-4">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-space-700/50">
              {runs.map((r, i) => {
                const met = isPrimary ? r.test_primary : r.test_clean;
                return (
                  <tr key={`${r.model}_${r.seed}`} className="hover:bg-space-700/30 transition">
                    <td className="py-2.5 px-4 font-mono font-medium text-slate-200">
                      {r.model}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{r.seed}</td>
                    <td className="py-2.5 px-3 text-slate-300 font-mono">{r.best_epoch}</td>
                    <td className="py-2.5 px-3 text-slate-300 font-mono">{r.val_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-4 font-mono">
                      <span className="font-bold text-white">{met.roc_auc.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-slate-400">{met.roc_auc_ci}</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono">
                      <span className="font-bold text-white">{met.pr_auc.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-slate-400">{met.pr_auc_ci}</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono">
                      <span className="font-bold text-cyan-300">{met.f1.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-slate-400">{met.f1_default_ci}</span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 font-mono">{r.tuned_th.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-[11px] text-slate-400 font-mono whitespace-nowrap">
                      {met.support}
                    </td>
                  </tr>
                );
              })}

              {/* Classical Baselines Rows */}
              {['Decision Tree (max_depth=3, balanced)', 'Logistic Regression (balanced)'].map((baseName) => {
                const bRow = (isPrimary ? baselines.primary : baselines.clean)?.find(b => b.model.includes(baseName.split(' ')[0]));
                if (!bRow) return null;
                return (
                  <tr key={baseName} className="bg-space-900/50 hover:bg-space-700/40 transition font-mono">
                    <td className="py-2.5 px-4 font-bold text-amber-300">{baseName}</td>
                    <td className="py-2.5 px-3 text-slate-500">—</td>
                    <td className="py-2.5 px-3 text-slate-500">—</td>
                    <td className="py-2.5 px-3 text-slate-500">—</td>
                    <td className="py-2.5 px-4 font-mono">
                      <span className="font-bold text-amber-200">{bRow.roc_auc.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-slate-400">{bRow.roc_auc_ci}</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono">
                      <span className="font-bold text-amber-200">{bRow.pr_auc.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-slate-400">{bRow.pr_auc_ci}</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono">
                      <span className="font-bold text-amber-300">{bRow.f1.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-slate-400">{bRow.f1_ci}</span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 font-mono">{bRow.threshold.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-[11px] text-slate-400 font-mono whitespace-nowrap">
                      {bRow.support}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 2. TRAINING CURVES TAB
// ─────────────────────────────────────────────────────────────────────────────
function CurvesTab({ curvesData, selectedModelRun, setSelectedModelRun }) {
  const runs = curvesData?.runs || [];
  const selectedRun = runs.find(r => `${r.model}_seed${r.seed}` === selectedModelRun) || runs[0];

  const curve = selectedRun?.curve || [];

  return (
    <div className="space-y-6">
      {/* Controls & Granularity Note */}
      <div className="p-4 rounded-xl border border-space-700 bg-space-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">
            Select Training Run Checkpoint
          </label>
          <select
            value={selectedModelRun}
            onChange={(e) => setSelectedModelRun(e.target.value)}
            className="bg-space-900 border border-space-600 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:outline-none focus:border-cyan-400"
          >
            {runs.map((r) => {
              const key = `${r.model}_seed${r.seed}`;
              return (
                <option key={key} value={key}>
                  {r.model} (Seed {r.seed}) — Best Epoch: {r.best_epoch}
                </option>
              );
            })}
          </select>
        </div>

        <div className="text-xs text-slate-400 max-w-xl">
          <div className="flex items-center space-x-1.5 text-cyan-400 font-medium mb-1">
            <Info className="h-4 w-4" />
            <span>Exact Un-Interpolated Log Granularity</span>
          </div>
          <p>
            Training logs print <strong>Epoch 1</strong>, and then <strong>every 5th epoch</strong> (5, 10, 15...) until early stopping (patience=5 without val AUC improvement). Data points below correspond directly to raw log lines.
          </p>
        </div>
      </div>

      {/* SVG Training Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss Curve */}
        <div className="p-5 rounded-xl border border-space-700 bg-space-800 shadow">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-bold text-white uppercase tracking-wider">Loss Trajectory</h4>
            <div className="flex items-center space-x-4 text-xs font-mono">
              <span className="flex items-center space-x-1 text-cyan-400">
                <span className="h-2 w-2 rounded-full bg-cyan-400"></span>
                <span>Train Loss</span>
              </span>
              <span className="flex items-center space-x-1 text-rose-400">
                <span className="h-2 w-2 rounded-full bg-rose-400"></span>
                <span>Val Loss</span>
              </span>
            </div>
          </div>

          <SimpleSvgChart
            data={curve}
            lines={[
              { key: 'train_loss', color: '#22d3ee', label: 'Train Loss' },
              { key: 'val_loss', color: '#fb7185', label: 'Val Loss' },
            ]}
            bestEpoch={selectedRun?.best_epoch}
            yDomain={[0, Math.max(...curve.map(d => Math.max(d.train_loss, d.val_loss)), 2.5)]}
          />
        </div>

        {/* Validation Metrics Curve */}
        <div className="p-5 rounded-xl border border-space-700 bg-space-800 shadow">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-bold text-white uppercase tracking-wider">Validation Metrics (AUC & F1)</h4>
            <div className="flex items-center space-x-4 text-xs font-mono">
              <span className="flex items-center space-x-1 text-emerald-400">
                <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
                <span>Val ROC-AUC</span>
              </span>
              <span className="flex items-center space-x-1 text-amber-400">
                <span className="h-2 w-2 rounded-full bg-amber-400"></span>
                <span>Val F1</span>
              </span>
            </div>
          </div>

          <SimpleSvgChart
            data={curve}
            lines={[
              { key: 'val_auc', color: '#34d399', label: 'Val AUC' },
              { key: 'val_f1', color: '#fbbf24', label: 'Val F1' },
            ]}
            bestEpoch={selectedRun?.best_epoch}
            yDomain={[0, 1.0]}
          />
        </div>
      </div>

      {/* Epoch Table */}
      <div className="rounded-xl border border-space-700 bg-space-800 overflow-hidden">
        <div className="p-4 border-b border-space-700">
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Logged Epoch Checkpoints</h4>
        </div>
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-space-900/60 text-slate-400 border-b border-space-700">
            <tr>
              <th className="py-2.5 px-4">Epoch</th>
              <th className="py-2.5 px-4">Train Loss</th>
              <th className="py-2.5 px-4">Val Loss</th>
              <th className="py-2.5 px-4">Val ROC-AUC</th>
              <th className="py-2.5 px-4">Val F1</th>
              <th className="py-2.5 px-4">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-space-700/50">
            {curve.map((row) => {
              const isBest = row.epoch === selectedRun?.best_epoch;
              return (
                <tr key={row.epoch} className={isBest ? 'bg-cyan-500/10 font-bold text-cyan-300' : 'text-slate-300'}>
                  <td className="py-2.5 px-4">Epoch {row.epoch}</td>
                  <td className="py-2.5 px-4">{row.train_loss.toFixed(4)}</td>
                  <td className="py-2.5 px-4">{row.val_loss.toFixed(4)}</td>
                  <td className="py-2.5 px-4">{row.val_auc.toFixed(4)}</td>
                  <td className="py-2.5 px-4">{row.val_f1.toFixed(3)}</td>
                  <td className="py-2.5 px-4">
                    {isBest ? (
                      <span className="px-2 py-0.5 rounded text-[10px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                        Best Val AUC
                      </span>
                    ) : (
                      <span className="text-slate-500 text-[10px]">Logged</span>
                    )}
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


// Lightweight SVG Curve Renderer
function SimpleSvgChart({ data, lines, bestEpoch, yDomain = [0, 1] }) {
  if (!data || data.length === 0) {
    return <div className="h-64 flex items-center justify-center text-slate-500 text-xs">No curve data available</div>;
  }

  const width = 500;
  const height = 240;
  const padding = { top: 20, right: 30, bottom: 30, left: 45 };

  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const minEpoch = 1;
  const maxEpoch = Math.max(...data.map(d => d.epoch), 15);
  const [minY, maxY] = yDomain;

  const getX = (ep) => padding.left + ((ep - minEpoch) / (maxEpoch - minEpoch || 1)) * innerW;
  const getY = (val) => padding.top + innerH - ((val - minY) / (maxY - minY || 1)) * innerH;

  return (
    <div className="w-full overflow-hidden">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-64">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
          const yVal = minY + pct * (maxY - minY);
          const y = getY(yVal);
          return (
            <g key={i}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#1f293d" strokeDasharray="3 3" />
              <text x={padding.left - 8} y={y + 4} fill="#64748b" fontSize="10" textAnchor="end" fontFamily="monospace">
                {yVal.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Best Epoch Line */}
        {bestEpoch && (
          <g>
            <line
              x1={getX(bestEpoch)}
              y1={padding.top}
              x2={getX(bestEpoch)}
              y2={height - padding.bottom}
              stroke="#06b6d4"
              strokeDasharray="4 2"
              strokeWidth="1.5"
            />
            <text
              x={getX(bestEpoch)}
              y={padding.top - 6}
              fill="#06b6d4"
              fontSize="10"
              textAnchor="middle"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Best @ Ep {bestEpoch}
            </text>
          </g>
        )}

        {/* Lines */}
        {lines.map((l) => {
          const pts = data.map(d => `${getX(d.epoch)},${getY(d[l.key])}`).join(' ');
          return (
            <g key={l.key}>
              <polyline
                fill="none"
                stroke={l.color}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={pts}
              />
              {data.map(d => (
                <circle
                  key={d.epoch}
                  cx={getX(d.epoch)}
                  cy={getY(d[l.key])}
                  r="4"
                  fill="#0a0d14"
                  stroke={l.color}
                  strokeWidth="2"
                />
              ))}
            </g>
          );
        })}

        {/* X axis labels */}
        {data.map(d => (
          <text
            key={d.epoch}
            x={getX(d.epoch)}
            y={height - padding.bottom + 18}
            fill="#94a3b8"
            fontSize="10"
            textAnchor="middle"
            fontFamily="monospace"
          >
            Ep {d.epoch}
          </text>
        ))}
      </svg>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 3. PATCHES TAB
// ─────────────────────────────────────────────────────────────────────────────
function PatchesTab({
  patchData,
  patchFilterLabel,
  setPatchFilterLabel,
  patchFilterStrip,
  setPatchFilterStrip,
  patchFilterAmbiguity,
  setPatchFilterAmbiguity,
  patchFilterOutcome,
  setPatchFilterOutcome,
  patchSelectedModel,
  setPatchSelectedModel,
  selectedPatch,
  setSelectedPatch,
}) {
  const allPatches = patchData?.patches || [];
  const modelRuns = patchData?.runs || [];

  // Filter logic
  const filtered = allPatches.filter((p) => {
    if (patchFilterLabel !== 'all' && p.label.toString() !== patchFilterLabel) return false;
    if (patchFilterStrip !== 'all' && p.strip_id.toString() !== patchFilterStrip) return false;
    if (patchFilterAmbiguity === 'clean' && !p.is_clean) return false;
    if (patchFilterAmbiguity === 'ambiguous' && p.is_clean) return false;

    if (patchFilterOutcome !== 'all') {
      const score = p.scores[patchSelectedModel] ?? 0.5;
      const th = 0.45; // default benchmark decision point
      const pred = score >= th ? 1 : 0;
      let outcome = '';
      if (p.label === 1 && pred === 1) outcome = 'TP';
      else if (p.label === 0 && pred === 1) outcome = 'FP';
      else if (p.label === 1 && pred === 0) outcome = 'FN';
      else if (p.label === 0 && pred === 0) outcome = 'TN';
      if (outcome !== patchFilterOutcome) return false;
    }

    return true;
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Filter Sidebar & Gallery */}
      <div className="lg:col-span-8 space-y-4">
        {/* Filter bar */}
        <div className="p-4 rounded-xl border border-space-700 bg-space-800 space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <div>
              <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Target Model</span>
              <select
                value={patchSelectedModel}
                onChange={(e) => setPatchSelectedModel(e.target.value)}
                className="bg-space-900 border border-space-600 rounded px-2.5 py-1 text-xs font-mono text-cyan-300"
              >
                {modelRuns.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            <div>
              <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Ground Truth</span>
              <select
                value={patchFilterLabel}
                onChange={(e) => setPatchFilterLabel(e.target.value)}
                className="bg-space-900 border border-space-600 rounded px-2.5 py-1 text-xs"
              >
                <option value="all">All Labels</option>
                <option value="1">Candidate Ice (Label 1)</option>
                <option value="0">Regolith (Label 0)</option>
              </select>
            </div>

            <div>
              <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Along-Track Strip</span>
              <select
                value={patchFilterStrip}
                onChange={(e) => setPatchFilterStrip(e.target.value)}
                className="bg-space-900 border border-space-600 rounded px-2.5 py-1 text-xs"
              >
                <option value="all">All Strips</option>
                <option value="6">Strip 6 (Prev 38.4%)</option>
                <option value="10">Strip 10 (Prev 47.3%)</option>
                <option value="14">Strip 14 (Prev 4.4%)</option>
              </select>
            </div>

            <div>
              <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Ambiguity Band</span>
              <select
                value={patchFilterAmbiguity}
                onChange={(e) => setPatchFilterAmbiguity(e.target.value)}
                className="bg-space-900 border border-space-600 rounded px-2.5 py-1 text-xs"
              >
                <option value="all">All Patches</option>
                <option value="clean">Clean (0 or ≥5 px)</option>
                <option value="ambiguous">Ambiguous (1–4 px)</option>
              </select>
            </div>

            <div>
              <span className="text-[11px] text-slate-400 font-bold uppercase block mb-1">Classification Outcome</span>
              <select
                value={patchFilterOutcome}
                onChange={(e) => setPatchFilterOutcome(e.target.value)}
                className="bg-space-900 border border-space-600 rounded px-2.5 py-1 text-xs"
              >
                <option value="all">All Outcomes</option>
                <option value="TP">True Positive (TP)</option>
                <option value="FP">False Positive (FP)</option>
                <option value="FN">False Negative (FN)</option>
                <option value="TN">True Negative (TN)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-space-700/60">
            <span>Showing <strong>{filtered.length}</strong> of <strong>{allPatches.length}</strong> patches</span>
            <span className="text-[11px] text-slate-500">Click any patch to inspect score distribution & coordinates</span>
          </div>
        </div>

        {/* Patch Grid */}
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2.5 max-h-[640px] overflow-y-auto p-2 bg-space-900/60 rounded-xl border border-space-800">
          {filtered.map((p) => {
            const isSel = selectedPatch?.patch_idx === p.patch_idx;
            const score = p.scores[patchSelectedModel] ?? 0;
            return (
              <button
                key={p.patch_idx}
                onClick={() => setSelectedPatch(p)}
                className={`relative group rounded-lg overflow-hidden border transition p-0.5 text-left ${
                  isSel
                    ? 'border-cyan-400 ring-2 ring-cyan-500/30 bg-cyan-950/40'
                    : 'border-space-700 hover:border-slate-500 bg-space-800'
                }`}
              >
                <img
                  src={`/${p.image_path}`}
                  alt={`Patch ${p.patch_idx}`}
                  className="w-full aspect-square object-cover rounded"
                  loading="lazy"
                />
                <div className="p-1 text-[10px] font-mono flex items-center justify-between">
                  <span className="text-slate-300">#{p.patch_idx}</span>
                  <span className={`px-1 rounded font-bold ${p.label === 1 ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-700 text-slate-400'}`}>
                    {p.label === 1 ? 'ICE' : 'REG'}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Patch Inspection Drawer */}
      <div className="lg:col-span-4 space-y-4">
        {selectedPatch ? (
          <div className="p-5 rounded-xl border border-space-700 bg-space-800 space-y-5 sticky top-24">
            <div className="flex items-center justify-between border-b border-space-700 pb-3">
              <div>
                <span className="text-xs font-mono text-cyan-400 font-bold uppercase">Inspection Target</span>
                <h3 className="text-lg font-bold text-white">Patch #{selectedPatch.patch_idx}</h3>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 rounded text-xs font-bold font-mono ${
                  selectedPatch.label === 1
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'bg-slate-700 text-slate-300'
                }`}>
                  {selectedPatch.label === 1 ? 'Ground Truth: ICE' : 'Ground Truth: REGOLITH'}
                </span>
              </div>
            </div>

            {/* Composite Display */}
            <div className="flex flex-col items-center">
              <div className="w-48 h-48 rounded-xl overflow-hidden border-2 border-space-600 shadow-xl relative group">
                <img
                  src={`/${selectedPatch.image_path}`}
                  alt={`Patch ${selectedPatch.patch_idx}`}
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-[11px] text-slate-400 mt-2 font-mono">
                HH (Red) · HV (Green) · VV (Blue) Log-Intensity
              </p>
            </div>

            {/* Physical Attributes */}
            <div className="grid grid-cols-2 gap-3 text-xs font-mono bg-space-900/80 p-3 rounded-lg border border-space-700">
              <div>
                <span className="text-slate-500 text-[10px] block">COORDINATES</span>
                <span className="text-slate-200">R: {selectedPatch.coords[0]}, C: {selectedPatch.coords[1]}</span>
              </div>
              <div>
                <span className="text-slate-500 text-[10px] block">ALONG-TRACK STRIP</span>
                <span className="text-slate-200">Strip {selectedPatch.strip_id}</span>
              </div>
              <div>
                <span className="text-slate-500 text-[10px] block">POSITIVE PIXELS</span>
                <span className={selectedPatch.pos_px >= 5 ? 'text-cyan-400 font-bold' : 'text-slate-300'}>
                  {selectedPatch.pos_px} / 4096 px ({(selectedPatch.ice_frac * 100).toFixed(2)}%)
                </span>
              </div>
              <div>
                <span className="text-slate-500 text-[10px] block">AMBIGUITY STATUS</span>
                <span className={selectedPatch.is_clean ? 'text-emerald-400' : 'text-amber-400'}>
                  {selectedPatch.is_clean ? 'Clean Ground Truth' : 'Ambiguous (1–4 px)'}
                </span>
              </div>
              <div>
                <span className="text-slate-500 text-[10px] block">MEAN CPR</span>
                <span className="text-slate-200">{selectedPatch.cpr_mean.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-slate-500 text-[10px] block">MEAN DOP</span>
                <span className="text-slate-200">{selectedPatch.dop_mean.toFixed(3)}</span>
              </div>
            </div>

            {/* Model Probability Breakdown */}
            <div>
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Model Confidence Scores</h4>
              <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                {Object.entries(selectedPatch.scores).map(([name, score]) => {
                  const isPos = score >= 0.45;
                  return (
                    <div key={name} className="flex items-center justify-between text-xs font-mono bg-space-900/50 p-1.5 rounded">
                      <span className="text-slate-300 truncate max-w-[170px]" title={name}>{name}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-16 h-1.5 bg-space-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${isPos ? 'bg-cyan-400' : 'bg-slate-500'}`}
                            style={{ width: `${Math.min(100, score * 100)}%` }}
                          />
                        </div>
                        <span className={`w-12 text-right ${isPos ? 'text-cyan-300 font-bold' : 'text-slate-400'}`}>
                          {score.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center text-slate-500 text-xs">Select a patch from the left to view details</div>
        )}
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 4. GRAD-CAM TAB
// ─────────────────────────────────────────────────────────────────────────────
function GradCamTab({ gradcamData, gradcamCategory, setGradcamCategory }) {
  const slots = gradcamData?.slots || [];

  const filteredSlots = slots.filter((s) => {
    if (gradcamCategory === 'failures') return s.slot.startsWith('failure_');
    if (gradcamCategory === 'tp') return s.slot.startsWith('tp_');
    if (gradcamCategory === 'fp') return s.slot.startsWith('fp_');
    if (gradcamCategory === 'fn') return s.slot.startsWith('fn_');
    if (gradcamCategory === 'tn') return s.slot.startsWith('tn_');
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Interpretability Banner */}
      <div className="p-5 rounded-xl border border-space-700 bg-space-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">
            <Sparkles className="h-4 w-4" />
            <span>Grad-CAM Interpretability & Verification (§3.6)</span>
          </div>
          <h3 className="text-lg font-bold text-white">Trained vs Untrained Weight Comparison</h3>
          <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
            Grad-CAM overlays generated for <strong>CustomCNN (seed 2024)</strong> compared against a randomly initialized network with identical architecture and seed 2024 init.
            Trained weights localize focus directly onto anomalous bright backscatter regions, whereas untrained weights display diffuse, non-specific activations.
          </p>
        </div>

        {/* Category Selector */}
        <div className="flex flex-wrap bg-space-900 border border-space-600 p-1 rounded-lg self-start md:self-auto gap-1">
          <button
            onClick={() => setGradcamCategory('failures')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
              gradcamCategory === 'failures' ? 'bg-cyan-500 text-space-900' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            4 Failure Cases (§3.7)
          </button>
          <button
            onClick={() => setGradcamCategory('tp')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
              gradcamCategory === 'tp' ? 'bg-cyan-500 text-space-900' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Top True Positives (6)
          </button>
          <button
            onClick={() => setGradcamCategory('fp')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
              gradcamCategory === 'fp' ? 'bg-cyan-500 text-space-900' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Top False Positives (6)
          </button>
          <button
            onClick={() => setGradcamCategory('fn')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
              gradcamCategory === 'fn' ? 'bg-cyan-500 text-space-900' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Top False Negatives (6)
          </button>
          <button
            onClick={() => setGradcamCategory('tn')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
              gradcamCategory === 'tn' ? 'bg-cyan-500 text-space-900' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Top True Negatives (6)
          </button>
        </div>
      </div>

      {/* Slots Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filteredSlots.map((s) => (
          <div key={s.slot} className="p-5 rounded-xl border border-space-700 bg-space-800 space-y-4 shadow">
            <div className="flex items-center justify-between border-b border-space-700/60 pb-2.5">
              <div>
                <span className="text-xs font-bold font-mono text-cyan-400 uppercase">{s.category}</span>
                <h4 className="text-base font-bold text-white">Patch #{s.patch_idx} (Strip {s.strip_id})</h4>
              </div>
              <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold ${
                s.true_label === 1 ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-700 text-slate-300'
              }`}>
                Label: {s.true_label === 1 ? 'Ice' : 'Regolith'}
              </span>
            </div>

            {/* Overlays Side-by-Side */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5 text-center">
                <span className="text-[11px] font-bold text-cyan-300 font-mono block">Trained Weights</span>
                <div className="rounded-lg overflow-hidden border border-space-600 bg-space-900 aspect-[2/1]">
                  <img src={`/${s.trained_model.gradcam_image}`} alt={`Trained ${s.slot}`} className="w-full h-full object-contain" />
                </div>
                <div className="text-[11px] font-mono text-slate-300">
                  Score: <strong className="text-cyan-400">{s.trained_model.score.toFixed(4)}</strong> (Th=0.45)
                </div>
              </div>

              <div className="space-y-1.5 text-center">
                <span className="text-[11px] font-bold text-slate-400 font-mono block">Untrained (Random Init)</span>
                <div className="rounded-lg overflow-hidden border border-space-600 bg-space-900 aspect-[2/1]">
                  <img src={`/${s.untrained_model.gradcam_image}`} alt={`Untrained ${s.slot}`} className="w-full h-full object-contain" />
                </div>
                <div className="text-[11px] font-mono text-slate-300">
                  Score: <strong className="text-slate-400">{s.untrained_model.score.toFixed(4)}</strong>
                </div>
              </div>
            </div>

            <div className="text-xs text-slate-400 font-mono bg-space-900/60 p-2.5 rounded border border-space-700/50 flex justify-between">
              <span>Coords: [{s.coords[0]}, {s.coords[1]}]</span>
              <span>Model Prediction: {s.trained_model.prediction === 1 ? 'Predicted ICE' : 'Predicted REGOLITH'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 5. PIPELINE STAGES TAB
// ─────────────────────────────────────────────────────────────────────────────
function PipelineTab({ pipelineData, selectedStage, setSelectedStage }) {
  const stages = pipelineData?.stages || [];
  const currentStage = stages.find(s => s.stage === selectedStage) || stages[0];

  return (
    <div className="space-y-6">
      <div className="p-5 rounded-xl border border-space-700 bg-space-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">
            <Layers className="h-4 w-4" />
            <span>Real DFSAR Ingestion & Label Generation Pipeline</span>
          </div>
          <h3 className="text-lg font-bold text-white">From Raw Complex SLI to Labeled Patches</h3>
          <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
            Visualizing an along-track tile crop (<strong className="text-slate-100">512 × 116 pixels</strong> at radar line 100,000) through every step of the real Chandrayaan-2 processing pipeline.
          </p>
        </div>
        <div className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-xs font-mono text-emerald-400 self-start md:self-auto">
          Source: Chandrayaan-2 DFSAR L1A SLI
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Stages Sidebar */}
        <div className="lg:col-span-5 space-y-2">
          {stages.map((st) => {
            const isSel = st.stage === selectedStage;
            return (
              <button
                key={st.stage}
                onClick={() => setSelectedStage(st.stage)}
                className={`w-full text-left p-3.5 rounded-xl border transition flex items-center justify-between ${
                  isSel
                    ? 'border-cyan-400 bg-cyan-950/40 text-cyan-300 shadow-md ring-1 ring-cyan-500/30'
                    : 'border-space-700 bg-space-800 text-slate-300 hover:border-space-600'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold font-mono ${
                    isSel ? 'bg-cyan-500 text-space-900' : 'bg-space-700 text-slate-400'
                  }`}>
                    {st.stage}
                  </span>
                  <div>
                    <div className="text-xs font-bold">{st.name}</div>
                    <div className="text-[10px] text-slate-500 font-mono mt-0.5">Image: {st.size_kb.toFixed(1)} KB</div>
                  </div>
                </div>
                <ChevronRight className={`h-4 w-4 ${isSel ? 'text-cyan-400' : 'text-slate-600'}`} />
              </button>
            );
          })}
        </div>

        {/* Display Pane */}
        <div className="lg:col-span-7 p-6 rounded-xl border border-space-700 bg-space-800 flex flex-col items-center justify-center space-y-4">
          <div className="text-center">
            <span className="text-xs font-mono text-cyan-400 uppercase font-bold tracking-wider">
              Stage {currentStage?.stage} of {stages.length}
            </span>
            <h4 className="text-base font-bold text-white mt-0.5">{currentStage?.name}</h4>
          </div>

          <div className="rounded-xl overflow-hidden border-2 border-space-700 bg-space-900 shadow-2xl max-h-[520px] flex items-center justify-center p-2">
            <img
              src={`/${currentStage?.image}`}
              alt={currentStage?.name}
              className="max-h-[500px] w-auto object-contain rounded"
            />
          </div>

          <p className="text-xs text-slate-400 text-center max-w-md">
            Extracted from raw L1A slant-range complex GeoTIFFs (latitude −81.7° to −86.3°).
          </p>
        </div>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 6. PER-STRIP ANALYSIS TAB
// ─────────────────────────────────────────────────────────────────────────────
function PerStripTab({ resultsData }) {
  const perStripTable = resultsData?.per_strip_table || [];

  return (
    <div className="space-y-6">
      {/* Overview Context */}
      <div className="p-5 rounded-xl border border-space-700 bg-space-800 space-y-3">
        <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider">
          <Compass className="h-4 w-4" />
          <span>Along-Track Spatial Strip Breakdown</span>
        </div>
        <h3 className="text-lg font-bold text-white">Spatial Prevalence Heterogeneity Across Strips</h3>
        <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
          The test split is partitioned into 3 along-track strips spanning lines 100,000–140,000.
          Strips 6 and 10 feature high ice candidate prevalence (<strong className="text-cyan-300">38.4%</strong> and <strong className="text-cyan-300">47.3%</strong>),
          whereas Strip 14 features near-zero ice candidate prevalence (<strong className="text-rose-400">4.4%</strong>, 5 positives out of 114 patches).
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
          <div className="p-3 bg-space-900 rounded-lg border border-space-700 font-mono text-xs">
            <span className="text-slate-400 block text-[10px]">STRIP 6</span>
            <span className="text-white font-bold text-sm">112 Patches</span>
            <div className="text-slate-400 text-[11px] mt-1">43 Pos · 69 Neg · 38.4% Prev</div>
          </div>
          <div className="p-3 bg-space-900 rounded-lg border border-space-700 font-mono text-xs">
            <span className="text-slate-400 block text-[10px]">STRIP 10</span>
            <span className="text-white font-bold text-sm">112 Patches</span>
            <div className="text-slate-400 text-[11px] mt-1">53 Pos · 59 Neg · 47.3% Prev</div>
          </div>
          <div className="p-3 bg-space-900 rounded-lg border border-space-700 font-mono text-xs">
            <span className="text-slate-400 block text-[10px]">STRIP 14</span>
            <span className="text-rose-400 font-bold text-sm">114 Patches</span>
            <div className="text-slate-400 text-[11px] mt-1">5 Pos · 109 Neg · 4.4% Prev</div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-space-700 bg-space-800 overflow-hidden shadow">
        <div className="p-4 border-b border-space-700">
          <h4 className="text-xs font-bold text-white uppercase tracking-wider">Per-Strip Performance Table</h4>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-space-900/60 text-slate-400 border-b border-space-700">
              <tr>
                <th className="py-2.5 px-4">Model</th>
                <th className="py-2.5 px-3">Seed</th>
                <th className="py-2.5 px-3">Strip</th>
                <th className="py-2.5 px-3">Pos / Neg</th>
                <th className="py-2.5 px-3">TP</th>
                <th className="py-2.5 px-3">FP</th>
                <th className="py-2.5 px-3">FN</th>
                <th className="py-2.5 px-3">TN</th>
                <th className="py-2.5 px-3">Recall</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-4">F1 Score</th>
                <th className="py-2.5 px-4">ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-space-700/50">
              {perStripTable.map((row, idx) => (
                <tr key={idx} className="hover:bg-space-700/30 transition">
                  <td className="py-2 px-4 font-medium text-slate-200">{row.model}</td>
                  <td className="py-2 px-3 text-slate-400">{row.seed ?? '—'}</td>
                  <td className="py-2 px-3 font-bold text-cyan-400">Strip {row.strip_id}</td>
                  <td className="py-2 px-3 text-slate-400">{row.n_pos} / {row.n_neg}</td>
                  <td className="py-2 px-3 text-slate-300">{row.tp}</td>
                  <td className="py-2 px-3 text-slate-300">{row.fp}</td>
                  <td className="py-2 px-3 text-slate-300">{row.fn}</td>
                  <td className="py-2 px-3 text-slate-300">{row.tn}</td>
                  <td className="py-2 px-3 text-slate-300">{row.recall.toFixed(4)}</td>
                  <td className="py-2 px-3 text-slate-300">{row.precision.toFixed(4)}</td>
                  <td className="py-2 px-4 font-bold text-cyan-300">{row.f1.toFixed(4)}</td>
                  <td className="py-2 px-4 text-slate-200">{row.roc_auc !== null ? row.roc_auc.toFixed(4) : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// 7. METADATA MODAL
// ─────────────────────────────────────────────────────────────────────────────
function MetaModal({ metaData, onClose }) {
  const feat = metaData?.features || {};
  const base = metaData?.classical_baselines || {};

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-space-800 border border-space-600 rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 space-y-5 shadow-2xl">
        <div className="flex items-center justify-between border-b border-space-700 pb-3">
          <div className="flex items-center space-x-2 text-cyan-400 font-bold">
            <Info className="h-5 w-5" />
            <h3 className="text-base text-white">Methodological Specification & Baseline Definitions</h3>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-white bg-space-700">
            ✕
          </button>
        </div>

        {/* Anti-Circularity Policy */}
        <div className="p-4 rounded-xl bg-cyan-950/30 border border-cyan-500/40 space-y-2">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider">
            <ShieldCheck className="h-4 w-4" />
            <span>Strict Anti-Circularity Policy</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            {feat.derived_polarimetry_circularity_guard}
          </p>
        </div>

        {/* Feature List */}
        <div>
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Model Input Features</h4>
          <div className="grid grid-cols-3 gap-2 font-mono text-xs">
            {feat.model_input_features?.map((f) => (
              <div key={f} className="p-2.5 bg-space-900 rounded border border-space-700 text-cyan-300 text-center font-bold">
                {f}
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-400 mt-2">{feat.feature_extraction_logic}</p>
        </div>

        {/* Classical Models */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Classical Machine Learning Baselines</h4>
          <div className="space-y-2 text-xs font-mono">
            <div className="p-3 bg-space-900 rounded border border-space-700">
              <span className="text-amber-400 font-bold block mb-1">Decision Tree</span>
              <p className="text-slate-300 text-[11px]">{base.decision_tree?.model}</p>
              <p className="text-slate-400 text-[11px] mt-1">{base.decision_tree?.threshold_tuning}</p>
            </div>
            <div className="p-3 bg-space-900 rounded border border-space-700">
              <span className="text-amber-400 font-bold block mb-1">Logistic Regression</span>
              <p className="text-slate-300 text-[11px]">{base.logistic_regression?.model}</p>
              <p className="text-slate-400 text-[11px] mt-1">{base.logistic_regression?.threshold_tuning}</p>
            </div>
          </div>
        </div>

        {/* Oracle Baseline */}
        <div className="p-3 bg-space-900/60 rounded border border-space-700 space-y-1 text-xs">
          <span className="text-slate-200 font-bold font-mono">ORACLE Baseline Status</span>
          <p className="text-slate-400 text-[11px]">
            {base.oracle_baseline?.definition} {base.oracle_baseline?.purpose}
          </p>
        </div>

        <div className="pt-2 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-space-900 font-semibold rounded-lg text-xs transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
