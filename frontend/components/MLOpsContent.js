import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  LineChart, Line, CartesianGrid, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, Cell, PieChart, Pie
} from 'recharts';

const API = 'http://localhost:8000';

const fmt = (v, suffix = '') =>
  v == null || isNaN(v) ? '—' : `${Number(v).toLocaleString('fr-TN', { maximumFractionDigits: 1 })}${suffix}`;

const fmtCurrency = (v) =>
  v == null ? '—' : new Intl.NumberFormat('fr-TN', { style: 'currency', currency: 'TND', maximumFractionDigits: 0 }).format(Number(v));

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) return { error: payload.detail || payload.error || `HTTP ${res.status}` };
  return payload;
}

// ── Design Tokens ─────────────────────────────────────────────────────────
const BRAND = '#004A8D';
const ACCENT = '#F38F1D';
const PALETTE = ['#004A8D', '#F38F1D', '#10b981', '#8b5cf6', '#ef4444', '#06b6d4', '#f59e0b'];

// ── Shared UI Components ──────────────────────────────────────────────────
function Badge({ label, color = 'gray' }) {
  const map = {
    green:  'bg-emerald-50 text-emerald-700 border-emerald-200',
    red:    'bg-rose-50 text-rose-700 border-rose-200',
    yellow: 'bg-amber-50 text-amber-700 border-amber-200',
    blue:   'bg-sky-50 text-sky-700 border-sky-200',
    purple: 'bg-violet-50 text-violet-700 border-violet-200',
    gray:   'bg-slate-50 text-slate-600 border-slate-200',
  };
  return (
    <span className={`px-2 py-0.5 rounded border text-[10px] uppercase font-bold tracking-wider ${map[color] || map.gray}`}>
      {label}
    </span>
  );
}

function SectionTitle({ children, icon }) {
  return (
    <h3 className="text-base font-bold text-[#004A8D] flex items-center gap-2 mb-4 pb-2 border-b border-slate-100">
      {icon && <span className="text-lg">{icon}</span>}
      {children}
    </h3>
  );
}

function StatCard({ label, value, sub, tone = 'blue' }) {
  const tones = {
    blue:   'from-[#004A8D]/8 to-transparent border-[#004A8D]/15 text-[#004A8D]',
    orange: 'from-[#F38F1D]/8 to-transparent border-[#F38F1D]/15 text-[#F38F1D]',
    green:  'from-emerald-500/8 to-transparent border-emerald-200 text-emerald-700',
    red:    'from-rose-500/8 to-transparent border-rose-200 text-rose-700',
    purple: 'from-violet-500/8 to-transparent border-violet-200 text-violet-700',
  };
  return (
    <div className={`bg-gradient-to-br ${tones[tone] || tones.blue} border rounded-xl p-4`}>
      <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">{label}</p>
      <p className="text-2xl font-black">{value}</p>
      {sub && <p className="text-xs opacity-60 mt-1">{sub}</p>}
    </div>
  );
}

function Spinner({ label = 'Chargement...' }) {
  return (
    <div className="h-64 flex flex-col justify-center items-center text-slate-400 gap-3">
      <div className="w-8 h-8 border-3 border-slate-200 border-t-[#004A8D] rounded-full animate-spin" />
      <p className="text-sm font-medium animate-pulse">{label}</p>
    </div>
  );
}

function ErrorBox({ msg }) {
  return (
    <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-xl text-sm flex items-center gap-2">
      ❌ {msg}
    </div>
  );
}

function MetricRow({ label, value, max = 1, color = BRAND }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="flex items-center gap-3">
      <p className="text-xs font-bold text-slate-500 w-24 shrink-0 uppercase tracking-wide">{label}</p>
      <div className="flex-1 bg-slate-100 rounded-full h-2">
        <div className="h-2 rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: color }} />
      </div>
      <p className="text-sm font-black text-slate-700 w-14 text-right">{fmt(value * 100, '%')}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 1 — Impayé Risk Scoring  (existing ml_pipeline.py)
// Notebook: impaye_risk_scoring.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function ImpayeTab() {
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState(null);
  const [trainError, setTrainError] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [explainResult, setExplainResult] = useState(null);
  const [form, setForm] = useState({
    branche: 'AUTO', periodicite: 'A', police_situation: 'V',
    annee_echeance: 2024, mois_echeance: 12,
    mt_pnet: 1500, mt_rc: 500, mt_commission: 150, bonus_malus: 0,
  });

  const m = trainResult?.metrics ?? {};
  const score = Number(explainResult?.probability_impaye ?? explainResult?.probability ?? 0);
  const threshold = Number(explainResult?.threshold ?? 0.5);
  const predicted = Number(explainResult?.predicted_label ?? explainResult?.prediction ?? (score >= threshold ? 1 : 0));
  const tone = score >= threshold
    ? 'border-rose-400 bg-rose-50 text-rose-600'
    : score >= threshold * 0.8
    ? 'border-amber-400 bg-amber-50 text-amber-600'
    : 'border-emerald-400 bg-emerald-50 text-emerald-600';

  const train = async () => {
    setTraining(true); setTrainError(null); setTrainResult(null);
    try {
      const data = await fetchJson(`${API}/api/v1/ml/train`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ year_from: null, year_to: null, test_size: 0.2 }),
      });
      if (data.error || data.status === 'error') throw new Error(data.error || 'Échec');
      setTrainResult(data);
    } catch (e) { setTrainError(e.message); } finally { setTraining(false); }
  };

  const predict = async () => {
    setExplaining(true); setExplainResult(null);
    try {
      const data = await fetchJson(`${API}/api/v1/ml/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          annee_echeance: Number(form.annee_echeance),
          mois_echeance: Number(form.mois_echeance),
          mt_pnet: Number(form.mt_pnet),
          mt_rc: Number(form.mt_rc),
          mt_commission: Number(form.mt_commission),
          bonus_malus: Number(form.bonus_malus),
        }),
      });
      setExplainResult(data);
    } catch (e) { setExplainResult({ error: e.message }); } finally { setExplaining(false); }
  };

  const field = (key, label, type = 'number', opts) => (
    <div key={key}>
      <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">{label}</label>
      {opts ? (
        <select value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })}
          className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#F38F1D]/50">
          {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      ) : (
        <input type={type} value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })}
          className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#F38F1D]/50" />
      )}
    </div>
  );

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="bg-gradient-to-r from-[#004A8D]/6 to-transparent p-5 rounded-2xl border border-[#004A8D]/10">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h4 className="font-bold text-[#004A8D] text-lg">Scoring Risque Impayé</h4>
              <Badge label="XGBoost / HGBC" color="blue" />
            </div>
            <p className="text-sm text-slate-500">
              Prédit la probabilité qu'une quittance devienne impayée avant son émission.
              Features : prime, commission, bonus-malus, historique agent/client/police.
            </p>
          </div>
          <button onClick={train} disabled={training}
            className="shrink-0 px-5 py-2.5 bg-[#004A8D] text-white rounded-xl shadow-md font-semibold hover:bg-blue-800 disabled:opacity-50 transition-all flex items-center gap-2 text-sm">
            {training ? <span className="animate-spin">⚙️</span> : '🚀'}
            {training ? 'Entraînement...' : 'Forcer Réentraînement'}
          </button>
        </div>
        {trainError && <div className="mt-3"><ErrorBox msg={trainError} /></div>}
        {trainResult && (
          <div className="mt-4 bg-emerald-50 border border-emerald-200 rounded-xl p-4">
            <p className="text-emerald-800 font-bold mb-3">✅ Entraînement terminé — {trainResult.selected_model || trainResult.model_type || '—'}</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[['Accuracy', m.accuracy], ['Precision', m.precision], ['Recall', m.recall], ['F1', m.f1]].map(([lbl, v]) => (
                <div key={lbl} className="bg-white rounded-lg p-2 text-center">
                  <p className="text-[10px] text-slate-400 font-bold uppercase">{lbl}</p>
                  <p className="text-lg font-black text-emerald-700">{fmt(v * 100, '%')}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Simulator */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon="🧪">Simulateur d'Inférence</SectionTitle>
          <div className="grid grid-cols-2 gap-4">
            {field('branche', 'Branche', 'text', [['AUTO', 'Automobile'], ['IRDS', 'IRDS'], ['SANTE', 'Santé']])}
            {field('periodicite', 'Périodicité', 'text', [['A', 'Annuel'], ['S', 'Semestriel'], ['T', 'Trimestriel'], ['C', 'Comptant']])}
            {field('police_situation', 'Situation Police', 'text', [['V', 'En vigueur'], ['R', 'Résiliée'], ['T', 'Terminée']])}
            {field('mois_echeance', 'Mois Échéance')}
            {field('mt_pnet', 'Prime Nette (TND)')}
            {field('mt_commission', 'Commission (TND)')}
            {field('mt_rc', 'Montant RC (TND)')}
            {field('bonus_malus', 'Bonus/Malus', 'number')}
          </div>
          <button onClick={predict} disabled={explaining}
            className="w-full mt-5 py-3 bg-[#F38F1D] text-white rounded-xl font-bold hover:bg-[#d97d16] disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-md shadow-[#F38F1D]/20">
            {explaining ? <span className="animate-pulse">Calcul en cours...</span> : '⚡ Évaluer le Risque d\'Impayé'}
          </button>
        </div>

        <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 flex flex-col items-center justify-center min-h-[320px]">
          {explainResult ? (
            explainResult.error ? <ErrorBox msg={explainResult.error} /> : (
              <div className="text-center w-full animate-in zoom-in duration-300">
                <p className="text-xs font-black text-slate-400 tracking-widest uppercase mb-5">Résultat Inférence</p>
                <div className={`mx-auto w-32 h-32 rounded-full flex flex-col items-center justify-center border-4 shadow-inner mb-5 ${tone}`}>
                  <span className="text-3xl font-black">{fmt(score * 100, '%')}</span>
                  <span className="text-[10px] font-bold uppercase mt-1">Risque</span>
                </div>
                <h4 className="text-lg font-bold text-slate-800 mb-2">
                  {predicted === 1 ? '⚠️ Impayé Anticipé' : '✅ Paiement Fiable'}
                </h4>
                <p className="text-slate-400 text-xs">Seuil décisionnel : <strong>{fmt(threshold * 100, '%')}</strong></p>
                {explainResult.feature_importance && (
                  <div className="mt-4 w-full space-y-2">
                    {Object.entries(explainResult.feature_importance).slice(0, 4).map(([f, v]) => (
                      <MetricRow key={f} label={f.replace(/_/g, ' ')} value={v} color={ACCENT} />
                    ))}
                  </div>
                )}
              </div>
            )
          ) : (
            <div className="text-slate-400 flex flex-col items-center gap-3">
              <span className="text-5xl opacity-30">🤖</span>
              <p className="text-sm font-medium">Simulez une quittance pour voir le score</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 2 — Churn Prediction
// Notebook: churn_prediction_v3.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function ChurnTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({
    branche: 'AUTO', bonus_malus: 1.0, nb_quittances: 4,
    mt_pnet: 1200, taux_impaye: 0, nb_sinistres: 0,
  });
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    fetchJson(`${API}/api/v1/ml/churn/summary`)
      .then(d => setData(d))
      .catch(() => setData({ error: 'Endpoint non disponible' }))
      .finally(() => setLoading(false));
  }, []);

  const predict = async () => {
    setPredicting(true); setResult(null);
    try {
      const d = await fetchJson(`${API}/api/v1/ml/churn/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, bonus_malus: Number(form.bonus_malus), nb_quittances: Number(form.nb_quittances), mt_pnet: Number(form.mt_pnet), taux_impaye: Number(form.taux_impaye), nb_sinistres: Number(form.nb_sinistres) }),
      });
      setResult(d);
    } catch (e) { setResult({ error: e.message }); } finally { setPredicting(false); }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Model info banner */}
      <div className="bg-gradient-to-r from-violet-500/6 to-transparent p-5 rounded-2xl border border-violet-200/40">
        <div className="flex items-center gap-3 mb-2">
          <h4 className="font-bold text-violet-800 text-lg">Prédiction du Churn</h4>
          <Badge label="XGBoost + SMOTE" color="purple" />
          <Badge label="v3" color="gray" />
        </div>
        <p className="text-sm text-slate-500">
          Identifie les polices à risque de résiliation à partir des features contrat, sinistralité,
          historique de paiement et profil client. Cible : <code className="bg-slate-100 px-1 rounded text-xs">CHURN = 1</code>.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stats panel */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon="📊">Vue Portefeuille — Risque Churn</SectionTitle>
          {loading ? <Spinner label="Chargement des statistiques churn..." /> :
           data?.error ? <ErrorBox msg={data.error} /> : data ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Taux Résiliation Réel" value={fmt(data.taux_churn_pct, '%')} sub={`${fmt(data.nb_churn)} polices résiliées / annulées`} tone="red" />
                <StatCard label="Polices Analysées" value={fmt(data.nb_polices)} sub="en portefeuille actif" tone="blue" />
              </div>
              {data.by_branche && data.by_branche.length > 0 && (
                <div>
                  <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Résiliation par Branche</p>
                  <div className="space-y-2 mt-1">
                    {data.by_branche.map((r, i) => {
                      const total   = Number(r.nb_total)   || 1;
                      const resilie = Number(r.nb_resilie) || 0;
                      const taux    = Number(r.taux_resiliation_pct) || 0;
                      const pct     = (resilie / total) * 100;
                      return (
                        <div key={i}>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs font-bold text-slate-600">{r.branche}</span>
                            <span className="text-xs text-slate-500">
                              <strong className="text-violet-700">{resilie.toLocaleString('fr-TN')}</strong>
                              <span className="text-slate-400"> / {total.toLocaleString('fr-TN')} — </span>
                              <strong className="text-rose-600">{taux}%</strong>
                            </span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                            <div
                              className="h-3 rounded-full transition-all duration-700"
                              style={{ width: `${Math.min(100, pct)}%`, background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)' }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              {data.top_features && (
                <div>
                  <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Features Importantes</p>
                  <div className="space-y-2">
                    {data.top_features.slice(0, 5).map(f => (
                      <MetricRow key={f.feature} label={f.feature.replace(/_/g, ' ')} value={f.importance} color="#8b5cf6" />
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </div>

        {/* Simulator */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon="🎯">Simulateur Churn</SectionTitle>
          <div className="grid grid-cols-2 gap-3">
            {[
              ['branche', 'Branche', 'select', [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']]],
              ['nb_quittances', 'Nb Quittances', 'number'],
              ['mt_pnet', 'Prime Nette (TND)', 'number'],
              ['bonus_malus', 'Bonus/Malus', 'number'],
              ['taux_impaye', 'Taux Impayé (0-1)', 'number'],
              ['nb_sinistres', 'Nb Sinistres', 'number'],
            ].map(([key, label, type, opts]) => (
              <div key={key}>
                <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">{label}</label>
                {opts ? (
                  <select value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })}
                    className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-violet-500/30">
                    {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                  </select>
                ) : (
                  <input type="number" step="any" value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })}
                    className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-violet-500/30" />
                )}
              </div>
            ))}
          </div>
          <button onClick={predict} disabled={predicting}
            className="w-full mt-4 py-3 bg-violet-600 text-white rounded-xl font-bold hover:bg-violet-700 disabled:opacity-50 transition-all">
            {predicting ? '⏳ Calcul...' : '🔮 Prédire le Risque de Churn'}
          </button>
          {result && !result.error && (
            <div className={`mt-4 rounded-xl p-4 border text-center ${result.churn_predicted ? 'bg-rose-50 border-rose-200' : 'bg-emerald-50 border-emerald-200'}`}>
              <p className={`text-2xl font-black mb-1 ${result.churn_predicted ? 'text-rose-600' : 'text-emerald-700'}`}>
                {fmt(result.churn_probability * 100, '%')}
              </p>
              <p className={`font-bold text-sm ${result.churn_predicted ? 'text-rose-700' : 'text-emerald-700'}`}>
                {result.churn_predicted ? '⚠️ Résiliation Probable' : '✅ Client Stable'}
              </p>
              {result.action && <p className="text-xs text-slate-500 mt-2 italic">{result.action}</p>}
            </div>
          )}
          {result?.error && <div className="mt-3"><ErrorBox msg={result.error} /></div>}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 3 — Fraud Detection + Anomaly Detection
// Notebooks: fraud_detection.ipynb + anomaly_detection.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function FraudAnomalyTab() {
  const [view, setView] = useState('fraud');
  const [fraudData, setFraudData] = useState(null);
  const [anomalyData, setAnomalyData] = useState(null);
  const [contam, setContam] = useState(0.05);
  const [loading, setLoading] = useState(false);

  const loadFraud = useCallback(async () => {
    setLoading(true);
    const d = await fetchJson(`${API}/api/v1/ml/fraud/summary`).catch(() => ({ error: 'Indisponible' }));
    setFraudData(d);
    setLoading(false);
  }, []);

  const loadAnomaly = useCallback(async () => {
    setLoading(true);
    const d = await fetchJson(`${API}/api/v1/ml/anomalies?contamination=${contam}`).catch(e => ({ error: e.message }));
    setAnomalyData(d);
    setLoading(false);
  }, [contam]);

  useEffect(() => { view === 'fraud' ? loadFraud() : loadAnomaly(); }, [view, loadFraud, loadAnomaly]);

  const riskColor = (level) => ({ Normal: '#10b981', 'Risque Modéré': '#f59e0b', 'Risque Élevé': '#f97316', Critique: '#ef4444' }[level] || '#94a3b8');

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Sub-tabs */}
      <div className="flex gap-2 p-1 bg-slate-100 rounded-xl w-fit">
        {[['fraud', '🚨 Fraude', 'IF + AE + LOF'], ['anomaly', '🔍 Anomalies Contrats', 'Isolation Forest']].map(([id, label, sub]) => (
          <button key={id} onClick={() => setView(id)}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${view === id ? 'bg-white shadow text-[#004A8D]' : 'text-slate-500 hover:text-slate-700'}`}>
            {label}
            <span className="ml-1.5 text-[9px] font-normal opacity-60">{sub}</span>
          </button>
        ))}
      </div>

      {/* FRAUD view */}
      {view === 'fraud' && (
        <div>
          <div className="bg-gradient-to-r from-rose-500/6 to-transparent p-5 rounded-2xl border border-rose-200/40 mb-5">
            <div className="flex items-center gap-3 mb-1">
              <h4 className="font-bold text-rose-800 text-lg">Détection de Fraude</h4>
              <Badge label="Non-Supervisé" color="red" />
            </div>
            <p className="text-sm text-slate-500">
              Ensemble : IF (40%) + Autoencoder (40%) + LOF (20%). Score composite [0,1].
              Seuils : Normal / Modéré (p90) / Élevé (p95) / Critique (p99).
            </p>
          </div>
          {loading ? <Spinner label="Analyse fraude en cours..." /> :
           fraudData?.error ? <ErrorBox msg={fraudData.error} /> : fraudData ? (
            <div className="space-y-5">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="Cas Critiques" value={fmt(fraudData.nb_critique)} sub="FRAUD_SCORE > p99" tone="red" />
                <StatCard label="Risque Élevé" value={fmt(fraudData.nb_eleve)} sub="p95–p99" tone="orange" />
                <StatCard label="Risque Modéré" value={fmt(fraudData.nb_modere)} sub="p90–p95" tone="purple" />
                <StatCard label="Normal" value={fmt(fraudData.nb_normal)} sub="< p90" tone="green" />
              </div>
              {fraudData.top_fraud && (
                <div className="bg-white rounded-2xl border border-slate-100 p-5">
                  <SectionTitle icon="⚠️">Top Sinistres Suspects</SectionTitle>
                  <div className="space-y-3">
                    {fraudData.top_fraud.slice(0, 6).map((item, i) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-slate-50 hover:bg-slate-100 transition">
                        <div>
                          <p className="font-bold text-sm text-slate-700">{item.num_sinistre || `#${i + 1}`}</p>
                          <p className="text-xs text-slate-500">{item.branche} — {item.nature_sinistre}</p>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="font-bold text-sm text-slate-700">{fmtCurrency(item.mt_evaluation)}</span>
                          <span className="px-2 py-1 rounded-lg text-[11px] font-bold text-white" style={{ background: riskColor(item.risk_level) }}>
                            {item.risk_level}
                          </span>
                          <span className="text-sm font-black text-rose-600">{fmt(item.fraud_score * 100, '%')}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {fraudData.score_distribution && (
                <div className="bg-white rounded-2xl border border-slate-100 p-5">
                  <SectionTitle icon="📊">Distribution des Scores Fraude</SectionTitle>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={fraudData.score_distribution}>
                      <XAxis dataKey="bin" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ borderRadius: '10px', border: 'none' }} />
                      <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                        {fraudData.score_distribution.map((_, i) => (
                          <Cell key={i} fill={i < fraudData.score_distribution.length * 0.7 ? '#10b981' : i < fraudData.score_distribution.length * 0.9 ? '#f59e0b' : '#ef4444'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* ANOMALY view */}
      {view === 'anomaly' && (
        <div>
          <div className="flex justify-between items-center bg-white border border-slate-200 p-5 rounded-2xl shadow-sm mb-5">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <h4 className="font-bold text-slate-800 text-lg">Anomalies Portefeuille</h4>
                <Badge label="4 Algorithmes" color="blue" />
              </div>
              <p className="text-xs text-slate-500 mb-3">Isolation Forest · LOF · Autoencoder · DBSCAN — Score consensus 0–4</p>
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">Sensibilité</label>
              <select value={contam} onChange={e => setContam(Number(e.target.value))}
                className="bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm w-64 focus:ring-2 focus:ring-[#004A8D]/30">
                <option value={0.03}>Strict — 3%</option>
                <option value={0.05}>Standard — 5%</option>
                <option value={0.10}>Souple — 10%</option>
                <option value={0.20}>Large — 20%</option>
              </select>
            </div>
            {!loading && anomalyData && !anomalyData.error && (
              <div className="flex gap-3">
                {[['score_4', '🚨', 'Score 4', 'rose'], ['score_3', '⚠️', 'Score 3', 'orange'], ['nb_anomalies', '📍', 'Total IF', 'blue']].map(([key, icon, lbl, tone]) => (
                  <div key={key} className={`flex flex-col items-center rounded-xl p-3 min-w-[70px] ${tone === 'rose' ? 'bg-rose-50 border border-rose-100' : tone === 'orange' ? 'bg-amber-50 border border-amber-100' : 'bg-sky-50 border border-sky-100'}`}>
                    <span className="text-xl font-black" style={{ color: tone === 'rose' ? '#ef4444' : tone === 'orange' ? '#f59e0b' : '#004A8D' }}>{anomalyData[key] ?? '—'}</span>
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mt-0.5">{lbl}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {loading ? <Spinner label="Analyse des anomalies..." /> :
           anomalyData?.error ? <ErrorBox msg={anomalyData.error} /> :
           (anomalyData?.anomalies ?? []).map((a, i) => (
            <div key={i} className="flex bg-white border border-slate-200 rounded-2xl overflow-hidden hover:shadow-md transition-all duration-200 mb-3">
              <div className={`w-20 p-4 flex flex-col justify-center items-center border-r border-slate-100 ${a.anomaly_score >= 3 ? 'bg-rose-50' : 'bg-amber-50'}`}>
                <span className={`text-2xl font-black ${a.anomaly_score >= 3 ? 'text-rose-500' : 'text-amber-500'}`}>{a.anomaly_score}/4</span>
                <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400 mt-1 text-center">Consensus</span>
              </div>
              <div className="flex-1 p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <span className="font-bold text-slate-800">Police #{a.id_police}</span>
                    <Badge label={a.branche || '—'} color="blue" />
                  </div>
                  <div className="flex gap-2">
                    {a.if_anomaly && <Badge label="IF" color="red" />}
                    {a.lof_anomaly && <Badge label="LOF" color="red" />}
                    {a.ae_anomaly && <Badge label="AE" color="red" />}
                    {a.dbscan_anomaly && <Badge label="DBSCAN" color="red" />}
                  </div>
                </div>
                <div className="grid grid-cols-3 md:grid-cols-5 gap-2 bg-slate-50 rounded-lg p-2">
                  {[
                    ['Loss Ratio', fmt(a.loss_ratio * 100, '%')],
                    ['Tx Impayé', fmt(a.taux_impaye * 100, '%')],
                    ['Nb Sinistres', fmt(a.nb_sinistres)],
                    ['Prime Nette', fmtCurrency(a.mt_pnet_total)],
                    ['IF Score', fmt(a.if_score)],
                  ].map(([lbl, val]) => (
                    <div key={lbl} className="text-center">
                      <p className="text-[9px] font-bold text-slate-400 uppercase">{lbl}</p>
                      <p className="text-sm font-bold text-slate-700">{val}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 4 — Forecast (Prophet + SARIMA + XGBoost + LSTM)
// Notebook: forecast_model.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function ForecastTab() {
  const [dept, setDept] = useState('AUTO');
  const [ind, setInd] = useState('primes_acquises_tnd');
  const [nbMois, setNbMois] = useState(6);
  const [model, setModel] = useState('prophet');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const indLabels = {
    primes_acquises_tnd: 'Primes Acquises (TND)',
    cout_sinistres_tnd:  'Coût Sinistres (TND)',
    nb_sinistres:        'Nombre de Sinistres',
    taux_resiliation:    'Taux de Résiliation',
    sp_ratio:            'Ratio S/P',
    impayes_tnd:         'Impayés (TND)',
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchJson(`${API}/api/v1/ml/forecast?departement=${dept}&indicateur=${ind}&nb_mois=${nbMois}&model=${model}`)
      .then(d => { if (!cancelled) setData(d); })
      .catch(e => { if (!cancelled) setData({ error: e.message }); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [dept, ind, nbMois, model]);

  const historique = Array.isArray(data?.historique) ? data.historique : [];
  const previsions  = Array.isArray(data?.previsions)  ? data.previsions  : [];

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Controls */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-wrap gap-4 items-end">
        {[
          ['Branche', dept, v => setDept(v), [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']]],
          ['Indicateur', ind, v => setInd(v), Object.entries(indLabels)],
          ['Modèle', model, v => setModel(v), [['prophet','🔮 Prophet'],['sarima','📊 SARIMA'],['xgboost','⚡ XGBoost'],['lstm','🧠 LSTM']]],
          ['Horizon', nbMois, v => setNbMois(Number(v)), [[3,'+3 mois'],[6,'+6 mois'],[12,'+12 mois']]],
        ].map(([label, val, setter, opts]) => (
          <div key={label} className="flex-1 min-w-[160px]">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">{label}</label>
            <select value={val} onChange={e => setter(e.target.value)}
              className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#004A8D]/30">
              {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
        ))}
      </div>

      <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
        {loading ? <Spinner label="Calcul de la projection en cours..." /> :
         data?.error ? <ErrorBox msg={data.error} /> : data ? (
          <div>
            <div className="flex justify-between items-end mb-5">
              <div>
                <h4 className="font-bold text-xl text-slate-800">{indLabels[data.indicateur] || data.indicateur}</h4>
                <p className="text-sm text-slate-500">{data.departement} — horizon {nbMois} mois</p>
              </div>
              <div className="flex gap-2">
                <Badge label={data.methode || model} color="sky" />
                {data.mape && <Badge label={`MAPE ${fmt(data.mape, '%')}`} color={data.mape < 10 ? 'green' : data.mape < 20 ? 'yellow' : 'red'} />}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="periode" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} dy={10} />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={v => v.toLocaleString()} axisLine={false} tickLine={false} width={80} />
                <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,.1)' }} formatter={v => Number(v).toLocaleString()} />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '16px' }} />
                <Line data={historique} dataKey="valeur" name="Historique Réel" stroke={BRAND} strokeWidth={3} dot={false} isAnimationActive={false} />
                <Line data={previsions} dataKey="valeur" name="Prévision IA" stroke={ACCENT} strokeWidth={3} strokeDasharray="5 5" dot={{ r: 3, fill: ACCENT, strokeWidth: 0 }} />
                {previsions[0]?.valeur_max && <Line data={previsions} dataKey="valeur_max" name="Borne Haute" stroke="#fdba74" strokeWidth={1} dot={false} strokeDasharray="3 3" />}
                {previsions[0]?.valeur_min && <Line data={previsions} dataKey="valeur_min" name="Borne Basse" stroke="#fdba74" strokeWidth={1} dot={false} strokeDasharray="3 3" />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 5 — Customer Segmentation
// Notebook: customer_segmentation.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function SegmentationTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetchJson(`${API}/api/v1/ml/segmentation/summary`)
      .then(d => { setData(d); if (d?.segments?.length) setSelected(d.segments[0]); })
      .catch(() => setData({ error: 'Endpoint non disponible' }))
      .finally(() => setLoading(false));
  }, []);

  const segColor = { Champions: '#004A8D', 'Fidèles': '#10b981', 'Potentiels': '#F38F1D', 'À Risque': '#ef4444', Perdus: '#94a3b8', 'Nouveaux': '#8b5cf6', default: '#64748b' };
  const getColor = (name) => segColor[name] || segColor.default;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="bg-gradient-to-r from-[#F38F1D]/6 to-transparent p-5 rounded-2xl border border-[#F38F1D]/20">
        <div className="flex items-center gap-3 mb-1">
          <h4 className="font-bold text-orange-800 text-lg">Segmentation Client</h4>
          <Badge label="K-Means + RFM" color="yellow" />
        </div>
        <p className="text-sm text-slate-500">
          RFM scoring (Recency · Frequency · Monetary) combiné à K-Means ML.
          Features : ancienneté, primes, sinistralité, taux impayé, multi-branches.
        </p>
      </div>

      {loading ? <Spinner label="Chargement des segments..." /> :
       data?.error ? <ErrorBox msg={data.error} /> : data ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Pie + segment list */}
          <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm">
            <SectionTitle icon="🥧">Répartition des Segments</SectionTitle>
            {data.segments && (
              <>
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie data={data.segments} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={70} strokeWidth={2}>
                      {data.segments.map((s, i) => <Cell key={i} fill={getColor(s.name)} />)}
                    </Pie>
                    <Tooltip formatter={(v, n) => [fmt(v), n]} contentStyle={{ borderRadius: '10px', border: 'none' }} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-1.5 mt-2">
                  {data.segments.map((s, i) => (
                    <button key={i} onClick={() => setSelected(s)}
                      className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-all ${selected?.name === s.name ? 'bg-slate-100 font-bold' : 'hover:bg-slate-50'}`}>
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ background: getColor(s.name) }} />
                        <span>{s.name}</span>
                      </div>
                      <span className="text-slate-500 text-xs font-bold">{fmt(s.share_pct, '%')}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Selected segment detail */}
          <div className="lg:col-span-2 bg-white p-5 rounded-2xl border border-slate-100 shadow-sm">
            {selected ? (
              <div>
                <div className="flex items-center gap-3 mb-5">
                  <div className="w-3 h-3 rounded-full" style={{ background: getColor(selected.name) }} />
                  <h4 className="font-bold text-xl text-slate-800">{selected.name}</h4>
                  <Badge label={`${fmt(selected.count)} clients`} color="gray" />
                  <Badge label={`${fmt(selected.share_pct, '%')}`} color="blue" />
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-5">
                  {[
                    ['Prime Moy.', fmtCurrency(selected.avg_prime)],
                    ['Ancienneté', `${fmt(selected.avg_anciennete_jours / 365, '')} ans`],
                    ['Taux Churn', fmt(selected.avg_churn_risk * 100, '%')],
                    ['Taux Impayé', fmt(selected.avg_taux_impaye * 100, '%')],
                    ['SP Ratio', fmt(selected.avg_sp_ratio * 100, '%')],
                    ['LTV Estimée', fmtCurrency(selected.avg_ltv)],
                  ].map(([lbl, val]) => (
                    <div key={lbl} className="bg-slate-50 rounded-xl p-3">
                      <p className="text-[10px] font-bold text-slate-400 uppercase">{lbl}</p>
                      <p className="text-base font-black text-slate-700 mt-0.5">{val || '—'}</p>
                    </div>
                  ))}
                </div>
                {selected.radar && (
                  <ResponsiveContainer width="100%" height={200}>
                    <RadarChart data={selected.radar}>
                      <PolarGrid stroke="#e2e8f0" />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
                      <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 1]} />
                      <Radar name={selected.name} dataKey="value" stroke={getColor(selected.name)} fill={getColor(selected.name)} fillOpacity={0.25} strokeWidth={2} />
                    </RadarChart>
                  </ResponsiveContainer>
                )}
                {selected.action && (
                  <div className="mt-3 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800 font-medium">
                    💡 <strong>Action recommandée :</strong> {selected.action}
                  </div>
                )}
              </div>
            ) : <div className="flex items-center justify-center h-full text-slate-400">Sélectionnez un segment</div>}
          </div>
        </div>
       ) : null}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 6 — Risk Scoring & Claim Severity
// Notebooks: risk_scoring_pricing.ipynb + claim_severity.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function RiskPricingTab() {
  const [view, setView] = useState('risk');
  const [form, setForm] = useState({
    branche: 'AUTO', bonus_malus: 1.0, puissance: 6, age_vehicule: 5,
    age_client: 40, nb_sinistres_hist: 0, mt_pnet: 1200,
    nature_sinistre: 'MATERIEL', mt_evaluation: 5000,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tableData, setTableData] = useState(null);

  useEffect(() => {
    fetchJson(`${API}/api/v1/ml/risk/table`)
      .then(d => setTableData(d))
      .catch(() => {});
  }, []);

  const submit = async () => {
    setLoading(true); setResult(null);
    const endpoint = view === 'risk' ? '/api/v1/ml/risk/score' : '/api/v1/ml/claim/predict';
    const payload = view === 'risk'
      ? { branche: form.branche, bonus_malus: Number(form.bonus_malus), puissance: Number(form.puissance), age_vehicule: Number(form.age_vehicule), age_client: Number(form.age_client), nb_sinistres_hist: Number(form.nb_sinistres_hist), mt_pnet: Number(form.mt_pnet) }
      : { branche: form.branche, nature_sinistre: form.nature_sinistre, mt_evaluation: Number(form.mt_evaluation), age_client: Number(form.age_client), bonus_malus: Number(form.bonus_malus) };
    try {
      const d = await fetchJson(`${API}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      setResult(d);
    } catch (e) { setResult({ error: e.message }); } finally { setLoading(false); }
  };

  const scoreColor = (s) => s > 700 ? '#ef4444' : s > 400 ? '#f59e0b' : '#10b981';

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Sub-tabs */}
      <div className="flex gap-2 p-1 bg-slate-100 rounded-xl w-fit">
        {[['risk', '🎯 Risk Scoring', 'Score 0–1000 + Prime Technique'], ['claim', '💰 Claim Severity', 'Prédiction Coût Sinistre']].map(([id, label, sub]) => (
          <button key={id} onClick={() => { setView(id); setResult(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${view === id ? 'bg-white shadow text-[#004A8D]' : 'text-slate-500 hover:text-slate-700'}`}>

            {label} <span className="text-[9px] opacity-50 font-normal ml-1">{sub}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon={view === 'risk' ? '🎯' : '💰'}>{view === 'risk' ? 'Simulateur de Risque' : 'Simulateur Sinistre'}</SectionTitle>
          <div className="grid grid-cols-2 gap-3">
            {(
              view === 'risk' ? [
                ['branche', 'Branche', 'select', [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']]],
                ['bonus_malus', 'Bonus/Malus', 'number'],
                ['puissance', 'Puissance (CV)', 'number'],
                ['age_vehicule', 'Âge Véhicule', 'number'],
                ['age_client', 'Âge Client', 'number'],
                ['nb_sinistres_hist', 'Sinistres Hist.', 'number'],
                ['mt_pnet', 'Prime Nette (TND)', 'number'],
              ] : [
                ['branche', 'Branche', 'select', [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']]],
                ['nature_sinistre', 'Nature', 'select', [['MATERIEL','Matériel'],['CORPOREL','Corporel'],['MIXTE','Mixte'],['INCENDIE','Incendie']]],
                ['mt_evaluation', 'Montant Déclaré (TND)', 'number'],
                ['age_client', 'Âge Client', 'number'],
                ['bonus_malus', 'Bonus/Malus', 'number'],
              ]
            ).map(([key, label, type, opts]) => (
              <div key={key}>
                <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">{label}</label>
                {opts ? (
                  <select value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })}
                    className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#004A8D]/30">
                    {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                  </select>
                ) : (
                  <input type="number" step="any" value={form[key]} onChange={e => setForm({ ...form, [key]: e.target.value })}
                    className="w-full bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#004A8D]/30" />
                )}
              </div>
            ))}
          </div>
          <button onClick={submit} disabled={loading}
            className="w-full mt-5 py-3 bg-[#004A8D] text-white rounded-xl font-bold hover:bg-blue-800 disabled:opacity-50 transition-all">
            {loading ? '⏳ Calcul...' : view === 'risk' ? '⚡ Calculer Score de Risque' : '🔮 Prédire Coût Sinistre'}
          </button>
        </div>

        {/* Result */}
        <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 flex flex-col items-center justify-center min-h-[300px]">
          {result ? (
            result.error ? <ErrorBox msg={result.error} /> : (
              <div className="w-full animate-in zoom-in duration-300">
                {view === 'risk' ? (
                  <div className="text-center">
                    <p className="text-xs font-black text-slate-400 tracking-widest uppercase mb-4">Score de Risque</p>
                    <div className="mx-auto w-28 h-28 rounded-full flex flex-col items-center justify-center border-4 mb-4"
                      style={{ borderColor: scoreColor(result.risk_score), background: `${scoreColor(result.risk_score)}15`, color: scoreColor(result.risk_score) }}>
                      <span className="text-3xl font-black">{fmt(result.risk_score)}</span>
                      <span className="text-[9px] font-bold uppercase">/1000</span>
                    </div>
                    <p className="font-bold text-lg text-slate-800 mb-3">{result.risk_label || '—'}</p>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="bg-white rounded-xl p-3 border border-slate-100">
                        <p className="text-[10px] text-slate-400 font-bold uppercase">Prime Technique</p>
                        <p className="text-xl font-black text-[#004A8D]">{fmtCurrency(result.prime_technique)}</p>
                      </div>
                      <div className="bg-white rounded-xl p-3 border border-slate-100">
                        <p className="text-[10px] text-slate-400 font-bold uppercase">Loading Factor</p>
                        <p className="text-xl font-black text-[#F38F1D]">{fmt(result.loading_factor * 100, '%')}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center">
                    <p className="text-xs font-black text-slate-400 tracking-widest uppercase mb-4">Coût Sinistre Prédit</p>
                    <p className="text-4xl font-black text-[#004A8D] mb-2">{fmtCurrency(result.predicted_severity)}</p>
                    <p className="text-slate-500 text-sm mb-4">IC 90% : [{fmtCurrency(result.ci_low)} — {fmtCurrency(result.ci_high)}]</p>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-white rounded-xl p-3 border border-slate-100">
                        <p className="text-[10px] text-slate-400 font-bold uppercase">Modèle</p>
                        <p className="font-bold text-slate-700 text-sm">{result.model_used || 'GradientBoosting'}</p>
                      </div>
                      <div className="bg-white rounded-xl p-3 border border-slate-100">
                        <p className="text-[10px] text-slate-400 font-bold uppercase">Réserve Recommandée</p>
                        <p className="font-black text-rose-600">{fmtCurrency(result.reserve_recommandee)}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          ) : (
            <div className="text-slate-400 flex flex-col items-center gap-3">
              <span className="text-5xl opacity-30">{view === 'risk' ? '🎯' : '💰'}</span>
              <p className="text-sm font-medium">Remplissez le formulaire pour calculer</p>
            </div>
          )}
        </div>
      </div>

      {/* Risk table */}
      {view === 'risk' && tableData?.table && (
        <div className="bg-white rounded-2xl border border-slate-100 p-5">
          <SectionTitle icon="📋">Table de Risque — Branche × Type Véhicule</SectionTitle>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-100">
                  {['Branche', 'Genre Véhicule', 'Fréq. Sin.', 'Sévérité Moy.', 'S/P Ratio', 'Prime Technique'].map(h => (
                    <th key={h} className="text-left py-2 px-3 font-bold text-slate-400 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableData.table.slice(0, 10).map((row, i) => (
                  <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                    <td className="py-2 px-3 font-bold text-slate-700">{row.branche}</td>
                    <td className="py-2 px-3">{row.genre_vehicule}</td>
                    <td className="py-2 px-3">{fmt(row.freq_sin_moy * 100, '%')}</td>
                    <td className="py-2 px-3">{fmtCurrency(row.sev_moy)}</td>
                    <td className={`py-2 px-3 font-bold ${row.sp_ratio_moy > 1 ? 'text-rose-600' : 'text-emerald-600'}`}>{fmt(row.sp_ratio_moy * 100, '%')}</td>
                    <td className="py-2 px-3 font-black text-[#004A8D]">{fmtCurrency(row.prime_technique)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB 7 — Data Drift (Evidently AI)
// Notebook: data_drift_evidently.ipynb
// ─────────────────────────────────────────────────────────────────────────────
function DriftTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [nbMoisRef, setNbMoisRef] = useState(12);
  const [nbMoisCur, setNbMoisCur] = useState(6);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchJson(`${API}/api/v1/ml/drift?nb_mois_reference=${nbMoisRef}&nb_mois_courant=${nbMoisCur}`)
      .then(d => { if (!cancelled) setData(d); })
      .catch(e => { if (!cancelled) setData({ error: e.message }); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [nbMoisRef, nbMoisCur]);

  const features = Array.isArray(data?.features)    ? data.features    : [];
  const comparaison = Array.isArray(data?.comparaison) ? data.comparaison : [];
  const psiData = Array.isArray(data?.psi_features) ? data.psi_features : [];

  const psiColor = (v) => v >= 0.2 ? '#ef4444' : v >= 0.1 ? '#f59e0b' : '#10b981';
  const psiLabel = (v) => v >= 0.2 ? '🚨 Critique' : v >= 0.1 ? '⚠️ Modéré' : '✅ Stable';

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Controls */}
      <div className="flex gap-4 bg-white border border-slate-200 p-4 rounded-xl shadow-sm items-end flex-wrap">
        <div>
          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">Fenêtre Référence</label>
          <select value={nbMoisRef} onChange={e => setNbMoisRef(Number(e.target.value))}
            className="bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#004A8D]/30">
            {[6,12,18,24].map(v => <option key={v} value={v}>{v} mois</option>)}
          </select>
        </div>
        <div>
          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">Fenêtre Courante</label>
          <select value={nbMoisCur} onChange={e => setNbMoisCur(Number(e.target.value))}
            className="bg-slate-50 border-0 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-[#004A8D]/30">
            {[3,6,9,12].map(v => <option key={v} value={v}>{v} mois</option>)}
          </select>
        </div>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Moteur</span>
          <Badge label="Evidently AI v0.7" color="blue" />
          <Badge label="KS + Chi² + PSI" color="gray" />
        </div>
      </div>

      {loading ? <Spinner label="Évaluation Evidently AI en cours..." /> :
       data?.error ? <ErrorBox msg={data.error} /> : data ? (
        <>
          {/* Global status */}
          <div className={`p-6 rounded-2xl border flex items-center justify-between shadow-sm ${data.dataset_drift ? 'bg-rose-50 border-rose-200' : 'bg-emerald-50 border-emerald-200'}`}>
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">{data.dataset_drift ? '⚠️' : '✅'}</span>
                <h3 className={`font-black text-xl ${data.dataset_drift ? 'text-rose-800' : 'text-emerald-800'}`}>
                  {data.dataset_drift ? 'DATA DRIFT DÉTECTÉ' : 'DONNÉES STABLES'}
                </h3>
                <Badge label="Evidently" color={data.dataset_drift ? 'red' : 'green'} />
              </div>
              <p className={`text-sm ${data.dataset_drift ? 'text-rose-600' : 'text-emerald-600'} font-medium`}>{data.message}</p>
            </div>
            <div className="text-right">
              <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Taux de Dérive</p>
              <p className="text-4xl font-black">{fmt(data.share_drift * 100, '%')}</p>
              <p className="text-xs text-slate-500 mt-1">{data.nb_drifted} / {data.nb_features} features</p>
            </div>
          </div>

          {/* PSI bar chart */}
          {psiData.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-sm">
              <SectionTitle icon="📊">PSI par Feature (Population Stability Index)</SectionTitle>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={psiData} layout="vertical" margin={{ left: 100, right: 40 }}>
                  <XAxis type="number" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(2)} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '10px', border: 'none' }} formatter={v => [v.toFixed(4), 'PSI']} />
                  <ReferenceLine x={0.1} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: '0.10', fontSize: 10, fill: '#f59e0b' }} />
                  <ReferenceLine x={0.2} stroke="#ef4444" strokeDasharray="4 2" label={{ value: '0.20', fontSize: 10, fill: '#ef4444' }} />
                  <Bar dataKey="psi" radius={[0, 4, 4, 0]}>
                    {psiData.map((entry, i) => <Cell key={i} fill={psiColor(entry.psi)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-2 justify-end">
                {[['#10b981', '< 0.10 Stable'], ['#f59e0b', '0.10–0.20 Modéré'], ['#ef4444', '> 0.20 Critique']].map(([c, l]) => (
                  <div key={l} className="flex items-center gap-1.5 text-xs text-slate-500">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: c }} /> {l}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Features grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {features.map((f, i) => {
              const comp = comparaison.find(c => c.feature === f.feature);
              const psiEntry = psiData.find(p => p.feature === f.feature);
              return (
                <div key={i} className={`p-5 rounded-2xl border bg-white hover:shadow-md transition ${f.drift_detecte ? 'border-rose-200' : 'border-slate-100'}`}>
                  <div className="flex justify-between items-center mb-3">
                    <p className="font-bold text-sm text-slate-700">{f.feature.replace(/_/g, ' ').toUpperCase()}</p>
                    <div className="flex gap-1.5">
                      <Badge label={f.drift_detecte ? 'Drift' : 'Stable'} color={f.drift_detecte ? 'red' : 'green'} />
                      {psiEntry && <Badge label={psiLabel(psiEntry.psi)} color={psiEntry.psi >= 0.2 ? 'red' : psiEntry.psi >= 0.1 ? 'yellow' : 'green'} />}
                    </div>
                  </div>
                  {comp && (
                    <div className="space-y-2">
                      <div className="flex gap-2">
                        <div className="flex-1 bg-slate-50 rounded-lg p-2 text-center">
                          <p className="text-[9px] font-bold text-slate-400 uppercase">Référence</p>
                          <p className="font-bold text-slate-700 text-sm">{fmt(comp.moyenne_ref)}</p>
                        </div>
                        <div className="flex items-center text-slate-300 text-xs">→</div>
                        <div className={`flex-1 rounded-lg p-2 text-center ${f.drift_detecte ? 'bg-rose-50' : 'bg-slate-50'}`}>
                          <p className="text-[9px] font-bold text-slate-400 uppercase">Courant</p>
                          <p className={`font-bold text-sm ${f.drift_detecte ? 'text-rose-600' : 'text-slate-700'}`}>{fmt(comp.moyenne_cur)}</p>
                        </div>
                      </div>
                      <p className={`text-xs text-center font-bold ${Math.abs(comp.variation_pct) > 10 ? 'text-rose-500' : 'text-slate-400'}`}>
                        {comp.variation_pct > 0 ? '+' : ''}{fmt(comp.variation_pct, '%')} variation
                      </p>
                      {psiEntry && (
                        <div className="mt-1">
                          <div className="flex justify-between text-[10px] text-slate-400 mb-0.5">
                            <span>PSI</span><span>{psiEntry.psi.toFixed(4)}</span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-1.5">
                            <div className="h-1.5 rounded-full transition-all" style={{ width: `${Math.min(100, (psiEntry.psi / 0.3) * 100)}%`, background: psiColor(psiEntry.psi) }} />
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      ) : null}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
const TABS = [
  { id: 'impaye',       label: 'Risque Impayé',       icon: '💳', notebook: 'impaye_risk_scoring' },
  { id: 'churn',        label: 'Prédiction Churn',     icon: '📉', notebook: 'churn_prediction_v3' },
  { id: 'fraud',        label: 'Fraude & Anomalies',   icon: '🚨', notebook: 'fraud_detection + anomaly_detection' },
  { id: 'forecast',     label: 'Projections IA',       icon: '📈', notebook: 'forecast_model' },
  { id: 'segmentation', label: 'Segmentation',         icon: '👥', notebook: 'customer_segmentation' },
  { id: 'risk',         label: 'Risque & Pricing',     icon: '🎯', notebook: 'risk_scoring_pricing + claim_severity' },
  { id: 'drift',        label: 'Data Drift',           icon: '⚖️', notebook: 'data_drift_evidently' },
];

export default function MLOpsContent() {
  const [tab, setTab] = useState('impaye');

  return (
    <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden flex flex-col md:flex-row min-h-[750px] m-4 lg:m-8">
      {/* Sidebar */}
      <div className="w-full md:w-64 bg-slate-50/80 border-r border-slate-100 flex flex-col p-4 md:p-6 space-y-1.5 shrink-0">
        <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-widest pl-3 mb-4">
          Workflows MLOps
        </h2>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`text-left px-4 py-3 rounded-xl text-sm font-bold transition-all duration-200 flex items-center gap-3 group ${
              tab === t.id
                ? 'bg-gradient-to-br from-[#004A8D] to-blue-800 text-white shadow-md shadow-[#004A8D]/20 scale-[1.02]'
                : 'text-slate-600 hover:bg-white hover:shadow-sm hover:text-slate-900 border border-transparent'
            }`}>
            <span className="text-lg">{t.icon}</span>
            <div>
              <p>{t.label}</p>
              <p className={`text-[9px] font-normal leading-tight mt-0.5 ${tab === t.id ? 'text-white/60' : 'text-slate-400'}`}>
                {t.notebook.split(' + ').map(n => n.replace(/_/g, ' ')).join(' · ')}
              </p>
            </div>
          </button>
        ))}
        {/* Model registry summary */}
        <div className="mt-auto pt-4 border-t border-slate-200">
          <p className="text-[9px] font-black text-slate-300 uppercase tracking-widest mb-2 pl-1">Registre Modèles</p>
          {[['💳 Impayé', 'XGBoost'], ['📉 Churn', 'XGBoost v3'], ['🚨 Fraude', 'IF+AE+LOF'], ['📈 Forecast', 'Prophet+ML'], ['👥 Segments', 'K-Means'], ['🎯 Pricing', 'GBM+GLM'], ['⚖️ Drift', 'Evidently']].map(([name, algo]) => (
            <div key={name} className="flex items-center justify-between py-1 px-1">
              <span className="text-[10px] text-slate-500 font-medium">{name}</span>
              <span className="text-[9px] text-slate-400 bg-slate-100 rounded px-1.5 py-0.5">{algo}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 p-6 md:p-8 overflow-y-auto bg-[#fcfdfd]">
        {tab === 'impaye'       && <ImpayeTab />}
        {tab === 'churn'        && <ChurnTab />}
        {tab === 'fraud'        && <FraudAnomalyTab />}
        {tab === 'forecast'     && <ForecastTab />}
        {tab === 'segmentation' && <SegmentationTab />}
        {tab === 'risk'         && <RiskPricingTab />}
        {tab === 'drift'        && <DriftTab />}
      </div>
    </div>
  );
}