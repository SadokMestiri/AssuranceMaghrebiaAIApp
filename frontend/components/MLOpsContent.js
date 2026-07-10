import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  LineChart, Line, CartesianGrid, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, Cell, PieChart, Pie
} from 'recharts';
import {
  CreditCard, TrendingDown, TrendingUp, ShieldAlert, Users, Target, Scale,
  Activity, BarChart2, FlaskConical, Bot, PieChart as PieChartIcon,
  Zap, Wand2, AlertTriangle, XCircle, Lightbulb, PlayCircle, DollarSign,
  Loader2, ChevronLeft, ChevronRight, Sparkles, Inbox, Info, Moon, Sun,
} from 'lucide-react';

const API =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://localhost:8000');

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
      {icon && <span className="opacity-60 flex-shrink-0">{icon}</span>}
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
      <Loader2 size={28} className="animate-spin opacity-40" />
      <p className="text-sm font-medium animate-pulse">{label}</p>
    </div>
  );
}

function ErrorBox({ msg }) {
  return (
    <div className="bg-rose-50 border border-rose-200 text-rose-700 px-4 py-3 rounded-xl text-sm flex items-center gap-2">
      <XCircle size={16} className="shrink-0" /> {msg}
    </div>
  );
}

function Pagination({ page, total, pageSize, onPage }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to   = Math.min(page * pageSize, total);
  if (total === 0) return null;
  return (
    <div className="flex items-center justify-between mt-3 px-1">
      <span className="text-[11px] text-slate-400">
        {from}–{to} <span className="opacity-70">sur</span> <span className="font-bold text-slate-600">{total}</span>
      </span>
      <div className="flex items-center gap-1">
        <button disabled={page === 1} onClick={() => onPage(page - 1)}
          className="px-3 py-1.5 rounded-lg text-[11px] font-bold border border-slate-200 text-slate-500 hover:border-[#004A8D]/40 hover:text-[#004A8D] disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center gap-1">
          <ChevronLeft size={13} /> Préc.
        </button>
        <span className="px-3 py-1 text-[11px] font-black text-[#004A8D] bg-[#004A8D]/6 rounded-lg">
          {page} / {totalPages}
        </span>
        <button disabled={page === totalPages} onClick={() => onPage(page + 1)}
          className="px-3 py-1.5 rounded-lg text-[11px] font-bold border border-slate-200 text-slate-500 hover:border-[#004A8D]/40 hover:text-[#004A8D] disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center gap-1">
          Suiv. <ChevronRight size={13} />
        </button>
      </div>
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
    mt_pnet: 1500, mt_rc: 500, mt_commission: 150, mt_fga: 0, mt_timbre: 0, bonus_malus: 0,
    type_police: 'I', client_nb_impayes: 0, client_mt_impaye_tot: 0,
    police_nb_impayes_hist: 0, police_taux_impaye_hist: 0, police_nb_annulations: 0,
    police_nb_sinistres: 0, police_sp_ratio: 0, age_client: 40,
    type_personne: 'P', sexe: 'M', puissance: 6, valeur_vehicule: 8000, code_usage: 0,
  });

  const m = trainResult?.model?.metrics ?? {};
  const score = Number(explainResult?.probability_impaye ?? explainResult?.probability ?? 0);
  const threshold = Number(explainResult?.threshold ?? 0.5);
  const predicted = Number(explainResult?.predicted_label ?? explainResult?.prediction ?? (score >= threshold ? 1 : 0));
  const tone = predicted === 1
    ? 'border-rose-400 bg-rose-50 text-rose-600'
    : 'border-emerald-400 bg-emerald-50 text-emerald-600';

  const train = async () => {
    setTraining(true); setTrainError(null); setTrainResult(null);
    try {
      const data = await fetchJson(`${API}/api/v1/ml/notebook-model-info`);
      if (data.error || data.status === 'error') throw new Error(data.error || 'Échec');
      setTrainResult(data);
    } catch (e) { setTrainError(e.message); } finally { setTraining(false); }
  };

  const predict = async () => {
    setExplaining(true); setExplainResult(null);
    try {
      const data = await fetchJson(`${API}/api/v1/ml/predict-notebook`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          annee_echeance: Number(form.annee_echeance),
          mois_echeance: Number(form.mois_echeance),
          mt_pnet: Number(form.mt_pnet),
          mt_rc: Number(form.mt_rc),
          mt_commission: Number(form.mt_commission),
          mt_fga: Number(form.mt_fga),
          mt_timbre: Number(form.mt_timbre),
          bonus_malus: Number(form.bonus_malus),
          client_nb_impayes: Number(form.client_nb_impayes),
          client_mt_impaye_tot: Number(form.client_mt_impaye_tot),
          police_nb_impayes_hist: Number(form.police_nb_impayes_hist),
          police_taux_impaye_hist: Number(form.police_taux_impaye_hist),
          police_nb_annulations: Number(form.police_nb_annulations),
          police_nb_sinistres: Number(form.police_nb_sinistres),
          police_sp_ratio: Number(form.police_sp_ratio),
          age_client: Number(form.age_client),
          puissance: Number(form.puissance),
          valeur_vehicule: Number(form.valeur_vehicule),
          code_usage: Number(form.code_usage),
        }),
      });
      setExplainResult(data);
    } catch (e) { setExplainResult({ error: e.message }); } finally { setExplaining(false); }
  };

  const scenarios = [
    {
      label: 'Client sain',
      tone: 'green',
      values: {
        mt_pnet: 1500, mt_commission: 150, mt_fga: 0, mt_timbre: 0,
        client_nb_impayes: 0, client_mt_impaye_tot: 0,
        police_nb_impayes_hist: 0, police_taux_impaye_hist: 0,
        police_nb_annulations: 0, age_client: 40, valeur_vehicule: 8000,
      },
    },
    {
      label: 'Prime elevee',
      tone: 'blue',
      values: {
        mt_pnet: 150000, mt_commission: 1500, mt_fga: 0, mt_timbre: 0,
        client_nb_impayes: 0, client_mt_impaye_tot: 0,
        police_nb_impayes_hist: 0, police_taux_impaye_hist: 0,
        police_nb_annulations: 0, age_client: 40, valeur_vehicule: 8000,
      },
    },
    {
      label: 'Historique impaye',
      tone: 'yellow',
      values: {
        mt_pnet: 1500, mt_commission: 150, mt_fga: 0, mt_timbre: 0,
        client_nb_impayes: 2, client_mt_impaye_tot: 3000,
        police_nb_impayes_hist: 1, police_taux_impaye_hist: 0.4,
        police_nb_annulations: 0, age_client: 40, valeur_vehicule: 8000,
      },
    },
    {
      label: 'Risque metier',
      tone: 'red',
      values: {
        mt_pnet: 1500, mt_commission: 150, mt_fga: 0, mt_timbre: 0,
        client_nb_impayes: 3, client_mt_impaye_tot: 5000,
        police_nb_impayes_hist: 2, police_taux_impaye_hist: 0.8,
        police_nb_annulations: 1, age_client: 40, valeur_vehicule: 8000,
      },
    },
  ];

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
              <Badge label="LightGBM calibre" color="blue" />
            </div>
            <p className="text-sm text-slate-500">
              Utilise le modele retenu dans impaye_risk_scoring.ipynb :
              LightGBM tune avec calibration isotonic et seuil metier.
            </p>
          </div>
          <button onClick={train} disabled={training}
            className="shrink-0 px-5 py-2.5 bg-[#004A8D] text-white rounded-xl shadow-md font-semibold hover:bg-blue-800 disabled:opacity-50 transition-all flex items-center gap-2 text-sm">
            {training ? <Loader2 size={15} className="animate-spin" /> : <PlayCircle size={15} />}
            {training ? 'Chargement...' : 'Verifier modele notebook'}
          </button>
        </div>
        {trainError && <div className="mt-3"><ErrorBox msg={trainError} /></div>}
        {trainResult && (
          <div className="mt-4 bg-emerald-50 border border-emerald-200 rounded-xl p-4">
            <p className="text-emerald-800 font-bold mb-1">Modele notebook charge - {trainResult.model?.model || 'LightGBM calibre'}</p>
            <p className="text-xs text-emerald-700/70 mb-3">
              Metriques au seuil metier {fmt((m.metrics_threshold ?? trainResult.model?.seuil_optimal ?? 0.15) * 100, '%')}
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[['Accuracy', m.Accuracy], ['Precision', m.Precision], ['Recall', m.Recall], ['F1', m.F1]].map(([lbl, v]) => (
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
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon={<FlaskConical size={16} />}>Simulateur d&apos;Inférence</SectionTitle>
          <div className="flex flex-wrap gap-2 mb-4">
            {scenarios.map(s => (
              <button
                key={s.label}
                type="button"
                onClick={() => setForm(prev => ({ ...prev, ...s.values }))}
                className="px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-xs font-bold text-slate-600 hover:bg-white hover:border-[#F38F1D]/40 transition-all"
              >
                {s.label}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-4">
            {field('branche', 'Branche', 'text', [['AUTO', 'Automobile'], ['IRDS', 'IRDS'], ['SANTE', 'Santé']])}
            {field('periodicite', 'Périodicité', 'text', [['A', 'Annuel'], ['S', 'Semestriel'], ['T', 'Trimestriel'], ['C', 'Comptant']])}
            {field('mois_echeance', 'Mois Échéance')}
            {field('mt_pnet', 'Prime Nette (TND)')}
            {field('mt_commission', 'Commission (TND)')}
            {field('mt_fga', 'FGA (TND)')}
            {field('mt_timbre', 'Timbre (TND)')}
            {field('client_nb_impayes', 'Nb impayes client')}
            {field('client_mt_impaye_tot', 'Total impaye client')}
            {field('police_nb_impayes_hist', 'Nb impayes police')}
            {field('police_taux_impaye_hist', 'Taux impaye police')}
            {field('police_nb_annulations', 'Annulations police')}
            {field('age_client', 'Age client')}
            {field('valeur_vehicule', 'Valeur vehicule')}
          </div>
          <button onClick={predict} disabled={explaining}
            className="w-full mt-5 py-3 bg-[#F38F1D] text-white rounded-xl font-bold hover:bg-[#d97d16] disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-md shadow-[#F38F1D]/20">
            {explaining
              ? <><Loader2 size={15} className="animate-spin" /> Calcul en cours...</>
              : <><Zap size={15} /> Évaluer le Risque d&apos;Impayé</>}
          </button>
        </div>

        <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100 flex flex-col items-center justify-center min-h-[320px]">
          {explainResult ? (
            explainResult.error ? <ErrorBox msg={explainResult.error} /> : (
              <div className="text-center w-full animate-in zoom-in duration-300">
                <p className="text-xs font-black text-slate-400 tracking-widest uppercase mb-5">Résultat Inférence</p>
                <div className={`mx-auto w-32 h-32 rounded-full flex flex-col items-center justify-center border-4 shadow-inner mb-5 ${tone}`}>
                  <span className="text-3xl font-black">{predicted === 1 ? '!' : 'OK'}</span>
                  <span className="text-[10px] font-bold uppercase mt-1">Decision</span>
                </div>
                <h4 className="text-lg font-bold text-slate-800 mb-2">
                  {predicted === 1 ? '⚠️ Impayé Anticipé' : '✅ Paiement Fiable'}
                </h4>
                <p className="text-slate-400 text-xs">Seuil décisionnel : <strong>{fmt(threshold * 100, '%')}</strong></p>
              </div>
            )
          ) : (
            <div className="text-slate-400 flex flex-col items-center gap-3">
              <Bot size={48} className="opacity-20" />
              <p className="text-sm font-medium">Simulez une quittance pour voir le score</p>
            </div>
          )}
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
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({
    branche: 'AUTO', bonus_malus: 1.0, nb_quittances: 4,
    mt_pnet: 1200, taux_impaye: 0, nb_sinistres: 0,
  });
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    Promise.all([
      fetchJson(`${API}/api/v1/ml/churn/summary`),
      fetchJson(`${API}/api/v1/ml/churn/model-info`).catch(() => null),
    ])
      .then(([summary, info]) => {
        setData(summary);
        setModelInfo(info);
      })
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
          <Badge label={modelInfo?.source || data?.model_source || 'Notebook v3'} color="purple" />
          <Badge label={`Seuil F1 ${fmt((modelInfo?.threshold ?? data?.threshold ?? 0) * 100, '%')}`} color="gray" />
        </div>
        <p className="text-sm text-slate-500">
          Identifie les polices à risque de résiliation à partir des features contrat, sinistralité,
          historique de paiement et profil client. Cible : <code className="bg-slate-100 px-1 rounded text-xs">CHURN = 1</code>.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stats panel */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon={<BarChart2 size={16} />}>Vue Portefeuille — Risque Churn</SectionTitle>
          {loading ? <Spinner label="Chargement des statistiques churn..." /> :
           data?.error ? <ErrorBox msg={data.error} /> : data ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Taux Résiliation Réel" value={fmt(data.taux_churn_pct, '%')} sub={`${fmt(data.nb_churn)} polices résiliées / annulées`} tone="red" />
                <StatCard label="Polices Analysées" value={fmt(data.nb_polices)} sub="en portefeuille actif" tone="blue" />
              </div>
              {(modelInfo?.metrics || data.model_metrics) && (
                <div className="grid grid-cols-4 gap-2">
                  {[
                    ['Accuracy', 'accuracy'],
                    ['Precision', 'precision'],
                    ['Recall', 'recall'],
                    ['F1', 'f1'],
                  ].map(([label, key]) => {
                    const metrics = modelInfo?.metrics || data.model_metrics || {};
                    return (
                      <div key={key} className="bg-violet-50 rounded-lg p-2 text-center border border-violet-100">
                        <p className="text-[10px] font-bold text-violet-400 uppercase">{label}</p>
                        <p className="text-sm font-black text-violet-800">{fmt((metrics[key] || 0) * 100, '%')}</p>
                      </div>
                    );
                  })}
                </div>
              )}
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
          <SectionTitle icon={<Target size={16} />}>Simulateur Churn</SectionTitle>
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
            className="w-full mt-4 py-3 inline-flex items-center justify-center gap-2 bg-gradient-to-br from-violet-600 to-indigo-700 text-white rounded-xl font-bold shadow-md shadow-violet-600/20 hover:shadow-lg hover:shadow-violet-600/30 disabled:opacity-50 disabled:shadow-none transition-all">
            {predicting
            ? <><Loader2 size={15} className="animate-spin" /> Calcul...</>
            : <><Wand2 size={15} /> Prédire le Risque de Churn</>}
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
const FRAUD_ANOMALY_PAGE_SIZE = 10;

function FraudAnomalyTab() {
  const [view, setView] = useState('fraud');
  const [fraudData, setFraudData] = useState(null);
  const [anomalyData, setAnomalyData] = useState(null);
  const [contam, setContam] = useState(0.05);
  const [riskFilter, setRiskFilter] = useState(null);
  const [scoreFilter, setScoreFilter] = useState(1);
  const [loading, setLoading] = useState(false);
  const [fraudPage, setFraudPage] = useState(1);
  const [anomalyPage, setAnomalyPage] = useState(1);

  const loadFraud = useCallback(async (level = riskFilter) => {
    setLoading(true);
    setFraudPage(1);
    const qs = level ? `?risk_level=${level}&top_n=100` : '?top_n=100';
    const d = await fetchJson(`${API}/api/v1/ml/fraud/summary${qs}`).catch(() => ({ error: 'Indisponible' }));
    setFraudData(d);
    setLoading(false);
  }, [riskFilter]);

  const loadAnomaly = useCallback(async (ms = scoreFilter) => {
    setLoading(true);
    setAnomalyPage(1);
    const d = await fetchJson(`${API}/api/v1/ml/anomalies?contamination=${contam}&min_score=${ms}`).catch(e => ({ error: e.message }));
    setAnomalyData(d);
    setLoading(false);
  }, [contam, scoreFilter]);

  useEffect(() => { view === 'fraud' ? loadFraud() : loadAnomaly(); }, [view, loadFraud, loadAnomaly]);

  const riskColor = (level) => ({ Normal: '#10b981', 'Risque Modéré': '#f59e0b', 'Risque Élevé': '#f97316', Critique: '#ef4444' }[level] || '#94a3b8');

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Sub-tabs */}
      <div className="flex gap-2 p-1 bg-slate-100 rounded-xl w-fit">
        {[['fraud', 'Fraude', 'IF + AE + LOF'], ['anomaly', 'Anomalies Contrats', 'IF · LOF · AE · DBSCAN']].map(([id, label, sub]) => (
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
                  <div className="flex items-center justify-between mb-4">
                    <SectionTitle icon={<AlertTriangle size={16} />}>Sinistres Suspects</SectionTitle>
                    <div className="flex gap-1.5 flex-wrap">
                      {[
                        [null,       'Tous suspects',  fraudData.nb_critique + fraudData.nb_eleve + fraudData.nb_modere],
                        ['critique', 'Critique',       fraudData.nb_critique],
                        ['eleve',    'Risque Élevé',   fraudData.nb_eleve],
                        ['modere',   'Risque Modéré',  fraudData.nb_modere],
                        ['normal',   'Normal',         fraudData.nb_normal],
                      ].map(([lvl, label, count]) => (
                        <button key={String(lvl)} onClick={() => { setRiskFilter(lvl); loadFraud(lvl); }}
                          className={`px-2.5 py-1 rounded-lg text-[11px] font-bold border transition-all ${riskFilter === lvl ? 'text-white border-transparent' : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'}`}
                          style={riskFilter === lvl ? { background: riskColor(label) } : {}}>
                          {label} <span className="opacity-70">({fmt(count)})</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  {(() => {
                    const allFraud = fraudData.top_fraud;
                    const fraudSlice = allFraud.slice(
                      (fraudPage - 1) * FRAUD_ANOMALY_PAGE_SIZE,
                      fraudPage * FRAUD_ANOMALY_PAGE_SIZE
                    );
                    const absIdx = (i) => (fraudPage - 1) * FRAUD_ANOMALY_PAGE_SIZE + i;
                    return (
                      <>
                        <div className="space-y-2">
                          {fraudSlice.map((item, i) => (
                            <div key={absIdx(i)} className="flex items-center gap-3 p-3 rounded-xl bg-slate-50 hover:bg-slate-100 transition border border-transparent hover:border-slate-200">
                              {/* Score badge */}
                              <div className="w-12 h-12 rounded-xl flex flex-col items-center justify-center flex-shrink-0 text-white font-black text-sm"
                                style={{ background: riskColor(item.risk_level) }}>
                                {fmt(item.fraud_score * 100, '')}
                                <span className="text-[8px] font-bold opacity-80">score</span>
                              </div>
                              {/* Sinistre info */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-0.5">
                                  <span className="font-bold text-sm text-slate-800">{item.num_sinistre || `#${absIdx(i)+1}`}</span>
                                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: riskColor(item.risk_level) + '20', color: riskColor(item.risk_level) }}>{item.risk_level}</span>
                                </div>
                                <p className="text-xs text-slate-500 truncate">{item.branche} · {item.nature_sinistre}</p>
                              </div>
                              {/* Client info */}
                              <div className="text-right min-w-[130px]">
                                <p className="font-bold text-sm text-slate-700 truncate">{item.client_nom || '—'}</p>
                                <p className="text-[10px] text-slate-400">{item.type_personne === 'M' ? '🏢 Morale' : '👤 Physique'} · {item.client_ville || '—'}</p>
                              </div>
                              {/* Amount */}
                              <div className="text-right flex-shrink-0 min-w-[100px]">
                                <p className="font-black text-sm text-slate-800">{fmtCurrency(item.mt_evaluation)}</p>
                                <p className="text-[10px] text-slate-400">MT Évaluation</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <Pagination
                          page={fraudPage}
                          total={allFraud.length}
                          pageSize={FRAUD_ANOMALY_PAGE_SIZE}
                          onPage={setFraudPage}
                        />
                      </>
                    );
                  })()}
                </div>
              )}
              {fraudData.score_distribution && (() => {
                const p90 = fraudData.thresholds?.p90 ?? 0.13;
                const p95 = fraudData.thresholds?.p95 ?? 0.17;
                const p99 = fraudData.thresholds?.p99 ?? 0.29;
                const binColor = (i) => {
                  const upper = (i + 1) * 0.1;
                  if (upper <= p90) return '#10b981';
                  if (upper <= p95) return '#f59e0b';
                  if (upper <= p99) return '#f97316';
                  return '#ef4444';
                };
                // log1p transform for display so the tail is visible
                const logData = fraudData.score_distribution.map(b => ({
                  ...b, count_log: Math.log1p(b.count), count_raw: b.count,
                }));
                return (
                  <div className="bg-white rounded-2xl border border-slate-100 p-5">
                    <div className="flex items-center justify-between mb-3">
                      <SectionTitle icon={<BarChart2 size={16} />}>Distribution des Scores Fraude</SectionTitle>
                      <div className="flex gap-3 text-[10px] font-bold">
                        {[['Normal', '#10b981'], ['Modéré p90', '#f59e0b'], ['Élevé p95', '#f97316'], ['Critique p99', '#ef4444']].map(([l, c]) => (
                          <span key={l} className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full inline-block" style={{ background: c }} />{l}
                          </span>
                        ))}
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-400 mb-2">Échelle logarithmique (log1p) — seuils p90={fmt(p90*100,'%')} / p95={fmt(p95*100,'%')} / p99={fmt(p99*100,'%')}</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={logData} barCategoryGap="10%">
                        <XAxis dataKey="bin" tick={{ fontSize: 9 }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={v => Math.round(Math.expm1(v)).toLocaleString()} />
                        <Tooltip
                          formatter={(v, _, p) => [p.payload.count_raw.toLocaleString('fr-TN') + ' sinistres', 'Effectif']}
                          contentStyle={{ borderRadius: '10px', border: 'none', fontSize: 12 }}
                        />
                        <Bar dataKey="count_log" radius={[4, 4, 0, 0]}>
                          {logData.map((_, i) => <Cell key={i} fill={binColor(i)} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                );
              })()}

              {/* Model info */}
              <div className="bg-white rounded-2xl border border-slate-100 p-5">
                <SectionTitle icon={<Bot size={16} />}>Informations du Modèle</SectionTitle>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                  <div className="rounded-xl p-4 bg-slate-50 border border-slate-200 text-center">
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Sinistres Analysés</p>
                    <p className="text-2xl font-black text-slate-700">{fmt(fraudData.model_info?.nb_sinistres_analysed)}</p>
                  </div>
                  <div className="rounded-xl p-4 bg-slate-50 border border-slate-200 text-center">
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Features</p>
                    <p className="text-2xl font-black text-slate-700">{fraudData.model_info?.features_count ?? '—'}</p>
                  </div>
                  <div className="rounded-xl p-4 bg-rose-50 border border-rose-100 text-center">
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Seuil p99</p>
                    <p className="text-2xl font-black text-rose-700">{fmt(fraudData.thresholds?.p99 * 100, '%')}</p>
                    <p className="text-[10px] opacity-60 mt-1">Critique</p>
                  </div>
                  <div className="rounded-xl p-4 bg-amber-50 border border-amber-100 text-center">
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Seuil p95</p>
                    <p className="text-2xl font-black text-amber-700">{fmt(fraudData.thresholds?.p95 * 100, '%')}</p>
                    <p className="text-[10px] opacity-60 mt-1">Risque Élevé</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  {[['IF', 0.40, '#004A8D'], ['AE', 0.40, '#8b5cf6'], ['LOF', 0.20, '#10b981']].map(([name, w, color]) => (
                    <div key={name} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 bg-slate-50">
                      <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                      <span className="text-xs font-bold text-slate-600">{name}</span>
                      <span className="text-xs font-black" style={{ color }}>{(w * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                  <span className="text-xs text-slate-400 self-center ml-2">Poids de l'ensemble</span>
                </div>
              </div>
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
                {[
                  ['score_4',  null, '4/4 Modèles', 'rose'],
                  ['score_3',  null, '3/4 Modèles', 'orange'],
                  ['nb_anomalies', null, 'Total Suspects', 'blue'],
                  ['nb_contracts_analysed', null, 'Contrats', 'gray'],
                ].map(([key, icon, lbl, tone]) => {
                  const colors = { rose: ['bg-rose-50 border-rose-100', '#ef4444'], orange: ['bg-amber-50 border-amber-100', '#f59e0b'], blue: ['bg-sky-50 border-sky-100', '#004A8D'], gray: ['bg-slate-50 border-slate-200', '#64748b'] };
                  const [cls, clr] = colors[tone];
                  return (
                    <div key={key} className={`flex flex-col items-center rounded-xl p-3 min-w-[70px] border ${cls}`}>
                      <span className="text-xl font-black" style={{ color: clr }}>{anomalyData[key] ?? '—'}</span>
                      <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mt-0.5 text-center">{lbl}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Score filter chips */}
          {!loading && anomalyData && !anomalyData.error && (
            <div className="flex gap-2 flex-wrap mb-2">
              {[
                [1, 'Tous (≥1)',  anomalyData.score_1 + anomalyData.score_2 + anomalyData.score_3 + anomalyData.score_4, '#64748b'],
                [2, 'Score ≥ 2', anomalyData.score_2 + anomalyData.score_3 + anomalyData.score_4, '#f59e0b'],
                [3, 'Score ≥ 3', anomalyData.score_3 + anomalyData.score_4, '#f97316'],
                [4, 'Score = 4', anomalyData.score_4, '#ef4444'],
              ].map(([ms, label, count, color]) => (
                <button key={ms} onClick={() => { setScoreFilter(ms); loadAnomaly(ms); }}
                  className={`px-3 py-1.5 rounded-lg text-[11px] font-bold border transition-all ${scoreFilter === ms ? 'text-white border-transparent' : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'}`}
                  style={scoreFilter === ms ? { background: color } : {}}>
                  {label} <span className="opacity-70">({fmt(count ?? 0)})</span>
                </button>
              ))}
            </div>
          )}

          {loading ? <Spinner label="Analyse des anomalies..." /> :
           anomalyData?.error ? <ErrorBox msg={anomalyData.error} /> :
           (() => {
            const allAnomalies = anomalyData?.anomalies ?? [];
            const anomalySlice = allAnomalies.slice(
              (anomalyPage - 1) * FRAUD_ANOMALY_PAGE_SIZE,
              anomalyPage * FRAUD_ANOMALY_PAGE_SIZE
            );
            return (
              <>
                {anomalySlice.map((a, i) => {
                  const scoreColor = a.anomaly_score === 4 ? '#ef4444' : a.anomaly_score === 3 ? '#f97316' : a.anomaly_score === 2 ? '#f59e0b' : '#64748b';
                  return (
                    <div key={(anomalyPage - 1) * FRAUD_ANOMALY_PAGE_SIZE + i} className="flex bg-white border border-slate-200 rounded-2xl overflow-hidden hover:shadow-md transition-all duration-200 mb-3">
                      {/* Score badge */}
                      <div className="w-16 flex flex-col justify-center items-center border-r border-slate-100 flex-shrink-0" style={{ background: scoreColor + '15' }}>
                        <span className="text-2xl font-black" style={{ color: scoreColor }}>{a.anomaly_score}/4</span>
                        <span className="text-[8px] font-bold uppercase tracking-wider text-slate-400 mt-0.5 text-center">Consensus</span>
                      </div>
                      <div className="flex-1 p-4 min-w-0">
                        {/* Header row */}
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="font-bold text-slate-800 flex-shrink-0">Police #{a.id_police}</span>
                            <Badge label={a.branche || '—'} color="blue" />
                          </div>
                          <div className="flex gap-1 flex-shrink-0">
                            {a.if_anomaly     && <Badge label="IF"     color="red" />}
                            {a.lof_anomaly    && <Badge label="LOF"    color="red" />}
                            {a.ae_anomaly     && <Badge label="AE"     color="red" />}
                            {a.dbscan_anomaly && <Badge label="DBSCAN" color="red" />}
                          </div>
                        </div>
                        {/* Client row */}
                        <div className="flex items-center gap-2 mb-2 px-2 py-1.5 rounded-lg bg-slate-50 border border-slate-100">
                          <span className="text-slate-400 text-sm">{a.type_personne === 'M' ? '🏢' : '👤'}</span>
                          <span className="font-bold text-sm text-slate-700 truncate">{a.client_nom || '—'}</span>
                          {a.client_ville && a.client_ville !== '—' && (
                            <span className="text-[10px] text-slate-400 flex-shrink-0">· {a.client_ville}</span>
                          )}
                        </div>
                        {/* Metrics row */}
                        <div className="grid grid-cols-5 gap-2">
                          {[
                            ['Loss Ratio', a.loss_ratio > 1 ? <span className="text-rose-600 font-black">{fmt(a.loss_ratio * 100, '%')}</span> : fmt(a.loss_ratio * 100, '%')],
                            ['Tx Impayé',  fmt(a.taux_impaye * 100, '%')],
                            ['Sinistres',  fmt(a.nb_sinistres)],
                            ['Prime Nette', fmtCurrency(a.mt_pnet_total)],
                            ['IF Score',   fmt(a.if_score * 100, '%')],
                          ].map(([lbl, val]) => (
                            <div key={lbl} className="text-center bg-slate-50 rounded-lg p-1.5">
                              <p className="text-[9px] font-bold text-slate-400 uppercase">{lbl}</p>
                              <p className="text-xs font-bold text-slate-700">{val}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  );
                })}
                <Pagination
                  page={anomalyPage}
                  total={allAnomalies.length}
                  pageSize={FRAUD_ANOMALY_PAGE_SIZE}
                  onPage={setAnomalyPage}
                />
              </>
            );
           })()}
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
    fetchJson(`${API}/api/v1/ml/forecast?departement=${dept}&indicateur=${ind}&nb_mois=${nbMois}`)
      .then(d => { if (!cancelled) setData(d); })
      .catch(e => { if (!cancelled) setData({ error: e.message }); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [dept, ind, nbMois]);

  const historique = Array.isArray(data?.historique) ? data.historique : [];
  const previsions  = Array.isArray(data?.previsions)  ? data.previsions  : [];
  const methode     = data?.methode || 'XGBoost';
  const mapeColor   = data?.mape < 20 ? 'green' : data?.mape < 40 ? 'yellow' : 'red';

  // Merge into one array so XAxis covers both historical and forecast dates
  const histMap = Object.fromEntries(historique.map(d => [d.periode, d.valeur]));
  const prevMap = Object.fromEntries(previsions.map(d => [d.periode, { v: d.valeur, lo: d.valeur_min, hi: d.valeur_max }]));
  const allPeriodes = [
    ...historique.map(d => d.periode),
    ...previsions.map(d => d.periode).filter(p => !histMap[p]),
  ];
  const chartData = allPeriodes.map(p => ({
    periode:   p,
    historique: histMap[p] ?? null,
    prevision:  prevMap[p]?.v  ?? null,
    borne_haute: prevMap[p]?.hi ?? null,
    borne_basse: prevMap[p]?.lo ?? null,
  }));

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header banner */}
      <div className="bg-gradient-to-r from-[#004A8D]/6 to-transparent p-5 rounded-2xl border border-[#004A8D]/10">
        <div className="flex items-center gap-3 mb-1">
          <h4 className="font-bold text-[#004A8D] text-lg">Projections IA — KPIs Assurance</h4>
          <Badge label="XGBoost" color="blue" />
          <Badge label="Prophet" color="gray" />
        </div>
        <p className="text-sm text-slate-500">
          Meilleur modèle : <strong>XGBoost</strong>{data?.mape != null ? ` (MAPE ${fmt(data.mape, '%')})` : ''} pour Primes Acquises.
          Autres KPIs : Prophet (multi-KPI loop). Horizon max : 12 mois.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-wrap gap-4 items-end">
        {[
          ['Branche',    dept,   v => setDept(v),          [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']]],
          ['Indicateur', ind,    v => setInd(v),           Object.entries(indLabels)],
          ['Horizon',    nbMois, v => setNbMois(Number(v)), [[3,'+3 mois'],[6,'+6 mois'],[12,'+12 mois']]],
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
            {previsions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-10 gap-2">
                <Inbox size={40} className="opacity-25" />
                <p className="font-medium text-slate-600">Données insuffisantes pour cette branche</p>
                <p className="text-sm text-slate-400 text-center max-w-sm">
                  Cet indicateur n'a pas assez de données historiques pour <strong>{data.departement}</strong> afin de générer une prévision fiable.
                </p>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-end mb-5">
                  <div>
                    <h4 className="font-bold text-xl text-slate-800">{indLabels[data.indicateur] || data.indicateur}</h4>
                    <p className="text-sm text-slate-500">{data.departement} — horizon {nbMois} mois</p>
                  </div>
                  <div className="flex gap-2">
                    <Badge label={`⚡ ${methode}`} color="blue" />
                    {data.mape != null && <Badge label={`MAPE ${fmt(data.mape, '%')}`} color={mapeColor} />}
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="periode" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} dy={10}
                      interval={Math.max(0, Math.floor(chartData.length / 10) - 1)} />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={v => v.toLocaleString()} axisLine={false} tickLine={false} width={80} />
                    <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,.1)' }}
                      formatter={(v, name) => [v != null ? Number(v).toLocaleString() : '—', name]} />
                    <Legend iconType="circle" wrapperStyle={{ paddingTop: '16px' }} />
                    <Line dataKey="historique" name="Historique Réel" stroke={BRAND} strokeWidth={3} dot={false} isAnimationActive={false} connectNulls={false} />
                    <Line dataKey="prevision" name={`Prévision ${methode}`} stroke={ACCENT} strokeWidth={3} strokeDasharray="5 5" dot={{ r: 3, fill: ACCENT, strokeWidth: 0 }} connectNulls={false} />
                    {chartData.some(d => d.borne_haute != null) && <Line dataKey="borne_haute" name="Borne Haute (IC 90%)" stroke="#fdba74" strokeWidth={1} dot={false} strokeDasharray="3 3" connectNulls={false} />}
                    {chartData.some(d => d.borne_basse != null) && <Line dataKey="borne_basse" name="Borne Basse (IC 90%)" stroke="#fdba74" strokeWidth={1} dot={false} strokeDasharray="3 3" connectNulls={false} />}
                  </LineChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-5">
                  <StatCard label="Dernière valeur réelle" value={fmtCurrency(data.derniere_valeur)} tone="blue" />
                  <StatCard label="Prochaine prévision" value={fmtCurrency(data.prochaine_valeur)} tone="orange" />
                  <StatCard
                    label="MAPE modèle"
                    value={data.mape != null ? fmt(data.mape, '%') : 'N/D'}
                    sub={data.mape != null ? 'sur données test' : 'non calculé'}
                    tone={data.mape == null ? 'gray' : data.mape < 40 ? 'green' : 'red'}
                  />
                </div>
              </>
            )}
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

  const getColor = (seg) => {
    if (seg?.color) return seg.color;
    if (typeof seg === 'string') {
      if (seg.includes('VIP'))        return '#f1c40f';
      if (seg.includes('Fidèle'))     return '#2ecc71';
      if (seg.includes('Potentiel'))  return '#3498db';
      if (seg.includes('Risque'))     return '#e74c3c';
      if (seg.includes('Dormant'))    return '#e67e22';
      if (seg.includes('Entreprise')) return '#9b59b6';
    }
    return '#64748b';
  };

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
        <>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Pie + segment list */}
          <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm">
            <SectionTitle icon={<PieChartIcon size={16} />}>Répartition des Segments</SectionTitle>
            {data.segments && (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie
                      data={data.segments}
                      dataKey="count"
                      nameKey="name"
                      cx="50%" cy="50%"
                      innerRadius={52}
                      outerRadius={88}
                      stroke="white"
                      strokeWidth={3}
                      paddingAngle={2}
                      label={({ cx, cy, midAngle, innerRadius, outerRadius, share_pct }) => {
                        if (share_pct < 4) return null;
                        const RADIAN = Math.PI / 180;
                        const r = innerRadius + (outerRadius - innerRadius) * 0.5;
                        const x = cx + r * Math.cos(-midAngle * RADIAN);
                        const y = cy + r * Math.sin(-midAngle * RADIAN);
                        return (
                          <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central"
                            fontSize={11} fontWeight="700">
                            {`${share_pct}%`}
                          </text>
                        );
                      }}
                      labelLine={false}
                    >
                      {data.segments.map((s, i) => (
                        <Cell key={i} fill={getColor(s)} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(v, n) => [fmt(v) + ' clients', n]}
                      contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-1 mt-1">
                  {data.segments.map((s, i) => (
                    <button key={i} onClick={() => setSelected(s)}
                      className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-all ${selected?.name === s.name ? 'ring-1 font-bold' : 'hover:bg-slate-50'}`}
                      style={selected?.name === s.name ? { background: getColor(s) + '18', ringColor: getColor(s) } : {}}>
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: getColor(s) }} />
                        <span className="truncate text-slate-700">{s.name}</span>
                      </div>
                      <span className="text-xs font-bold ml-2 flex-shrink-0" style={{ color: getColor(s) }}>{s.share_pct}%</span>
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
                  <div className="w-3 h-3 rounded-full" style={{ background: getColor(selected) }} />
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
                      <Radar name={selected.name} dataKey="value" stroke={getColor(selected)} fill={getColor(selected)} fillOpacity={0.25} strokeWidth={2} />
                    </RadarChart>
                  </ResponsiveContainer>
                )}
                {selected.action && (
                  <div className="mt-3 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800 font-medium flex items-start gap-2">
                    <Lightbulb size={15} className="shrink-0 mt-0.5" />
                    <span><strong>Action recommandée :</strong> {selected.action}</span>
                  </div>
                )}
              </div>
            ) : <div className="flex items-center justify-center h-full text-slate-400">Sélectionnez un segment</div>}
          </div>
        </div>

        {/* Model metrics */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5">
          <SectionTitle icon={<Activity size={16} />}>Performance du Modèle K-Means</SectionTitle>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {/* Silhouette */}
            <div className={`rounded-xl p-4 border text-center ${data.silhouette != null && data.silhouette > 0.15 ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
              <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Silhouette</p>
              <p className={`text-2xl font-black ${data.silhouette != null && data.silhouette > 0.15 ? 'text-emerald-700' : 'text-amber-700'}`}>
                {data.silhouette != null ? data.silhouette.toFixed(3) : '—'}
              </p>
              <p className="text-[10px] opacity-60 mt-1">
                {data.silhouette != null && data.silhouette > 0.15 ? '✓ Bonne séparation' : 'Acceptable'}
              </p>
            </div>
            {/* Davies-Bouldin */}
            <div className={`rounded-xl p-4 border text-center ${data.davies_bouldin != null && data.davies_bouldin < 2.0 ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
              <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Davies-Bouldin</p>
              <p className={`text-2xl font-black ${data.davies_bouldin != null && data.davies_bouldin < 2.0 ? 'text-emerald-700' : 'text-amber-700'}`}>
                {data.davies_bouldin != null ? data.davies_bouldin.toFixed(3) : '—'}
              </p>
              <p className="text-[10px] opacity-60 mt-1">
                {data.davies_bouldin != null && data.davies_bouldin < 2.0 ? '✓ Clusters distincts' : 'À améliorer'}
              </p>
            </div>
            {/* Calinski-Harabasz */}
            <div className="rounded-xl p-4 border bg-sky-50 border-sky-200 text-center">
              <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Calinski-Harabasz</p>
              <p className="text-2xl font-black text-sky-700">
                {data.calinski_harabasz != null ? Math.round(data.calinski_harabasz).toLocaleString('fr-TN') : '—'}
              </p>
              <p className="text-[10px] opacity-60 mt-1">Plus élevé = meilleur</p>
            </div>
            {/* K */}
            <div className="rounded-xl p-4 border bg-violet-50 border-violet-200 text-center">
              <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Clusters K</p>
              <p className="text-2xl font-black text-violet-700">{data.k ?? data.nb_clusters ?? '—'}</p>
              <p className="text-[10px] opacity-60 mt-1">{fmt(data.nb_clients)} clients segmentés</p>
            </div>
            {/* Date entraînement */}
            <div className="rounded-xl p-4 border bg-slate-50 border-slate-200 text-center">
              <p className="text-[10px] font-black uppercase tracking-widest opacity-60 mb-1">Entraîné le</p>
              <p className="text-base font-black text-slate-700">{data.ref_date ?? '—'}</p>
              <p className="text-[10px] opacity-60 mt-1">{data.model_source ?? 'artifact'}</p>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-3">
            Silhouette : [-1, 1], plus élevé = meilleur | Davies-Bouldin : plus bas = meilleur | Calinski-Harabasz : plus élevé = meilleur
          </p>
        </div>
        </>
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
    age_client: 40, taux_impaye: 0.0, mt_pnet: 1200, valeur_vehicule: 12000,
    nature_sinistre: 'MATERIEL', delai_declaration: 9, prime_contrat: 500,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tableData, setTableData] = useState(null);
  const [riskInfo, setRiskInfo] = useState(null);
  const [claimInfo, setClaimInfo] = useState(null);

  useEffect(() => {
    fetchJson(`${API}/api/v1/ml/risk/table`).then(d => setTableData(d)).catch(() => {});
    fetchJson(`${API}/api/v1/ml/risk/model-info`).then(d => setRiskInfo(d)).catch(() => {});
    fetchJson(`${API}/api/v1/ml/claim/model-info`).then(d => setClaimInfo(d)).catch(() => {});
  }, []);

  const submit = async () => {
    setLoading(true); setResult(null);
    const endpoint = view === 'risk' ? '/api/v1/ml/risk/score' : '/api/v1/ml/claim/predict';
    const payload = view === 'risk'
      ? { branche: form.branche, bonus_malus: Number(form.bonus_malus), puissance: Number(form.puissance), age_vehicule: Number(form.age_vehicule), age_client: Number(form.age_client), taux_impaye: Number(form.taux_impaye), mt_pnet: Number(form.mt_pnet), valeur_vehicule: Number(form.valeur_vehicule) }
      : { branche: form.branche, nature_sinistre: form.nature_sinistre, valeur_vehicule: Number(form.valeur_vehicule), age_client: Number(form.age_client), bonus_malus: Number(form.bonus_malus), delai_declaration: Number(form.delai_declaration), prime_contrat: Number(form.prime_contrat) };
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
        {[['risk', 'Risk Scoring', 'Score 0–1000 + Prime Technique'], ['claim', 'Claim Severity', 'Prédiction Coût Sinistre']].map(([id, label, sub]) => (
          <button key={id} onClick={() => { setView(id); setResult(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${view === id ? 'bg-white shadow text-[#004A8D]' : 'text-slate-500 hover:text-slate-700'}`}>

            {label} <span className="text-[9px] opacity-50 font-normal ml-1">{sub}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
          <SectionTitle icon={view === 'risk' ? <Target size={16} /> : <DollarSign size={16} />}>{view === 'risk' ? 'Simulateur de Risque' : 'Simulateur Sinistre'}</SectionTitle>
          <div className="grid grid-cols-2 gap-3">
            {(
              view === 'risk' ? [
                // [key, label, type, opts, autoOnly]
                ['branche', 'Branche', 'select', [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']], false],
                ['bonus_malus', 'Bonus/Malus', 'number', null, false],
                ['puissance', 'Puissance (CV)', 'number', null, true],
                ['age_vehicule', 'Âge Véhicule (ans)', 'number', null, true],
                ['age_client', 'Âge Client', 'number', null, false],
                ['taux_impaye', 'Taux Impayés (0–1)', 'number', null, false],
                ['mt_pnet', 'Prime Nette (TND)', 'number', null, false],
                ['valeur_vehicule', 'Valeur Véhicule (TND)', 'number', null, true],
              ] : [
                ['branche', 'Branche', 'select', [['AUTO','Automobile'],['IRDS','IRDS'],['SANTE','Santé']], false],
                ['nature_sinistre', 'Nature', 'select', form.branche === 'AUTO'
                  ? [['MATERIEL','Matériel'],['CORPOREL','Corporel'],['MIXTE','Mixte']]
                  : form.branche === 'IRDS'
                  ? [['INCENDIE','Incendie'],['MATERIEL','Dommages mat.'],['VOL','Vol']]
                  : [['MATERIEL','Soins'],['CORPOREL','Hospitalisation']], false],
                ['valeur_vehicule', 'Valeur Véhicule (TND)', 'number', null, true],
                ['age_client', 'Âge Client', 'number', null, false],
                ['bonus_malus', 'Bonus/Malus', 'number', null, false],
                ['delai_declaration', 'Délai Déclaration (j.)', 'number', null, false],
                ['prime_contrat', 'Prime Contrat (TND)', 'number', null, false],
              ]
            ).filter(([, , , , autoOnly]) => !autoOnly || form.branche === 'AUTO')
             .map(([key, label, type, opts]) => (
              <div key={key}>
                <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1 block">{label}</label>
                {opts ? (
                  <select value={form[key]} onChange={e => {
                    const val = e.target.value;
                    if (key === 'branche') {
                      const defaultNature = val === 'IRDS' ? 'INCENDIE' : val === 'SANTE' ? 'MATERIEL' : 'MATERIEL';
                      setForm({ ...form, branche: val, nature_sinistre: defaultNature });
                    } else {
                      setForm({ ...form, [key]: val });
                    }
                  }}
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
            className="w-full mt-5 py-3 inline-flex items-center justify-center gap-2 bg-gradient-to-br from-[#004A8D] to-blue-800 text-white rounded-xl font-bold shadow-md shadow-[#004A8D]/20 hover:shadow-lg hover:shadow-[#004A8D]/30 disabled:opacity-50 disabled:shadow-none transition-all">
            {loading
              ? <><Loader2 size={15} className="animate-spin" /> Calcul...</>
              : view === 'risk'
                ? <><Zap size={15} /> Calculer Score de Risque</>
                : <><Wand2 size={15} /> Prédire Coût Sinistre</>}
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
              {view === 'risk' ? <Target size={48} className="opacity-20" /> : <DollarSign size={48} className="opacity-20" />}
              <p className="text-sm font-medium">Remplissez le formulaire pour calculer</p>
            </div>
          )}
        </div>
      </div>

      {/* Model metrics */}
      {(() => {
        const info = view === 'risk' ? riskInfo : claimInfo;
        if (!info || info.status !== 'ready') return null;
        const fmtPct  = v => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` : '—';
        const fmtVal  = v => v != null ? v.toFixed(0) : '—';
        const fmtDec  = v => v != null ? v.toFixed(3) : '—';

        const MetBox = ({ label, value, sub, color }) => (
          <div className="bg-white rounded-xl p-3 border border-slate-100 text-center">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{label}</p>
            <p className={`text-lg font-black ${color || 'text-slate-800'}`}>{value}</p>
            {sub && <p className="text-[10px] text-slate-400 mt-0.5">{sub}</p>}
          </div>
        );

        if (view === 'risk') {
          const mf = info.metrics_freq || {};
          const ms = info.metrics_sev  || {};
          return (
            <div className="bg-slate-50 rounded-2xl border border-slate-100 p-5">
              <SectionTitle icon={<TrendingUp size={16} />}>Performance du Modèle — {info.source}</SectionTitle>
              <p className="text-xs text-slate-400 -mt-2 mb-4">
                {info.notebook} · {info.features} features · Chargement ×{info.chargement}
                &nbsp;|&nbsp; Fréq : XGB {Math.round((info.weights?.freq?.xgb || 0)*100)}% / LGB {Math.round((info.weights?.freq?.lgb || 0)*100)}%
                &nbsp;·&nbsp; Sév : XGB {Math.round((info.weights?.sev?.xgb || 0)*100)}% / LGB {Math.round((info.weights?.sev?.lgb || 0)*100)}%
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">Modèle Fréquence</p>
                  <div className="grid grid-cols-3 gap-2">
                    <MetBox label="MAE" value={fmtDec(mf.MAE)} sub="sin/an" />
                    <MetBox label="Gini" value={fmtDec(mf.Gini)} color={mf.Gini > 0.3 ? 'text-emerald-600' : 'text-amber-600'} />
                    <MetBox label="Biais" value={fmtPct(mf['Biais%'])} color={Math.abs(mf['Biais%'] || 0) < 5 ? 'text-emerald-600' : 'text-amber-600'} />
                  </div>
                </div>
                <div>
                  <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">Modèle Sévérité</p>
                  <div className="grid grid-cols-3 gap-2">
                    <MetBox label="MAE" value={fmtVal(ms.MAE)} sub="DT" />
                    <MetBox label="Gini" value={fmtDec(ms.Gini)} color={ms.Gini > 0.3 ? 'text-emerald-600' : 'text-amber-600'} />
                    <MetBox label="Biais" value={fmtPct(ms['Biais%'])} color={Math.abs(ms['Biais%'] || 0) < 20 ? 'text-emerald-600' : 'text-rose-600'} />
                  </div>
                </div>
              </div>
            </div>
          );
        } else {
          const m = info.metrics || {};
          return (
            <div className="bg-slate-50 rounded-2xl border border-slate-100 p-5">
              <SectionTitle icon={<TrendingUp size={16} />}>Performance du Modèle — {info.source}</SectionTitle>
              <p className="text-xs text-slate-400 -mt-2 mb-4">
                {info.notebook} · {info.features} features
                &nbsp;|&nbsp; XGB {Math.round((info.weights?.xgb || 0)*100)}% / LGB {Math.round((info.weights?.lgb || 0)*100)}%
                &nbsp;·&nbsp; Couverture IC : {((info.quantile_coverage || 0)*100).toFixed(1)}%
                &nbsp;·&nbsp; Réserve factor : ×{info.reserve_factor}
              </p>
              <div className="grid grid-cols-5 gap-2">
                <MetBox label="MAE" value={fmtVal(m.MAE)} sub="DT" />
                <MetBox label="RMSE" value={fmtVal(m.RMSE)} sub="DT" />
                <MetBox label="MAPE" value={m.MAPE != null ? `${m.MAPE.toFixed(1)}%` : '—'} color={m.MAPE < 50 ? 'text-emerald-600' : 'text-rose-600'} />
                <MetBox label="Gini" value={fmtDec(m.Gini)} color={m.Gini > 0.3 ? 'text-emerald-600' : 'text-amber-600'} />
                <MetBox label="Biais" value={fmtPct(m['Biais%'])} color={Math.abs(m['Biais%'] || 0) < 20 ? 'text-emerald-600' : 'text-rose-600'} />
              </div>
            </div>
          );
        }
      })()}

      {/* Risk table */}
      {view === 'risk' && tableData?.table && (
        <div className="bg-white rounded-2xl border border-slate-100 p-5">
          <SectionTitle icon={<BarChart2 size={16} />}>Statistiques du Portefeuille par Branche</SectionTitle>
          <p className="text-xs text-slate-400 mb-4 -mt-2">Données historiques agrégées — fréquence et sévérité réelles du portefeuille</p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-100">
                  {['Branche', 'Nb Polices', 'Fréq. Sinistres', 'Sévérité Moy.', 'Ratio S/P', 'Prime Technique'].map(h => (
                    <th key={h} className="text-left py-2 px-3 font-bold text-slate-400 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableData.table.map((row, i) => (
                  <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                    <td className="py-2 px-3">
                      <span className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        row.branche === 'AUTO' ? 'bg-blue-50 text-blue-700' :
                        row.branche === 'IRDS' ? 'bg-amber-50 text-amber-700' :
                        'bg-emerald-50 text-emerald-700'
                      }`}>{row.branche}</span>
                    </td>
                    <td className="py-2 px-3 text-slate-600">{fmt(row.nb_polices)}</td>
                    <td className="py-2 px-3 text-slate-600">{fmt(row.freq_sin_moy * 100, '%')}</td>
                    <td className="py-2 px-3 text-slate-600">{fmtCurrency(row.sev_moy)}</td>
                    <td className={`py-2 px-3 font-bold ${row.sp_ratio_moy > 1 ? 'text-rose-600' : 'text-emerald-600'}`}>
                      {fmt(row.sp_ratio_moy * 100, '%')}
                      {row.sp_ratio_moy > 1 && <span className="ml-1 text-[9px]">⚠</span>}
                    </td>
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
  const psiLabel = (v) => v >= 0.2 ? 'Critique' : v >= 0.1 ? 'Modéré' : 'Stable';

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
        <div className="ml-auto text-right">
          <div className="flex items-center gap-2 justify-end mb-1">
            <Badge label="Evidently AI" color="blue" />
            <Badge label="KS · Chi² · PSI" color="gray" />
          </div>
          {data && !data.error && (
            <p className="text-[10px] text-slate-400">
              Réf. <span className="font-bold text-slate-600">{data.date_ref_debut} → {data.date_ref_fin}</span>
              <span className="mx-1.5">·</span>
              Courant <span className="font-bold text-slate-600">{data.date_cur_debut} → {data.date_max}</span>
              <span className="mx-1.5">·</span>
              {fmt(data.nb_ref)} / {fmt(data.nb_courant)} quittances
            </p>
          )}
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
              <SectionTitle icon={<BarChart2 size={16} />}>PSI par Feature (Population Stability Index)</SectionTitle>
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
                  {comp ? (
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
                  ) : (
                    /* Categorical feature — no mean comparison, show PSI + test score */
                    <div className="space-y-3 mt-1">
                      <div className="flex items-center justify-between bg-slate-50 rounded-lg px-3 py-2">
                        <span className="text-[10px] font-bold text-slate-400 uppercase">Méthode</span>
                        <span className="text-xs font-semibold text-slate-600">{f.methode || 'Jensen-Shannon'}</span>
                      </div>
                      <div className="flex items-center justify-between bg-slate-50 rounded-lg px-3 py-2">
                        <span className="text-[10px] font-bold text-slate-400 uppercase">Score</span>
                        <span className={`text-sm font-bold ${f.p_value < 0.05 ? 'text-rose-600' : 'text-emerald-600'}`}>
                          {typeof f.p_value === 'number' ? f.p_value.toFixed(4) : '—'}
                        </span>
                      </div>
                      {psiEntry && (
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span className="font-bold uppercase">PSI</span>
                            <span style={{ color: psiColor(psiEntry.psi) }} className="font-bold">{psiEntry.psi.toFixed(4)}</span>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-2">
                            <div className="h-2 rounded-full transition-all" style={{ width: `${Math.min(100, (psiEntry.psi / 0.3) * 100)}%`, background: psiColor(psiEntry.psi) }} />
                          </div>
                          <div className="flex justify-between text-[9px] text-slate-300 mt-0.5">
                            <span>0</span><span>0.10</span><span>0.20</span><span>0.30+</span>
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
  { id: 'impaye',       label: 'Risque Impayé',     Icon: CreditCard,   notebook: 'impaye_risk_scoring' },
  { id: 'churn',        label: 'Prédiction Churn',   Icon: TrendingDown, notebook: 'churn_prediction_v3' },
  { id: 'fraud',        label: 'Fraude & Anomalies', Icon: ShieldAlert,  notebook: 'fraud_detection + anomaly_detection' },
  { id: 'forecast',     label: 'Projections IA',     Icon: TrendingUp,   notebook: 'forecast_model' },
  { id: 'segmentation', label: 'Segmentation',       Icon: Users,        notebook: 'customer_segmentation' },
  { id: 'risk',         label: 'Risque & Pricing',   Icon: Target,       notebook: 'risk_scoring_pricing + claim_severity' },
  { id: 'drift',        label: 'Data Drift',         Icon: Scale,        notebook: 'data_drift_evidently' },
];

export default function MLOpsContent() {
  const [tab, setTab] = useState('impaye');

  return (
    <div className="mlops-root bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden flex flex-col md:flex-row min-h-[750px] m-4 lg:m-8">
      {/* Sidebar */}
      <div className="mlops-sidebar w-full md:w-64 bg-slate-50/80 border-r border-slate-100 flex flex-col p-4 md:p-6 space-y-1.5 shrink-0">
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
            <t.Icon size={17} className="shrink-0" />
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
          {[['Impayé', 'LightGBM'], ['Churn', 'XGBoost v3'], ['Fraude', 'IF+AE+LOF'], ['Forecast', 'Prophet+ML'], ['Segments', 'K-Means'], ['Pricing', 'GBM+GLM'], ['Drift', 'Evidently']].map(([name, algo]) => (
            <div key={name} className="flex items-center justify-between py-1 px-1">
              <span className="text-[10px] text-slate-500 font-medium">{name}</span>
              <span className="text-[9px] text-slate-400 bg-slate-100 rounded px-1.5 py-0.5">{algo}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="mlops-main flex-1 p-6 md:p-8 overflow-y-auto bg-[#fcfdfd]">
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
