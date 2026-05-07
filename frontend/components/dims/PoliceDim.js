import {
  Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";

const fmt = new Intl.NumberFormat("fr-TN");

const SITUATION_COLORS = {
  V: "#2E7D32", R: "#C62828", T: "#F38F1D", S: "#6A1B9A", A: "#94a3b8",
};
const SITUATION_LABELS = {
  V: "En vigueur", R: "Résiliée", T: "Terminée", S: "Suspendue", A: "Annulée",
};
const PERIODICITE_COLORS = {
  A: "#004A8D", S: "#F38F1D", T: "#2E7D32", C: "#C62828",
};
const PERIODICITE_LABELS = {
  A: "Annuelle", S: "Semestrielle", T: "Trimestrielle", C: "Comptant",
};
const BM_COLORS = ["#2E7D32", "#004A8D", "#F38F1D", "#C62828"];

export default function PoliceDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement polices…</p>;

  const { kpis = {}, byType = [], bySituation = [], byPeriodicite = [], byDuree = [], bonusMalus = [] } = data;
  const prev = dataPrev?.kpis || {};

  const situationChart = bySituation.map((e) => ({
    ...e,
    label: SITUATION_LABELS[e.label] || e.label,
    _orig: e.label,
  }));
  const periodiciteChart = byPeriodicite.map((e) => ({
    ...e,
    label: PERIODICITE_LABELS[e.label] || e.label,
    _orig: e.label,
  }));

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "📋", title: "Total polices",       value: fmt.format(kpis.total),           sub: "portefeuille",              current: kpis.total,         previous: prev.total },
        { icon: "✅", title: "En vigueur",           value: fmt.format(kpis.en_vigueur),      sub: `${Number(kpis.pct_vigueur || 0).toFixed(1)} %`,   current: kpis.en_vigueur,    previous: prev.en_vigueur },
        { icon: "🔴", title: "Résiliées",            value: fmt.format(kpis.resiliees),       sub: `${Number(kpis.pct_resiliees || 0).toFixed(1)} %`, current: kpis.resiliees,     previous: prev.resiliees,     invertColor: true },
        { icon: "👤", title: "Polices individuelles",value: fmt.format(kpis.individuelles),   sub: `${Number(kpis.pct_indiv || 0).toFixed(1)} %`,     current: kpis.individuelles, previous: prev.individuelles },
        { icon: "🚌", title: "Polices flotte",       value: fmt.format(kpis.flottes),         sub: `${Number(kpis.pct_flotte || 0).toFixed(1)} %`,    current: kpis.flottes,       previous: prev.flottes },
        { icon: "🎯", title: "BM moyen",             value: Number(kpis.avg_bm || 0).toFixed(2),           sub: "bonus-malus moyen",         current: kpis.avg_bm,        previous: prev.avg_bm,        invertColor: true },
      ]} />

      <div className="dim-charts-grid">
        {/* Situation portefeuille */}
        <article className="panel chart-panel">
          <h3>Situation du portefeuille</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={situationChart} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
              >
                {bySituation.map((entry) => (
                  <Cell key={entry.label} fill={SITUATION_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Type de police */}
        <article className="panel chart-panel">
          <h3>Type de police</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={byType} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ percent }) => `${(percent * 100).toFixed(1)}%`}
              >
                <Cell fill="#004A8D" />
                <Cell fill="#F38F1D" />
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Périodicité */}
        <article className="panel chart-panel">
          <h3>Périodicité de règlement</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={periodiciteChart}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {byPeriodicite.map((entry) => (
                  <Cell key={entry.label} fill={PERIODICITE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Durée de police */}
        <article className="panel chart-panel">
          <h3>Durée de police</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byDuree}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" fill="#1B68B2" radius={[6, 6, 0, 0]}
                label={{ position: "top", fontSize: 11 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Bonus-Malus */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Distribution Bonus-Malus</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={bonusMalus}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {bonusMalus.map((entry, i) => (
                  <Cell key={entry.label} fill={BM_COLORS[i % BM_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>
      </div>
    </div>
  );
}