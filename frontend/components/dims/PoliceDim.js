import {
  Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";
import { FileText, CheckCircle2, XCircle, User, Layers, Target } from 'lucide-react';

const fmt = new Intl.NumberFormat("fr-TN");

const RADIAN = Math.PI / 180;
function renderPieLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }) {
  if ((percent || 0) < 0.01) return null;
  if ((percent || 0) >= 0.07) {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    return (
      <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600}>
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  }
  const sin = Math.sin(-midAngle * RADIAN);
  const cos = Math.cos(-midAngle * RADIAN);
  const x1 = cx + (outerRadius + 8) * cos;
  const y1 = cy + (outerRadius + 8) * sin;
  const x2 = cx + (outerRadius + 24) * cos;
  const y2 = cy + (outerRadius + 24) * sin;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#94a3b8" strokeWidth={1} />
      <text x={x2 + (cos >= 0 ? 3 : -3)} y={y2} fill="#64748b"
        textAnchor={cos >= 0 ? "start" : "end"} dominantBaseline="central" fontSize={10} fontWeight={600}>
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    </g>
  );
}

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

  const bonusMalusFiltered = bonusMalus.filter((e) => (e.count || 0) > 0);

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: <FileText size={18} />,     title: "Total polices",       value: fmt.format(kpis.total),           sub: "portefeuille",              current: kpis.total,         previous: prev.total },
        { icon: <CheckCircle2 size={18} />,title: "En vigueur",           value: fmt.format(kpis.en_vigueur),      sub: `${Number(kpis.pct_vigueur || 0).toFixed(1)} %`,   current: kpis.en_vigueur,    previous: prev.en_vigueur },
        { icon: <XCircle size={18} />,     title: "Résiliées",            value: fmt.format(kpis.resiliees),       sub: `${Number(kpis.pct_resiliees || 0).toFixed(1)} %`, current: kpis.resiliees,     previous: prev.resiliees,     invertColor: true },
        { icon: <User size={18} />,        title: "Polices individuelles",value: fmt.format(kpis.individuelles),   sub: `${Number(kpis.pct_indiv || 0).toFixed(1)} %`,     current: kpis.individuelles, previous: prev.individuelles },
        { icon: <Layers size={18} />,      title: "Polices flotte",       value: fmt.format(kpis.flottes),         sub: `${Number(kpis.pct_flotte || 0).toFixed(1)} %`,    current: kpis.flottes,       previous: prev.flottes },
        { icon: <Target size={18} />,      title: "BM moyen",             value: Number(kpis.avg_bm || 0).toFixed(2),           sub: "bonus-malus moyen",         current: kpis.avg_bm,        previous: prev.avg_bm,        invertColor: true },
      ]} />

      <div className="dim-charts-grid">
        {/* Situation portefeuille */}
        <article className="panel chart-panel">
          <h3>Situation du portefeuille</h3>
          <ResponsiveContainer width="100%" height={290}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={situationChart} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {bySituation.map((entry) => (
                  <Cell key={entry.label} fill={SITUATION_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Type de police */}
        <article className="panel chart-panel">
          <h3>Type de police</h3>
          <ResponsiveContainer width="100%" height={270}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={byType} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                <Cell fill="#004A8D" />
                <Cell fill="#F38F1D" />
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Périodicité */}
        <article className="panel chart-panel">
          <h3>Périodicité de règlement</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={periodiciteChart} margin={{ top: 20, bottom: 10 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10 }}
                angle={-25}
                textAnchor="end"
                height={58}
                interval={0}
              />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Polices']} />
              <Bar dataKey="count" name="Polices" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11 }}>
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
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={byDuree} margin={{ top: 20 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Polices']} />
              <Bar dataKey="count" name="Polices" fill="#1B68B2" radius={[6, 6, 0, 0]}
                label={{ position: "top", fontSize: 11 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Bonus-Malus */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Distribution Bonus-Malus</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={bonusMalusFiltered} margin={{ top: 20 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Polices']} />
              <Bar dataKey="count" name="Polices" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11 }}>
                {bonusMalusFiltered.map((entry, i) => (
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