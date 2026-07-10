import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";
import { formatShortCurrency } from "../shared";
import { AlertTriangle, CircleDot, CheckCircle2, Banknote, Wallet, ScrollText } from 'lucide-react';

const fmt = new Intl.NumberFormat("fr-TN");

// Status semantics: open = pending (amber), closed = resolved (green),
// refused = negative outcome (red) — a status color, not a series identity.
const ETAT_COLORS  = { Ouvert: "#F38F1D", Clos: "#2E7D32", "Refusé": "#C62828" };
const RESP_COLORS  = { "0": "#2E7D32", "50": "#F38F1D", "100": "#C62828" };
const RESP_LABELS  = { "0": "Tiers resp.", "50": "Partage", "100": "Assuré" };

// Same fixed branch→hue mapping used app-wide (ChartsPanel.js, ProduitDim.js)
// so AUTO/IRDS/SANTE always carry the same identity color everywhere.
const BRANCHE_COLORS = { AUTO: "#004A8D", IRDS: "#F38F1D", SANTE: "#2E7D32" };
const NATURE_COLOR = "#004A8D";

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

export default function SinistreDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement sinistres…</p>;

  const { kpis = {}, byNature = [], byEtat = [], byResponsabilite = [], byBranche = [], monthly = [] } = data;
  const prev = dataPrev?.kpis || {};

  const respChart = byResponsabilite.map((e) => ({
    ...e, label: RESP_LABELS[e.label] || e.label,
  }));

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: <AlertTriangle size={18} />, title: "Total sinistres",    value: fmt.format(kpis.total),                sub: "toutes périodes",                      current: kpis.total,       previous: prev.total,       invertColor: true },
        { icon: <CircleDot size={18} />,     title: "Ouverts",             value: fmt.format(kpis.ouverts),              sub: `${Number(kpis.pct_ouverts || 0).toFixed(1)} %`,     current: kpis.ouverts,     previous: prev.ouverts,     invertColor: true },
        { icon: <CheckCircle2 size={18} />,  title: "Clos",                value: fmt.format(kpis.clos),                 sub: `${Number(kpis.pct_clos || 0).toFixed(1)} %`,        current: kpis.clos,        previous: prev.clos },
        { icon: <Banknote size={18} />,      title: "Montant évalué",      value: formatShortCurrency(kpis.total_eval),  sub: "provisions ouvertes",                  current: kpis.total_eval,  previous: prev.total_eval,  invertColor: true },
        { icon: <Wallet size={18} />,        title: "Montant payé",        value: formatShortCurrency(kpis.total_paye),  sub: `${Number(kpis.taux_paiement || 0).toFixed(1)} % du provisonné`, current: kpis.total_paye, previous: prev.total_paye, invertColor: true },
        { icon: <ScrollText size={18} />,    title: "Sinistres matériels", value: fmt.format(kpis.nb_materiel),          sub: "1er sinistre par nature",              current: kpis.nb_materiel, previous: prev.nb_materiel, invertColor: true },
      ]} />

      <div className="dim-charts-grid">
        {/* Nature sinistre */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Sinistres par nature</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={byNature} layout="vertical" margin={{ right: 45 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={130} interval={0} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Bar dataKey="count" name="Sinistres" fill={NATURE_COLOR} radius={[0, 6, 6, 0]}
                label={{ position: "right", fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* État sinistre */}
        <article className="panel chart-panel">
          <h3>État des sinistres</h3>
          <ResponsiveContainer width="100%" height={270}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={byEtat} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {byEtat.map((entry) => (
                  <Cell key={entry.label} fill={ETAT_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Responsabilité */}
        <article className="panel chart-panel">
          <h3>Responsabilité engagée</h3>
          <ResponsiveContainer width="100%" height={270}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={respChart} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {byResponsabilite.map((entry) => (
                  <Cell key={entry.label} fill={RESP_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Par branche */}
        <article className="panel chart-panel">
          <h3>Sinistres par branche</h3>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={byBranche} margin={{ top: 20 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Sinistres']} />
              <Bar dataKey="count" name="Sinistres" radius={[6, 6, 0, 0]}
                label={{ position: "top", fontSize: 11 }}
              >
                {byBranche.map((entry) => (
                  <Cell key={entry.label} fill={BRANCHE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Trend mensuel */}
        {monthly && monthly.length > 0 && (
          <article className="panel chart-panel dim-chart-wide">
            <h3>Évolution mensuelle sinistres</h3>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={monthly}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [fmt.format(v), 'Sinistres']} />
                <Line type="monotone" dataKey="count" name="Sinistres" stroke="#004A8D" strokeWidth={2}
                  dot={{ r: 3, fill: "#004A8D" }} activeDot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </article>
        )}
      </div>
    </div>
  );
}