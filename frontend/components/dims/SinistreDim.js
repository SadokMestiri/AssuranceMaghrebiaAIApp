import {
  Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";

const fmt = new Intl.NumberFormat("fr-TN");
const fmtTND = new Intl.NumberFormat("fr-TN", { style: "currency", currency: "TND", maximumFractionDigits: 0 });

function formatShortCurrency(value) {
  const num = Number(value || 0);
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M DT`;
  } else if (num >= 1_000) {
    return `${Math.round(num / 1_000)}K DT`;
  }
  return `${num} DT`;
}

const ETAT_COLORS  = { Clos: "#2E7D32", Ouvert: "#C62828", "Refusé": "#F38F1D" };
const RESP_COLORS  = { "0": "#2E7D32", "50": "#F38F1D", "100": "#C62828" };
const RESP_LABELS  = { "0": "0 % (Tiers resp.)", "50": "50 % (Partage)", "100": "100 % (Assuré)" };
const NATURE_COLORS = [
  "#C62828","#004A8D","#F38F1D","#2E7D32",
  "#6A1B9A","#00838F","#5D4037","#3949AB","#E91E8C","#00ACC1",
];

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
        { icon: "⚠️", title: "Total sinistres",    value: fmt.format(kpis.total),                sub: "toutes périodes",                      current: kpis.total,       previous: prev.total,       invertColor: true },
        { icon: "🔴", title: "Ouverts",             value: fmt.format(kpis.ouverts),              sub: `${Number(kpis.pct_ouverts || 0).toFixed(1)} %`,     current: kpis.ouverts,     previous: prev.ouverts,     invertColor: true },
        { icon: "✅", title: "Clos",                value: fmt.format(kpis.clos),                 sub: `${Number(kpis.pct_clos || 0).toFixed(1)} %`,        current: kpis.clos,        previous: prev.clos },
        { icon: "💰", title: "Montant évalué",      value: formatShortCurrency(kpis.total_eval),  sub: "provisions ouvertes",                  current: kpis.total_eval,  previous: prev.total_eval,  invertColor: true },
        { icon: "💸", title: "Montant payé",        value: formatShortCurrency(kpis.total_paye),  sub: `${Number(kpis.taux_paiement || 0).toFixed(1)} % du provisonné`, current: kpis.total_paye, previous: prev.total_paye, invertColor: true },
        { icon: "🧾", title: "Sinistres matériels", value: fmt.format(kpis.nb_materiel),          sub: "1er sinistre par nature",              current: kpis.nb_materiel, previous: prev.nb_materiel, invertColor: true },
      ]} />

      <div className="dim-charts-grid">
        {/* Nature sinistre */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Sinistres par nature</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={byNature} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={130} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" name="Sinistres">
                {byNature.map((entry, i) => (
                  <Cell key={entry.label} fill={NATURE_COLORS[i % NATURE_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* État sinistre */}
        <article className="panel chart-panel">
          <h3>État des sinistres</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={byEtat} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
              >
                {byEtat.map((entry) => (
                  <Cell key={entry.label} fill={ETAT_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Responsabilité */}
        <article className="panel chart-panel">
          <h3>Responsabilité engagée</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={respChart} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
              >
                {byResponsabilite.map((entry) => (
                  <Cell key={entry.label} fill={RESP_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Par branche */}
        <article className="panel chart-panel">
          <h3>Sinistres par branche</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byBranche}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]} fill="#C62828"
                label={{ position: "top", fontSize: 11 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Trend mensuel */}
        {monthly && monthly.length > 0 && (
          <article className="panel chart-panel dim-chart-wide">
            <h3>Évolution mensuelle sinistres</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={monthly}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => fmt.format(v)} />
                <Bar dataKey="count" fill="#C62828" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </article>
        )}
      </div>
    </div>
  );
}