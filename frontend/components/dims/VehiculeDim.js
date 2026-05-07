import {
  Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";

const fmt = new Intl.NumberFormat("fr-TN");

const GENRE_COLORS = {
  VP: "#004A8D", VU: "#F38F1D", TC: "#2E7D32",
  AR: "#C62828", PL: "#6A1B9A",
};
const PUISS_COLORS = ["#004A8D", "#1B68B2", "#F38F1D", "#C62828"];
const MARQUE_COLORS = [
  "#004A8D","#F38F1D","#2E7D32","#C62828",
  "#6A1B9A","#00838F","#5D4037","#3949AB","#E91E8C","#00ACC1",
];

export default function VehiculeDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement véhicules…</p>;

  const { kpis = {}, byGenre = [], byPuissance = [], topMarques = [], byType = [] } = data;
  const prev = dataPrev?.kpis || {};

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "🚗", title: "Total véhicules",    value: fmt.format(kpis.total),              sub: "assurés",            current: kpis.total,          previous: prev.total },
        { icon: "🏎", title: "Voitures part.",     value: fmt.format(kpis.nb_vp),              sub: `${Number(kpis.pct_vp || 0).toFixed(1)} % du parc`, current: kpis.nb_vp, previous: prev.nb_vp },
        { icon: "🚚", title: "Véh. utilitaires",  value: fmt.format(kpis.nb_vu),              sub: "VU + PL + TC",       current: kpis.nb_vu,          previous: prev.nb_vu },
        { icon: "🔧", title: "Marques distinctes", value: fmt.format(kpis.nb_marques),         sub: "constructeurs",      current: kpis.nb_marques,     previous: prev.nb_marques },
        { icon: "⚡", title: "Puissance moy.",     value: kpis.avg_puissance + " CV",          sub: "moyenne parc",       current: kpis.avg_puissance,  previous: prev.avg_puissance },
        { icon: "📅", title: "Ancienneté moy.",    value: kpis.avg_age + " ans",               sub: "âge moyen véhicule", current: kpis.avg_age,        previous: prev.avg_age,       invertColor: true },
      ]} />

      <div className="dim-charts-grid">
        {/* Top marques */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 marques — parc assuré</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topMarques}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" name="Véhicules">
                {topMarques.map((entry, i) => (
                  <Cell key={entry.label} fill={MARQUE_COLORS[i % MARQUE_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Genre véhicule */}
        <article className="panel chart-panel">
          <h3>Genre de véhicule</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={byGenre} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {byGenre.map((entry) => (
                  <Cell key={entry.label} fill={GENRE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Tranche de puissance */}
        <article className="panel chart-panel">
          <h3>Tranche de puissance fiscale</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byPuissance}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {byPuissance.map((entry, i) => (
                  <Cell key={entry.label} fill={PUISS_COLORS[i % PUISS_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Type véhicule */}
        <article className="panel chart-panel">
          <h3>Type de véhicule</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byType} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={145} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" fill="#004A8D" radius={[0, 6, 6, 0]}
                label={{ position: "right", fontSize: 11 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </div>
    </div>
  );
}