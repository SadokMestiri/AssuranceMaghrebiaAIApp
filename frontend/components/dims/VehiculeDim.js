import {
  Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";
import { Car, Gauge, Truck, Wrench, Zap, CalendarDays, Info } from 'lucide-react';

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

const GENRE_COLORS = {
  VP: "#004A8D", VU: "#F38F1D", TC: "#2E7D32",
  AR: "#C62828", PL: "#6A1B9A",
};
const PUISS_COLORS = ["#004A8D", "#1B68B2", "#F38F1D", "#C62828"];
const MARQUE_COLORS = [
  "#004A8D","#F38F1D","#2E7D32","#C62828",
  "#6A1B9A","#00838F","#5D4037","#3949AB","#E91E8C","#00ACC1",
];

export default function VehiculeDim({ data, dataPrev, branch = "ALL" }) {
  if (!data) return <p className="dim-loading">Chargement véhicules…</p>;

  const { kpis = {}, byGenre = [], byPuissance = [], topMarques = [], byType = [] } = data;
  const prev = dataPrev?.kpis || {};

  const branchNotice =
    branch !== "ALL" && branch !== "AUTO" ? (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "10px 14px",
          marginBottom: "16px",
          background: "rgba(217,119,6,0.08)",
          border: "1px solid rgba(217,119,6,0.3)",
          borderRadius: "8px",
          fontSize: "13px",
          color: "#92400e",
        }}
      >
        <Info size={16} />
        <span>
          La branche <strong>{branch}</strong> n&apos;a pas de données véhicules. Le parc affiché
          couvre toutes les branches.
        </span>
      </div>
    ) : null;

  return (
    <div className="dim-panel">
      {branchNotice}
      <DimKpiRow cards={[
        { icon: <Car size={18} />,         title: "Total véhicules",    value: fmt.format(kpis.total),              sub: "assurés",            current: kpis.total,          previous: prev.total },
        { icon: <Gauge size={18} />,       title: "Voitures part.",     value: fmt.format(kpis.nb_vp),              sub: `${Number(kpis.pct_vp || 0).toFixed(1)} % du parc`, current: kpis.nb_vp, previous: prev.nb_vp },
        { icon: <Truck size={18} />,       title: "Véh. utilitaires",  value: fmt.format(kpis.nb_vu),              sub: "VU + PL + TC",       current: kpis.nb_vu,          previous: prev.nb_vu },
        { icon: <Wrench size={18} />,      title: "Marques distinctes", value: fmt.format(kpis.nb_marques),         sub: "constructeurs",      current: kpis.nb_marques },
        { icon: <Zap size={18} />,         title: "Puissance moy.",     value: kpis.avg_puissance + " CV",          sub: "moyenne parc",       current: kpis.avg_puissance,  previous: prev.avg_puissance },
        { icon: <CalendarDays size={18} />,title: "Ancienneté moy.",    value: kpis.avg_age + " ans",               sub: "âge moyen véhicule", current: kpis.avg_age,        previous: prev.avg_age,       invertColor: true },
      ]} />

      <div className="dim-charts-grid">
        {/* Top marques */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 marques — parc assuré</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topMarques} margin={{ top: 20 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Bar dataKey="count" name="Véhicules" label={{ position: "top", fontSize: 10 }}>
                {topMarques.map((entry, i) => (
                  <Cell key={entry.label} fill={MARQUE_COLORS[i % MARQUE_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Genre + Tranche + Type — flex row, Type gets all remaining space */}
        <div style={{ gridColumn: "1 / -1", display: "flex", gap: "1rem" }}>
          <article className="panel chart-panel" style={{ flex: "0 0 250px" }}>
            <h3>Genre de véhicule</h3>
            <ResponsiveContainer width="100%" height={270}>
              <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
                <Pie data={byGenre} dataKey="count" nameKey="label"
                  outerRadius={68} innerRadius={30} paddingAngle={3}
                  label={renderPieLabel} labelLine={false}
                >
                  {byGenre.map((entry) => (
                    <Cell key={entry.label} fill={GENRE_COLORS[entry.label] || "#94a3b8"} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </article>

          <article className="panel chart-panel" style={{ flex: "0 0 280px" }}>
            <h3>Tranche de puissance fiscale</h3>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={byPuissance} margin={{ top: 20 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={0} angle={-35} textAnchor="end" height={45} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [fmt.format(v), 'Véhicules']} />
                <Bar dataKey="count" name="Véhicules" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11 }}>
                  {byPuissance.map((entry, i) => (
                    <Cell key={entry.label} fill={PUISS_COLORS[i % PUISS_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </article>

          <article className="panel chart-panel" style={{ flex: 1, minWidth: 0 }}>
            <h3>Type de véhicule</h3>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={byType} layout="vertical" margin={{ right: 55 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={145} interval={0} />
                <Tooltip formatter={(v) => [fmt.format(v), 'Véhicules']} />
                <Bar dataKey="count" name="Véhicules" fill="#004A8D" radius={[0, 6, 6, 0]}
                  label={{ position: "right", fontSize: 11 }}
                />
              </BarChart>
            </ResponsiveContainer>
          </article>
        </div>
      </div>
    </div>
  );
}