import {
  Bar, BarChart, CartesianGrid, Cell, Legend,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import DimKpiRow from "./DimKpiRow";

const fmt = new Intl.NumberFormat("fr-TN");
const fmtPct = (v) => `${Number(v || 0).toFixed(1)} %`;

export default function ClientDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement clients…</p>;

  const { kpis = {}, sexe = [], typePersonne = [], ageTranches = [], natp = [], topVilles = [], churnBySexe = [] } = data;
  const prev = dataPrev?.kpis || {};

  const SEXE_COLORS = { F: "#E91E8C", M: "#004A8D", "N/A": "#94a3b8" };
  const TYPE_COLORS  = { P: "#004A8D", M: "#F38F1D" };
  const AGE_COLOR    = "#1B68B2";
  const NATP_COLORS  = ["#004A8D", "#F38F1D", "#2E7D32", "#C62828"];

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "👥", title: "Total clients",     value: fmt.format(kpis.total),         sub: "portefeuille actif",             current: kpis.total,      previous: prev.total },
        { icon: "♀",  title: "Femmes",            value: fmtPct(kpis.pct_f),             sub: fmt.format(kpis.nb_f) + " clients", current: kpis.pct_f,    previous: prev.pct_f },
        { icon: "♂",  title: "Hommes",            value: fmtPct(kpis.pct_m),             sub: fmt.format(kpis.nb_m) + " clients", current: kpis.pct_m,    previous: prev.pct_m },
        { icon: "🏢", title: "Personnes morales", value: fmtPct(kpis.pct_moral),         sub: "vs physiques",                   current: kpis.pct_moral,  previous: prev.pct_moral },
        { icon: "📍", title: "Villes couvertes",  value: fmt.format(kpis.nb_villes),     sub: "localisations distinctes",       current: kpis.nb_villes,  previous: prev.nb_villes },
        { icon: "🎂", title: "Âge moyen",         value: kpis.age_moyen + " ans",        sub: "clients avec date naissance",    current: kpis.age_moyen,  previous: prev.age_moyen },
      ]} />

      <div className="dim-charts-grid">
        {/* Répartition par sexe */}
        <article className="panel chart-panel">
          <h3>Répartition par sexe</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={sexe} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
              >
                {sexe.map((entry) => (
                  <Cell key={entry.label} fill={SEXE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Distribution par tranche d'âge */}
        <article className="panel chart-panel">
          <h3>Tranches d'âge</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={ageTranches} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 12 }} width={55} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" fill={AGE_COLOR} radius={[0, 6, 6, 0]} label={{ position: "right", fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Type de personne */}
        <article className="panel chart-panel">
          <h3>Type de personne</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={typePersonne} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
              >
                {typePersonne.map((entry, i) => (
                  <Cell key={entry.label} fill={TYPE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Nationalité */}
        <article className="panel chart-panel">
          <h3>Nationalité (NATP)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie data={natp} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={32} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
              >
                {natp.map((entry, i) => (
                  <Cell key={entry.label} fill={NATP_COLORS[i % NATP_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Churn by sexe */}
        {churnBySexe && churnBySexe.length > 0 && (
          <article className="panel chart-panel">
            <h3>Résiliations par sexe</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={churnBySexe}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                <XAxis dataKey="label" tick={{ fontSize: 13 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => fmt.format(v)} />
                <Bar dataKey="resiliees" name="Polices résiliées" fill="#E91E8C" radius={[6, 6, 0, 0]} />
                <Bar dataKey="total" name="Total polices" fill="#004A8D" radius={[6, 6, 0, 0]} />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </article>
        )}

        {/* Top 10 villes */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 villes — concentration clients</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={topVilles}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" fill="#004A8D" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>

      </div>
    </div>
  );
}