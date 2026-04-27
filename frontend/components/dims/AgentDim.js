import {
  Bar, BarChart, CartesianGrid, Cell,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import DimKpiRow from "./DimKpiRow";

const fmt = new Intl.NumberFormat("fr-TN");
const fmtTND = new Intl.NumberFormat("fr-TN", { style: "currency", currency: "TND", maximumFractionDigits: 0 });

const GROUPE_COLORS = {
  "Réseau Direct": "#004A8D",
  "Bancassurance": "#F38F1D",
  "Mandataire":    "#2E7D32",
  "Courtier":      "#C62828",
  "Agent Général": "#6A1B9A",
};
const ETAT_COLORS = { A: "#2E7D32", R: "#C62828", S: "#F38F1D" };
const ETAT_LABELS = { A: "Actif", R: "Résilié", S: "Suspendu" };

export default function AgentDim({ data }) {
  if (!data) return <p className="dim-loading">Chargement agents…</p>;

  const { kpis, groupes, typeAgent, etat, topAgentsPnet, topAgentsPolices, localites } = data;

  const etatChart = etat.map((e) => ({ ...e, label: ETAT_LABELS[e.label] || e.label }));

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "🏢", title: "Total agents",     value: fmt.format(kpis.total),            sub: "réseau distribution" },
        { icon: "✅", title: "Actifs",            value: fmt.format(kpis.actifs),           sub: `${kpis.pct_actifs.toFixed(1)} %` },
        { icon: "🔴", title: "Inactifs",          value: fmt.format(kpis.inactifs),         sub: "résiliés + suspendus" },
        { icon: "💼", title: "Groupes",           value: fmt.format(kpis.nb_groupes),       sub: "canaux distribution" },
        { icon: "📍", title: "Localités",         value: fmt.format(kpis.nb_localites),     sub: "couverture géo" },
        { icon: "💰", title: "Prime nette moy.",  value: fmtTND.format(kpis.avg_pnet),      sub: "par agent actif" },
      ]} />

      <div className="dim-charts-grid">
        {/* Top agents by prime nette */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 agents — prime nette émise</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topAgentsPnet} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v/1e6).toFixed(1)}M`} />
              <YAxis type="category" dataKey="nom" tick={{ fontSize: 11 }} width={110} />
              <Tooltip formatter={(v) => fmtTND.format(v)} />
              <Bar dataKey="pnet" name="Prime nette" fill="#004A8D" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Top agents by polices */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 agents — nombre de polices</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topAgentsPolices} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="nom" tick={{ fontSize: 11 }} width={110} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="nb_polices" name="Polices" fill="#F38F1D" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Répartition par groupe */}
        <article className="panel chart-panel">
          <h3>Répartition par groupe</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={groupes} dataKey="count" nameKey="label"
                outerRadius={85} innerRadius={40} paddingAngle={3}
                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
              >
                {groupes.map((entry) => (
                  <Cell key={entry.label} fill={GROUPE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* État agents */}
        <article className="panel chart-panel">
          <h3>État des agents</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={etatChart} dataKey="count" nameKey="label"
                outerRadius={85} innerRadius={40} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {etat.map((entry) => (
                  <Cell key={entry.label} fill={ETAT_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Type agent */}
        <article className="panel chart-panel">
          <h3>Type d'agent</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={typeAgent}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" fill="#1B68B2" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Top localités */}
        <article className="panel chart-panel">
          <h3>Top localités agents</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={localites.slice(0, 10)}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={55} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" fill="#2E7D32" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </div>
    </div>
  );
}
