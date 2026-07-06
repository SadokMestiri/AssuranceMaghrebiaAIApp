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

const GROUPE_COLORS = {
  "Réseau Direct": "#004A8D",
  "Bancassurance": "#F38F1D",
  "Mandataire":    "#2E7D32",
  "Courtier":      "#C62828",
  "Agent Général": "#6A1B9A",
};
const ETAT_COLORS = { A: "#2E7D32", R: "#C62828", S: "#F38F1D" };
const ETAT_LABELS = { A: "Actif", R: "Résilié", S: "Suspendu" };

export default function AgentDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement agents…</p>;

  const { kpis = {}, groupes = [], typeAgent = [], etat = [], topAgentsPnet = [], topAgentsPolices = [], localites = [] } = data;
  const prev = dataPrev?.kpis || {};

  const etatChart = etat.map((e) => ({ ...e, label: ETAT_LABELS[e.label] || e.label }));

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "🏢", title: "Total agents",     value: fmt.format(kpis.total),           sub: "réseau distribution",  current: kpis.total,     previous: prev.total },
        { icon: "✅", title: "Actifs",            value: fmt.format(kpis.actifs),          sub: `${Number(kpis.pct_actifs || 0).toFixed(1)} %`,               current: kpis.actifs,    previous: prev.actifs },
        { icon: "🔴", title: "Inactifs",          value: fmt.format(kpis.inactifs),        sub: "résiliés + suspendus", current: kpis.inactifs,  previous: prev.inactifs,  invertColor: true },
        { icon: "💼", title: "Groupes",           value: fmt.format(kpis.nb_groupes),      sub: "canaux distribution",  current: kpis.nb_groupes },
        { icon: "📍", title: "Localités",         value: fmt.format(kpis.nb_localites),    sub: "couverture géo",       current: kpis.nb_localites, previous: prev.nb_localites },
        { icon: "💰", title: "Prime nette moy.",  value: formatShortCurrency(kpis.avg_pnet), sub: "par agent actif",    current: kpis.avg_pnet,  previous: prev.avg_pnet },
      ]} />

      <div className="dim-charts-grid">
        {/* Top agents by prime nette */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 agents — prime nette émise</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topAgentsPnet} layout="vertical" margin={{ right: 80 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v/1e6).toFixed(1)}M`} />
              <YAxis type="category" dataKey="nom" tick={{ fontSize: 11 }} width={110} />
              <Tooltip formatter={(v) => formatShortCurrency(v)} />
              <Bar dataKey="pnet" name="Prime nette" fill="#004A8D" radius={[0, 6, 6, 0]}
                label={{ position: "right", fontSize: 10, formatter: (v) => formatShortCurrency(v) }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Top agents by polices */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Top 10 agents — nombre de polices</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topAgentsPolices} layout="vertical" margin={{ right: 45 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="nom" tick={{ fontSize: 11 }} width={110} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Bar dataKey="nb_polices" name="Polices" fill="#F38F1D" radius={[0, 6, 6, 0]}
                label={{ position: "right", fontSize: 10 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Répartition par groupe */}
        <article className="panel chart-panel">
          <h3>Répartition par groupe</h3>
          <ResponsiveContainer width="100%" height={265}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={groupes} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {groupes.map((entry) => (
                  <Cell key={entry.label} fill={GROUPE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: "12px" }} />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* État agents */}
        <article className="panel chart-panel">
          <h3>État des agents</h3>
          <ResponsiveContainer width="100%" height={270}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={etatChart} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {etat.map((entry) => (
                  <Cell key={entry.label} fill={ETAT_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Type agent */}
        <article className="panel chart-panel">
          <h3>Type d'agent</h3>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={typeAgent} margin={{ top: 20 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis dataKey="label" tick={{ fontSize: 13 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Agents']} />
              <Bar dataKey="count" name="Agents" fill="#1B68B2" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Top localités */}
        <article className="panel chart-panel">
          <h3>Top localités agents</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={localites.slice(0, 10)} layout="vertical" margin={{ right: 45 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={80} interval={0} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Agents']} />
              <Bar dataKey="count" name="Agents" fill="#2E7D32" radius={[0, 6, 6, 0]}
                label={{ position: "right", fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </div>
    </div>
  );
}