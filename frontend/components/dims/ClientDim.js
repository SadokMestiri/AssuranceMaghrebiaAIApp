import {
  Bar, BarChart, CartesianGrid, Cell, Legend,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import DimKpiRow from "./DimKpiRow";
import { Users, UserRound, UserCheck, Building2, MapPin, Calendar } from 'lucide-react';

const fmt = new Intl.NumberFormat("fr-TN");
const fmtPct = (v) => `${Number(v || 0).toFixed(1)} %`;

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

export default function ClientDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement clients…</p>;

  const { kpis = {}, sexe = [], typePersonne = [], ageTranches: rawAgeTranches = [], natp = [], topVilles = [], churnBySexe = [] } = data;
  const ageTranches = rawAgeTranches.filter((e) => (e.count || 0) > 0);
  const prev = dataPrev?.kpis || {};

  const SEXE_COLORS = { F: "#E91E8C", M: "#004A8D", "N/A": "#94a3b8" };
  const TYPE_COLORS  = { P: "#004A8D", M: "#F38F1D" };
  const AGE_COLOR    = "#1B68B2";
  const NATP_COLORS  = ["#004A8D", "#F38F1D", "#2E7D32", "#C62828"];

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: <Users size={18} />,       title: "Total clients",     value: fmt.format(kpis.total),         sub: "portefeuille actif",             current: kpis.total,      previous: prev.total },
        { icon: <UserRound size={18} />,   title: "Femmes",            value: fmtPct(kpis.pct_f),             sub: fmt.format(kpis.nb_f) + " clients", current: kpis.pct_f,    previous: prev.pct_f },
        { icon: <UserCheck size={18} />,   title: "Hommes",            value: fmtPct(kpis.pct_m),             sub: fmt.format(kpis.nb_m) + " clients", current: kpis.pct_m,    previous: prev.pct_m },
        { icon: <Building2 size={18} />,   title: "Personnes morales", value: fmtPct(kpis.pct_moral),         sub: "vs physiques",                   current: kpis.pct_moral,  previous: prev.pct_moral },
        { icon: <MapPin size={18} />,      title: "Villes couvertes",  value: fmt.format(kpis.nb_villes),     sub: "localisations distinctes",       current: kpis.nb_villes,  previous: prev.nb_villes },
        { icon: <Calendar size={18} />,    title: "Âge moyen",         value: kpis.age_moyen + " ans",        sub: "clients avec date naissance",    current: kpis.age_moyen,  previous: prev.age_moyen },
      ]} />

      <div className="dim-charts-grid">
        {/* Répartition par sexe */}
        <article className="panel chart-panel">
          <h3>Répartition par sexe</h3>
          <ResponsiveContainer width="100%" height={270}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={sexe} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {sexe.map((entry) => (
                  <Cell key={entry.label} fill={SEXE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Distribution par tranche d'âge */}
        <article className="panel chart-panel">
          <h3>Tranches d'âge</h3>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={ageTranches} layout="vertical" margin={{ right: 45 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 12 }} width={55} interval={0} />
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Bar dataKey="count" name="Clients" fill={AGE_COLOR} radius={[0, 6, 6, 0]} label={{ position: "right", fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Type de personne */}
        <article className="panel chart-panel">
          <h3>Type de personne</h3>
          <ResponsiveContainer width="100%" height={270}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={typePersonne} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {typePersonne.map((entry, i) => (
                  <Cell key={entry.label} fill={TYPE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Nationalité */}
        <article className="panel chart-panel">
          <h3>Nationalité (NATP)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={natp} dataKey="count" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {natp.map((entry, i) => (
                  <Cell key={entry.label} fill={NATP_COLORS[i % NATP_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Résiliations par sexe + Top 10 villes — side by side, spanning full grid width */}
        <div style={{ gridColumn: "1 / -1", display: "flex", gap: "1rem" }}>
          {churnBySexe && churnBySexe.length > 0 && (
            <article className="panel chart-panel" style={{ flex: "0 0 300px" }}>
              <h3>Résiliations par sexe</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={churnBySexe}>
                  <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                  <XAxis dataKey="label" tick={{ fontSize: 13 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
                  <Bar dataKey="resiliees" name="Polices résiliées" fill="#E91E8C" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="total" name="Total polices" fill="#004A8D" radius={[6, 6, 0, 0]} />
                  <Legend />
                </BarChart>
              </ResponsiveContainer>
            </article>
          )}

          <article className="panel chart-panel" style={{ flex: 1, minWidth: 0 }}>
            <h3>Top 10 villes — concentration clients</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={topVilles} layout="vertical" margin={{ right: 45 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={85} interval={0} />
                <Tooltip formatter={(v) => [fmt.format(v), 'Nombre']} />
                <Bar dataKey="count" name="Clients" fill="#004A8D" radius={[0, 6, 6, 0]}
                  label={{ position: "right", fontSize: 11 }} />
              </BarChart>
            </ResponsiveContainer>
          </article>
        </div>

      </div>
    </div>
  );
}