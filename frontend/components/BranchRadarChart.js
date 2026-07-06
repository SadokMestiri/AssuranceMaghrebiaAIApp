import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const BRANCH_COLORS = { AUTO: "#004A8D", IRDS: "#F38F1D", SANTE: "#2E7D32" };

function buildRadarData(items) {
  const totalPolices = items.reduce((s, i) => s + Number(i.nb_polices || 0), 1);

  // Derive per-unit metrics so branches are comparable regardless of size
  const derived = items.map((i) => ({
    branche:          i.branche,
    part_portefeuille: Number(i.nb_polices || 0) / totalPolices * 100,
    freq_sinistres:   Number(i.nb_polices || 0) > 0
                        ? Number(i.nb_sinistres || 0) / Number(i.nb_polices) * 100
                        : 0,
    prime_moyenne:    Number(i.nb_polices || 0) > 0
                        ? Number(i.total_pnet || 0) / Number(i.nb_polices)
                        : 0,
    taux_sp:          Number(i.taux_sinistres_pct || 0),
  }));

  const axes = [
    { key: "part_portefeuille", label: "Part portefeuille" },
    { key: "freq_sinistres",    label: "Fréq. sinistres" },
    { key: "prime_moyenne",     label: "Prime moyenne" },
    { key: "taux_sp",           label: "Taux S/P" },
  ];

  return axes.map(({ key, label }) => {
    const max = Math.max(...derived.map((d) => d[key]), 1);
    const entry = { metric: label };
    derived.forEach((d) => {
      entry[d.branche] = Math.round((d[key] / max) * 100);
    });
    return entry;
  });
}

export default function BranchRadarChart({ data }) {
  if (!data || data.length === 0) return null;

  const radarData = buildRadarData(data);
  const branches = data.map((d) => d.branche);

  return (
    <article className="panel chart-panel">
      <h3>Profil de risque par branche</h3>
      <ResponsiveContainer width="100%" height={270}>
        <RadarChart data={radarData} margin={{ top: 10, right: 40, bottom: 10, left: 40 }}>
          <PolarGrid stroke="rgba(0,0,0,0.1)" />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fontWeight: 500 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} tickCount={4} />
          {branches.map((branch) => (
            <Radar
              key={branch}
              name={branch}
              dataKey={branch}
              stroke={BRANCH_COLORS[branch] || "#94a3b8"}
              fill={BRANCH_COLORS[branch] || "#94a3b8"}
              fillOpacity={0.18}
              strokeWidth={2}
            />
          ))}
          <Legend />
          <Tooltip
            formatter={(value, name) => [`${value} / 100`, name]}
            labelFormatter={(label) => `Métrique : ${label}`}
          />
        </RadarChart>
      </ResponsiveContainer>
    </article>
  );
}
