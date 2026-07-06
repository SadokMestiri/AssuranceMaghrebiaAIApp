import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const BRANCH_COLORS = { AUTO: "#004A8D", IRDS: "#F38F1D", SANTE: "#2E7D32" };

function buildChartData(items) {
  // YTD cumulative frequency: nb_sinistres / nb_polices × 100, resets each year
  const byBranche = {};
  (items || []).forEach((row) => {
    if (!byBranche[row.branche]) byBranche[row.branche] = [];
    byBranche[row.branche].push(row);
  });

  const ytd = {};
  Object.entries(byBranche).forEach(([branche, rows]) => {
    rows.sort((a, b) => a.periode.localeCompare(b.periode));
    let cumSin = 0, cumPol = 0, currentYear = null;
    rows.forEach(({ periode, nb_sinistres, nb_polices }) => {
      const year = periode.slice(0, 4);
      if (year !== currentYear) { cumSin = 0; cumPol = 0; currentYear = year; }
      cumSin += Number(nb_sinistres || 0);
      cumPol += Number(nb_polices  || 0);
      if (!ytd[periode]) ytd[periode] = { periode };
      ytd[periode][branche] = cumPol > 0
        ? Math.round(cumSin / cumPol * 1000) / 10
        : null;
    });
  });

  return Object.values(ytd).sort((a, b) => a.periode.localeCompare(b.periode));
}

export default function SpEvolutionChart({ data }) {
  const chartData = buildChartData(data);
  const branches = [...new Set((data || []).map((d) => d.branche))].sort();

  return (
    <article className="panel chart-panel" style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
      <h3>Fréquence sinistres YTD par branche (%)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.12)" />
          <XAxis
            dataKey="periode"
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => v.slice(2)}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => `${v}%`}
            width={42}
          />
          <Tooltip
            formatter={(v, name) => [`${Number(v || 0).toFixed(1)} % sinistres/polices`, name]}
            labelFormatter={(l) => `Période : ${l}`}
          />
          <Legend />
          {branches.map((branch) => (
            <Line
              key={branch}
              type="monotone"
              dataKey={branch}
              stroke={BRANCH_COLORS[branch] || "#94a3b8"}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </article>
  );
}
