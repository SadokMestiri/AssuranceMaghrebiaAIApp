import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const BRANCH_COLORS = { AUTO: "#004A8D", IRDS: "#F38F1D", SANTE: "#2E7D32" };

function buildChartData(items) {
  const map = new Map();
  (items || []).forEach(({ periode, branche, taux_annulation_pct }) => {
    if (!map.has(periode)) map.set(periode, { periode });
    map.get(periode)[branche] = taux_annulation_pct;
  });
  return Array.from(map.values()).sort((a, b) => a.periode.localeCompare(b.periode));
}

export default function ResiliationChart({ data }) {
  const chartData = buildChartData(data);
  const branches = [...new Set((data || []).map((d) => d.branche))].sort();
  const avg = (data || []).length > 0
    ? Math.round((data || []).reduce((s, d) => s + Number(d.taux_annulation_pct || 0), 0) / data.length * 10) / 10
    : null;

  return (
    <article className="panel chart-panel" style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
      <h3>Taux d&apos;annulation de quittances par branche</h3>
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
          {avg !== null && (
            <ReferenceLine
              y={avg}
              stroke="#64748b"
              strokeDasharray="5 3"
              strokeWidth={1.2}
              label={{ value: `Moy. ${avg}%`, position: "insideTopRight", fontSize: 10, fill: "#64748b" }}
            />
          )}
          <Tooltip
            formatter={(v, name) => [`${Number(v || 0).toFixed(1)} % annulées`, name]}
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
