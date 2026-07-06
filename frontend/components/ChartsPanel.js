import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const BRANCH_COLORS = {
  AUTO: "#004A8D",
  IRDS: "#F38F1D",
  SANTE: "#2E7D32",
};

const FALLBACK_PIE_COLORS = ["#8E24AA", "#C62828", "#00838F", "#5D4037", "#3949AB", "#FFB300"];

function getBranchColor(branch, index) {
  const normalized = String(branch || "").trim().toUpperCase();
  return BRANCH_COLORS[normalized] || FALLBACK_PIE_COLORS[index % FALLBACK_PIE_COLORS.length];
}

const RADIAN = Math.PI / 180;
function pieLabelRenderer({ cx, cy, midAngle, innerRadius, outerRadius, percent }) {
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

function aggregateMonthlyTrend(items, monthFilter) {
  const map = new Map();

  (items || []).forEach((item) => {
    const month = Number(item.mois || 0);
    if (monthFilter !== "ALL" && month !== Number(monthFilter)) {
      return;
    }

    const key = String(item.periode || `${item.annee}-${String(item.mois).padStart(2, "0")}`);
    const current = map.get(key) || { periode: key, total_pnet: 0, total_ptt: 0, nb_quittances: 0 };

    current.total_pnet += Number(item.total_pnet || 0);
    current.total_ptt += Number(item.total_ptt || 0);
    current.nb_quittances += Number(item.nb_quittances || 0);
    map.set(key, current);
  });

  return Array.from(map.values()).sort((a, b) => a.periode.localeCompare(b.periode));
}

export default function ChartsPanel({ dashboard, monthFilter }) {
  const monthlyTrend = aggregateMonthlyTrend(dashboard?.production_monthly_trend || [], monthFilter);
  const branchShare = dashboard?.production_branch_share || [];
  const impayeRate = dashboard?.impayes_rate_by_branch || [];

  return (
    <section className="charts-grid">
      <article className="panel chart-panel">
        <h3>Evolution primes nettes</h3>
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={monthlyTrend}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(0, 74, 141, 0.2)" />
            <XAxis dataKey="periode" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Line type="monotone" dataKey="total_pnet" stroke="#004A8D" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </article>

      <article className="panel chart-panel">
        <h3>Part de production par branche</h3>
        <ResponsiveContainer width="100%" height={270}>
          <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
            <Pie
              data={branchShare}
              dataKey="total_pnet"
              nameKey="branche"
              outerRadius={68}
              innerRadius={30}
              paddingAngle={3}
              label={pieLabelRenderer}
              labelLine={false}
            >
              {branchShare.map((entry, index) => (
                <Cell
                  key={`${entry.branche}-${index}`}
                  fill={getBranchColor(entry?.branche, index)}
                />
              ))}
            </Pie>
            <Legend verticalAlign="bottom" height={36} />
            <Tooltip formatter={(v) => [new Intl.NumberFormat("fr-TN", { style: "currency", currency: "TND", maximumFractionDigits: 0 }).format(v), 'Prime nette']} />
          </PieChart>
        </ResponsiveContainer>
      </article>

      <article className="panel chart-panel">
        <h3>Taux impayes sur prime nette</h3>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={impayeRate} margin={{ top: 20 }}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(0, 74, 141, 0.2)" />
            <XAxis dataKey="branche" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="taux_impaye_sur_pnet_pct" fill="#F38F1D" radius={[10, 10, 0, 0]}
              label={{ position: "top", fontSize: 11, formatter: (v) => `${Number(v || 0).toFixed(1)}%` }} />
          </BarChart>
        </ResponsiveContainer>
      </article>
    </section>
  );
}