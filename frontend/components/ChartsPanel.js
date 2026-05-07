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

function pieLabelRenderer({ percent }) {
  return `${(Number(percent || 0) * 100).toFixed(1)}%`;
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
        <ResponsiveContainer width="100%" height={280}>
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
        <ResponsiveContainer width="100%" height={280}>
          <PieChart margin={{ top: 10, right: 40, bottom: 10, left: 40 }}>
            <Pie
              data={branchShare}
              dataKey="total_pnet"
              nameKey="branche"
              outerRadius={72}
              innerRadius={34}
              paddingAngle={3}
              label={pieLabelRenderer}
            >
              {branchShare.map((entry, index) => (
                <Cell
                  key={`${entry.branche}-${index}`}
                  fill={getBranchColor(entry?.branche, index)}
                />
              ))}
            </Pie>
            <Legend verticalAlign="bottom" height={36} />
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </article>

      <article className="panel chart-panel">
        <h3>Taux impayes sur prime nette</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={impayeRate}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(0, 74, 141, 0.2)" />
            <XAxis dataKey="branche" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="taux_impaye_sur_pnet_pct" fill="#F38F1D" radius={[10, 10, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </article>
    </section>
  );
}