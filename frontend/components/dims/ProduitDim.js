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

const BRANCHE_COLORS = { AUTO: "#004A8D", IRDS: "#F38F1D", SANTE: "#2E7D32" };
const FAMILLE_COLORS = [
  "#004A8D", "#F38F1D", "#2E7D32", "#C62828",
  "#6A1B9A", "#00838F", "#5D4037", "#3949AB",
];
const FAMILLE_SHORT = {
  "Santé Locale Groupe":  "Santé Locale",
  "Responsabilité Civile": "Resp. Civile",
  "Risques spéciaux":     "Risques spéc.",
};

export default function ProduitDim({ data, dataPrev }) {
  if (!data) return <p className="dim-loading">Chargement produits…</p>;

  const { kpis = {}, byBranche = [], byFamille: rawFamille = [], topProduits = [], topProduitsQuittances = [] } = data;
  const byFamille = rawFamille.map((e) => ({ ...e, label: FAMILLE_SHORT[e.label] || e.label }));
  const totalFamillePnet = byFamille.reduce((sum, e) => sum + (e.pnet || 0), 0);
  const prev = dataPrev?.kpis || {};

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "📦", title: "Produits distincts", value: fmt.format(kpis.nb_produits),         sub: "catalogue actif",    current: kpis.nb_produits },
        { icon: "🔖", title: "Familles de risque", value: fmt.format(kpis.nb_familles),         sub: "catégories",         current: kpis.nb_familles },
        { icon: "🏦", title: "Branches couvertes", value: fmt.format(kpis.nb_branches),         sub: "AUTO · IRDS · SANTE",current: kpis.nb_branches },
        { icon: "💰", title: "Prime nette totale", value: formatShortCurrency(kpis.total_pnet), sub: "toutes périodes",    current: kpis.total_pnet,   previous: prev.total_pnet },
        { icon: "📄", title: "Quittances émises",  value: fmt.format(kpis.total_quitt),         sub: "toutes périodes",    current: kpis.total_quitt,  previous: prev.total_quitt },
        { icon: "🥇", title: "Produit top",        value: kpis.top_produit,                     sub: formatShortCurrency(kpis.top_pnet), current: kpis.top_pnet, previous: prev.top_pnet },
      ]} />

      <div className="dim-charts-grid">
        {/* Prime nette par produit */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Prime nette par produit (Top 10)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topProduits} layout="vertical" margin={{ right: 75 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v/1e6).toFixed(0)}M`} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={160} />
              <Tooltip formatter={(v) => [formatShortCurrency(v), 'Prime nette']} />
              <Bar dataKey="pnet" name="Prime nette" label={{ position: "right", fontSize: 10, formatter: (v) => formatShortCurrency(v) }}>
                {topProduits.map((entry) => (
                  <Cell key={entry.label} fill={BRANCHE_COLORS[entry.branche] || "#94a3b8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Quittances par produit */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Volume quittances par produit (Top 10)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topProduitsQuittances} layout="vertical" margin={{ right: 45 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={160} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" name="Quittances" fill="#F38F1D" radius={[0, 6, 6, 0]}
                label={{ position: "right", fontSize: 10 }} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Part par branche */}
        <article className="panel chart-panel">
          <h3>Part de prime par branche</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={byBranche} dataKey="pnet" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={3}
                label={renderPieLabel} labelLine={false}
              >
                {byBranche.map((entry) => (
                  <Cell key={entry.label} fill={BRANCHE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [formatShortCurrency(v), 'Prime nette']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Famille de risque */}
        <article className="panel chart-panel">
          <h3>Prime nette par famille de risque</h3>
          <ResponsiveContainer width="100%" height={290}>
            <PieChart margin={{ top: 20, right: 38, bottom: 20, left: 38 }}>
              <Pie data={byFamille} dataKey="pnet" nameKey="label"
                outerRadius={68} innerRadius={30} paddingAngle={2}
                label={renderPieLabel} labelLine={false}
              >
                {byFamille.map((entry, i) => (
                  <Cell key={entry.label} fill={FAMILLE_COLORS[i % FAMILLE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => [formatShortCurrency(v), 'Prime nette']} />
              <Legend
                iconSize={10}
                wrapperStyle={{ fontSize: "11px" }}
                formatter={(value, entry) => {
                  const pct = totalFamillePnet > 0
                    ? (entry.payload?.pnet || 0) / totalFamillePnet * 100
                    : 0;
                  return `${value} (${pct < 1 ? pct.toFixed(1) : pct.toFixed(0)}%)`;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </article>
      </div>
    </div>
  );
}