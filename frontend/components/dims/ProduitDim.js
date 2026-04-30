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

const BRANCHE_COLORS = { AUTO: "#004A8D", IRDS: "#F38F1D", SANTE: "#2E7D32" };
const FAMILLE_COLORS = [
  "#004A8D", "#F38F1D", "#2E7D32", "#C62828",
  "#6A1B9A", "#00838F", "#5D4037", "#3949AB",
];

export default function ProduitDim({ data }) {
  if (!data) return <p className="dim-loading">Chargement produits…</p>;

  const { kpis, byBranche, byFamille, topProduits, topProduitsQuittances } = data;

  return (
    <div className="dim-panel">
      <DimKpiRow cards={[
        { icon: "📦", title: "Produits distincts",  value: fmt.format(kpis.nb_produits),    sub: "catalogue actif" },
        { icon: "🔖", title: "Familles de risque",  value: fmt.format(kpis.nb_familles),    sub: "catégories" },
        { icon: "🏦", title: "Branches couvertes",  value: fmt.format(kpis.nb_branches),    sub: "AUTO · IRDS · SANTE" },
        { icon: "💰", title: "Prime nette totale",  value: formatShortCurrency(kpis.total_pnet),  sub: "toutes périodes" },
        { icon: "📄", title: "Quittances émises",   value: fmt.format(kpis.total_quitt),    sub: "toutes périodes" },
        { icon: "🥇", title: "Produit top",         value: kpis.top_produit,                sub: formatShortCurrency(kpis.top_pnet) },
      ]} />

      <div className="dim-charts-grid">
        {/* Prime nette par produit */}
        <article className="panel chart-panel dim-chart-wide">
          <h3>Prime nette par produit (Top 10)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topProduits} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v/1e6).toFixed(0)}M`} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={160} />
              <Tooltip formatter={(v) => formatShortCurrency(v)} />
              <Bar dataKey="pnet" name="Prime nette">
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
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topProduitsQuittances} layout="vertical">
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,74,141,0.15)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={160} />
              <Tooltip formatter={(v) => fmt.format(v)} />
              <Bar dataKey="count" name="Quittances" fill="#F38F1D" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>

        {/* Part par branche */}
        <article className="panel chart-panel">
          <h3>Part de prime par branche</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={byBranche} dataKey="pnet" nameKey="label"
                outerRadius={85} innerRadius={40} paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
              >
                {byBranche.map((entry) => (
                  <Cell key={entry.label} fill={BRANCHE_COLORS[entry.label] || "#94a3b8"} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => formatShortCurrency(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>

        {/* Famille de risque */}
        <article className="panel chart-panel">
          <h3>Prime nette par famille de risque</h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={byFamille} dataKey="pnet" nameKey="label"
                outerRadius={85} innerRadius={40} paddingAngle={3}
                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
              >
                {byFamille.map((entry, i) => (
                  <Cell key={entry.label} fill={FAMILLE_COLORS[i % FAMILLE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => formatShortCurrency(v)} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </article>
      </div>
    </div>
  );
}
