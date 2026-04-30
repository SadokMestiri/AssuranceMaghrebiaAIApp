function formatShortCurrency(value) {
  const num = Number(value || 0);
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M DT`;
  } else if (num >= 1_000) {
    return `${Math.round(num / 1_000)}K DT`;
  }
  return `${num} DT`;
}

const numberFormatter = new Intl.NumberFormat("fr-TN", {
  maximumFractionDigits: 2,
});

function asNumber(value, suffix = "") {
  return `${numberFormatter.format(Number(value || 0))}${suffix}`;
}

export default function KpiCards({ dashboard }) {
  const overview = dashboard?.overview || {};
  const production = overview.production || {};
  const portefeuille = overview.portefeuille || {};
  const impayes = overview.impayes || {};

  const ml_ltv = dashboard?.ml_ltv || {};
  const ml_churn = dashboard?.ml_churn || {};

  const cards = [
    {
      title: "Prime nette",
      value: formatShortCurrency(production.total_pnet),
      helper: `${asNumber(production.nb_quittances)} quittances`,
    },
    {
      title: "Ratio Combiné",
      value: asNumber(overview.ratio_combine, "%"),
      
    },
    {
      title: "Taux resiliation",
      value: asNumber(portefeuille.taux_churn_pct, "%"),
      helper: `${asNumber(portefeuille.polices_resiliees)} polices resiliees`,
    },
    {
      title: "Nombre de Sinistres",
      value: asNumber(overview.nb_sinistres),
      
    },
  ];

  return (
    <section className="kpi-grid">
      {cards.map((card) => (
        <article key={card.title} className="kpi-card">
          <p className="kpi-label">{card.title}</p>
          <p className="kpi-value">{card.value}</p>
          <p className="kpi-helper">{card.helper}</p>
        </article>
      ))}
    </section>
  );
}
