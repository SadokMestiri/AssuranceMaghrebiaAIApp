const currencyFormatter = new Intl.NumberFormat("fr-TN", {
  style: "currency",
  currency: "TND",
  maximumFractionDigits: 0,
});

const numberFormatter = new Intl.NumberFormat("fr-TN", {
  maximumFractionDigits: 2,
});

function asCurrency(value) {
  return currencyFormatter.format(Number(value || 0));
}

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
      value: asCurrency(production.total_pnet),
      helper: `${asNumber(production.nb_quittances)} quittances`,
    },
    {
      title: "Ratio Combiné",
      value: asNumber(overview.ratio_combine, "%"),
      helper: `Selon filtre branche`,
    },
    {
      title: "Taux resiliation",
      value: asNumber(portefeuille.taux_churn_pct, "%"),
      helper: `${asNumber(portefeuille.polices_resiliees)} polices resiliees`,
    },
    {
      title: "Nombre de Sinistres",
      value: asNumber(overview.nb_sinistres),
      helper: `Selon filtre branche`,
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
