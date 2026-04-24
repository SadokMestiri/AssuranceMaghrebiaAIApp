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
      title: "LTV Moyen (ML)",
      value: asCurrency(ml_ltv.avg_ltv),
      helper: `Ancienneté Moy: ${asNumber(ml_ltv.avg_anciennete_annees)} ans`,
    },
    {
      title: "Taux resiliation",
      value: asNumber(portefeuille.taux_churn_pct, "%"),
      helper: `${asNumber(portefeuille.polices_resiliees)} polices resiliees`,
    },
    {
      title: "Risque Churn (ML)",
      value: asNumber(ml_churn.avg_sinistralite_pct, "%"),
      helper: `Impayé lié: ${asNumber(ml_churn.avg_impaye_pct, "%")}`,
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
