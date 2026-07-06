import { asNumber, formatShortCurrency, VariationBadge } from "./shared";

export default function KpiCards({ dashboard, dashboardPrev }) {
  const overview     = dashboard?.overview     || {};
  const production   = overview.production     || {};
  const portefeuille = overview.portefeuille   || {};

  const prevOverview     = dashboardPrev?.overview     || {};
  const prevProduction   = prevOverview.production     || {};
  const prevPortefeuille = prevOverview.portefeuille   || {};

  const cards = [
    {
      title:       "Prime nette",
      value:       formatShortCurrency(production.total_pnet),
      helper:      `${asNumber(production.nb_quittances)} quittances`,
      current:     production.total_pnet,
      previous:    prevProduction.total_pnet,
      invertColor: false,
    },
    {
      title:       "Ratio Combiné",
      value:       asNumber(overview.ratio_combine, "%"),
      current:     overview.ratio_combine,
      previous:    prevOverview.ratio_combine,
      invertColor: true,
    },
    {
      title:       "Taux resiliation",
      value:       asNumber(portefeuille.taux_churn_pct, "%"),
      helper:      `${asNumber(portefeuille.polices_resiliees)} polices resiliees`,
      current:     portefeuille.taux_churn_pct,
      previous:    prevPortefeuille.taux_churn_pct,
      invertColor: true,
    },
    {
      title:       "Nombre de Sinistres",
      value:       asNumber(overview.nb_sinistres),
      current:     overview.nb_sinistres,
      previous:    prevOverview.nb_sinistres,
      invertColor: true,
    },
  ];

  return (
    <section className="kpi-grid">
      {cards.map((card) => (
        <article key={card.title} className="kpi-card">
          <p className="kpi-label">{card.title}</p>
          <p className="kpi-value">{card.value}</p>
          {card.helper && <p className="kpi-helper">{card.helper}</p>}
          <VariationBadge
            current={card.current}
            previous={card.previous}
            invertColor={card.invertColor}
          />
        </article>
      ))}
    </section>
  );
}