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

function computeVariation(current, previous) {
  const curr = Number(current || 0);
  const prev = Number(previous || 0);
  if (prev === 0 || previous == null) return null;
  return ((curr - prev) / Math.abs(prev)) * 100;
}

function VariationBadge({ current, previous, invertColor = false }) {
  const pct = computeVariation(current, previous);
  if (pct === null) return null;

  const isPositive = pct >= 0;
  // invertColor: for metrics where an increase is bad (résiliation, sinistres, ratio combiné)
  const isGood = invertColor ? !isPositive : isPositive;

  const color = isGood ? "#16a34a" : "#dc2626";
  const bg    = isGood ? "rgba(22,163,74,0.10)" : "rgba(220,38,38,0.10)";
  const arrow = isPositive ? "▲" : "▼";

  return (
    <span style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "3px",
      fontSize: "12px",
      fontWeight: 600,
      color,
      background: bg,
      borderRadius: "4px",
      padding: "2px 7px",
      marginTop: "4px",
    }}>
      {arrow} {Math.abs(pct).toFixed(1)}% vs N-1
    </span>
  );
}

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