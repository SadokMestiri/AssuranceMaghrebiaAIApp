import { VariationBadge } from "../shared";

export default function DimKpiRow({ cards }) {
  return (
    <div className="dim-kpi-row">
      {cards.map((card) => (
        <article key={card.title} className="dim-kpi-card">
          <span className="dim-kpi-icon">{card.icon}</span>
          <div className="dim-kpi-body">
            <p className="dim-kpi-value">{card.value}</p>
            <p className="dim-kpi-label">{card.title}</p>
            {card.sub && <p className="dim-kpi-sub">{card.sub}</p>}
            <VariationBadge
              current={card.current}
              previous={card.previous}
              invertColor={card.invertColor ?? false}
            />
          </div>
        </article>
      ))}
    </div>
  );
}