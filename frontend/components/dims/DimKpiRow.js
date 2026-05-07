function computeVariation(current, previous) {
  const curr = Number(current ?? 0);
  const prev = Number(previous ?? 0);
  if (previous == null || previous === undefined || prev === 0) return null;
  return ((curr - prev) / Math.abs(prev)) * 100;
}

function VariationBadge({ current, previous, invertColor = false }) {
  const pct = computeVariation(current, previous);
  if (pct === null) return null;

  const isPositive = pct >= 0;
  const isGood     = invertColor ? !isPositive : isPositive;
  const color      = isGood ? "#16a34a" : "#dc2626";
  const bg         = isGood ? "rgba(22,163,74,0.10)" : "rgba(220,38,38,0.10)";
  const arrow      = isPositive ? "▲" : "▼";

  return (
    <span style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "3px",
      fontSize: "11px",
      fontWeight: 600,
      color,
      background: bg,
      borderRadius: "4px",
      padding: "2px 6px",
      marginTop: "4px",
    }}>
      {arrow} {Math.abs(pct).toFixed(1)}% vs N-1
    </span>
  );
}

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