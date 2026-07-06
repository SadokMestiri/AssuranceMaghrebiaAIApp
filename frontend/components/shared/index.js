const numberFormatter = new Intl.NumberFormat("fr-TN", { maximumFractionDigits: 2 });

export function formatShortCurrency(value) {
  const num = Number(value || 0);
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M DT`;
  if (num >= 1_000)     return `${Math.round(num / 1_000)}K DT`;
  return `${num} DT`;
}

export function asNumber(value, suffix = "") {
  return `${numberFormatter.format(Number(value || 0))}${suffix}`;
}

function computeVariation(current, previous) {
  const curr = Number(current ?? 0);
  const prev = Number(previous ?? 0);
  if (previous == null || previous === undefined || prev === 0) return null;
  return ((curr - prev) / Math.abs(prev)) * 100;
}

export function VariationBadge({ current, previous, invertColor = false }) {
  const pct = computeVariation(current, previous);
  if (pct === null) return null;

  const isZero     = Math.abs(pct) < 0.05;
  const isPositive = pct > 0;
  const isGood     = invertColor ? !isPositive : isPositive;

  const color  = isZero ? "#64748b" : isGood ? "#16a34a" : "#dc2626";
  const bg     = isZero ? "rgba(100,116,139,0.10)" : isGood ? "rgba(22,163,74,0.10)" : "rgba(220,38,38,0.10)";
  const arrow  = isZero ? "—" : isPositive ? "▲" : "▼";

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
