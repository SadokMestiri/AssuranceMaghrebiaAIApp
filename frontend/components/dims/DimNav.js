const DIMS = [
  { key: "overview",  label: "Vue Globale",  icon: "⬡" },
  { key: "clients",   label: "Clients",       icon: "👥" },
  { key: "agents",    label: "Agents",        icon: "🏢" },
  { key: "produits",  label: "Produits",      icon: "📦" },
  { key: "vehicules", label: "Véhicules",     icon: "🚗" },
  { key: "polices",   label: "Polices",       icon: "📋" },
  { key: "sinistres", label: "Sinistres",     icon: "⚠️" },
];

export default function DimNav({ activeDim, onDimChange }) {
  return (
    <nav className="dim-nav">
      <div className="dim-nav-track">
        {DIMS.map((dim) => (
          <button
            key={dim.key}
            type="button"
            className={`dim-nav-tab ${activeDim === dim.key ? "active" : ""}`}
            onClick={() => onDimChange(dim.key)}
          >
            <span className="dim-tab-icon">{dim.icon}</span>
            <span className="dim-tab-label">{dim.label}</span>
            {activeDim === dim.key && <span className="dim-tab-indicator" />}
          </button>
        ))}
      </div>
    </nav>
  );
}
