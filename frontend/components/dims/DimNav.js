import { LayoutDashboard, Users, Building2, Package, Car, FileText, AlertTriangle } from 'lucide-react';

const DIMS = [
  { key: "overview",  label: "Vue Globale",  Icon: LayoutDashboard },
  { key: "clients",   label: "Clients",       Icon: Users },
  { key: "agents",    label: "Agents",        Icon: Building2 },
  { key: "produits",  label: "Produits",      Icon: Package },
  { key: "vehicules", label: "Véhicules",     Icon: Car },
  { key: "polices",   label: "Polices",       Icon: FileText },
  { key: "sinistres", label: "Sinistres",     Icon: AlertTriangle },
];

export default function DimNav({ activeDim, onDimChange, allowedDims }) {
  const visibleDims = allowedDims ? DIMS.filter((dim) => allowedDims.includes(dim.key)) : DIMS;
  return (
    <nav className="dim-nav">
      <div className="dim-nav-track">
        {visibleDims.map((dim) => (
          <button
            key={dim.key}
            type="button"
            className={`dim-nav-tab ${activeDim === dim.key ? "active" : ""}`}
            onClick={() => onDimChange(dim.key)}
          >
            <span className="dim-tab-icon"><dim.Icon size={18} /></span>
            <span className="dim-tab-label">{dim.label}</span>
            {activeDim === dim.key && <span className="dim-tab-indicator" />}
          </button>
        ))}
      </div>
    </nav>
  );
}
