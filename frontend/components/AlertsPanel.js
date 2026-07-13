import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, TrendingDown, Cpu, Percent, UserMinus, UserPlus, ChevronLeft, ChevronRight } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const PAGE_SIZE = 10;

const TYPE_ICON = {
  impaye_rate: AlertTriangle,
  production_drop: TrendingDown,
  ml_readiness: Cpu,
  ratio_combine: Percent,
  resiliation: UserMinus,
  new_user: UserPlus,
};

const TYPE_LABELS = {
  impaye_rate: "Taux impayé",
  production_drop: "Baisse production",
  ml_readiness: "Readiness ML",
  ratio_combine: "Ratio combiné",
  resiliation: "Résiliation",
  new_user: "Nouvel utilisateur",
};

const SEVERITY_LABELS = { high: "Critique", medium: "Attention", info: "Info" };

export default function AlertsPanel({ branch }) {
  const [alerts, setAlerts]     = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState("");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [typeFilter, setTypeFilter]         = useState("all");
  const [page, setPage]         = useState(1);

  useEffect(() => {
    setLoading(true);
    setError("");
    const params = new URLSearchParams({ months: "12" });
    if (branch && branch !== "ALL") params.set("branch", branch);
    fetch(`${API_BASE}/api/v1/alerts?${params.toString()}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => setAlerts(data.alerts || []))
      .catch(() => setError("Impossible de charger les alertes."))
      .finally(() => setLoading(false));
  }, [branch]);

  const filtered = useMemo(() => {
    return alerts.filter((a) => {
      if (severityFilter !== "all" && a.severity !== severityFilter) return false;
      if (typeFilter !== "all" && a.type !== typeFilter) return false;
      return true;
    });
  }, [alerts, severityFilter, typeFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageItems = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const updateSeverity = (value) => { setSeverityFilter(value); setPage(1); };
  const updateType = (value) => { setTypeFilter(value); setPage(1); };

  const highCount = alerts.filter((a) => a.severity === "high").length;
  const mediumCount = alerts.filter((a) => a.severity === "medium").length;
  const infoCount = alerts.filter((a) => a.severity === "info").length;

  return (
    <section className="alerts-panel-section">
      <div className="panel-headline">
        <h2>Alertes</h2>
        <div className="alerts-summary-pills">
          <span className="alerts-pill critical">{highCount} critique{highCount !== 1 ? "s" : ""}</span>
          <span className="alerts-pill warning">{mediumCount} attention</span>
          <span className="alerts-pill info">{infoCount} info</span>
        </div>
      </div>

      <div className="alerts-filters">
        <label>
          <span>Statut</span>
          <select value={severityFilter} onChange={(e) => updateSeverity(e.target.value)}>
            <option value="all">Tous</option>
            <option value="high">Critique</option>
            <option value="medium">Attention</option>
            <option value="info">Info</option>
          </select>
        </label>
        <label>
          <span>Type</span>
          <select value={typeFilter} onChange={(e) => updateType(e.target.value)}>
            <option value="all">Tous</option>
            <option value="impaye_rate">Taux impayé</option>
            <option value="production_drop">Baisse production</option>
            <option value="ml_readiness">Readiness ML</option>
            <option value="ratio_combine">Ratio combiné</option>
            <option value="resiliation">Résiliation</option>
            <option value="new_user">Nouvel utilisateur</option>
          </select>
        </label>
      </div>

      {error && <p className="admin-error">{error}</p>}

      {loading ? (
        <p className="dim-loading">Chargement des alertes…</p>
      ) : filtered.length === 0 ? (
        <p className="notif-empty">Aucune alerte pour ce filtre.</p>
      ) : (
        <>
          <ul className="alerts-list">
            {pageItems.map((alert, i) => {
              const Icon = TYPE_ICON[alert.type] || AlertTriangle;
              return (
                <li key={`${alert.type}-${alert.period}-${i}`} className={`alerts-row severity-${alert.severity}`}>
                  <Icon size={18} />
                  <div className="alerts-row-body">
                    <p className="alerts-row-message">{alert.message}</p>
                    <p className="alerts-row-meta">
                      {TYPE_LABELS[alert.type] || alert.type} · {alert.period}
                    </p>
                  </div>
                  <span className={`alerts-severity-badge ${alert.severity}`}>
                    {SEVERITY_LABELS[alert.severity] || alert.severity}
                  </span>
                </li>
              );
            })}
          </ul>

          <div className="alerts-pagination">
            <button type="button" disabled={currentPage <= 1} onClick={() => setPage((p) => p - 1)}>
              <ChevronLeft size={15} /> Precedent
            </button>
            <span>Page {currentPage} / {totalPages}</span>
            <button type="button" disabled={currentPage >= totalPages} onClick={() => setPage((p) => p + 1)}>
              Suivant <ChevronRight size={15} />
            </button>
          </div>
        </>
      )}
    </section>
  );
}
