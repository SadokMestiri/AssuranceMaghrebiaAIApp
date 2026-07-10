import { FileDown, Loader2 } from "lucide-react";
import { useFilters, YEAR_MAX, YEAR_MIN } from "../contexts/FilterContext";

const BRANCH_OPTIONS = ["ALL", "AUTO", "IRDS", "SANTE"];
const MONTH_OPTIONS = [
  { value: "ALL", label: "Tous" },
  { value: "1", label: "Jan" },
  { value: "2", label: "Fev" },
  { value: "3", label: "Mar" },
  { value: "4", label: "Avr" },
  { value: "5", label: "Mai" },
  { value: "6", label: "Jun" },
  { value: "7", label: "Jul" },
  { value: "8", label: "Aou" },
  { value: "9", label: "Sep" },
  { value: "10", label: "Oct" },
  { value: "11", label: "Nov" },
  { value: "12", label: "Dec" },
];

function FilterLabel({ children, support }) {
  const na = support === false;
  const partial = support === "partial";
  const autoOnly = support === "auto";

  let hint = null;
  if (na)       hint = "non applicable à cette vue";
  if (partial)  hint = "filtrage partiel côté client";
  if (autoOnly) hint = "branche AUTO uniquement";

  return (
    <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
      <span style={{ opacity: na ? 0.45 : 1 }}>{children}</span>
      {hint && (
        <span
          title={hint}
          style={{
            fontSize: "10px",
            fontWeight: 600,
            color: na ? "#94a3b8" : "#d97706",
            background: na ? "rgba(148,163,184,0.12)" : "rgba(217,119,6,0.1)",
            borderRadius: "3px",
            padding: "1px 5px",
            cursor: "default",
            whiteSpace: "nowrap",
          }}
        >
          {na ? "—" : autoOnly ? "AUTO" : "~"}
        </span>
      )}
    </span>
  );
}

export default function FiltersBar({ governorates, loading, filterSupport = {}, onExportPdf, exportingPdf = false }) {
  const { filters, setFilter, resetFilters } = useFilters();

  const branchSupport = filterSupport.branch ?? true;
  const yearSupport   = filterSupport.year   ?? true;
  const monthSupport  = filterSupport.month  ?? true;
  const govSupport    = filterSupport.gouvernorat ?? true;

  return (
    <section className="panel filters-panel">
      <div className="panel-headline">
        <h2>Filtres analytiques</h2>
        <div className="panel-headline-actions">
          {onExportPdf && (
            <button
              type="button"
              className="ghost-btn"
              onClick={onExportPdf}
              disabled={exportingPdf}
              title="Exporter un snapshot PDF de ce tableau de bord"
            >
              {exportingPdf ? <Loader2 size={14} className="animate-spin" /> : <FileDown size={14} />}
              {exportingPdf ? "Export en cours..." : "Exporter PDF"}
            </button>
          )}
          <button type="button" className="ghost-btn" onClick={resetFilters}>
            Reinitialiser
          </button>
        </div>
      </div>

      <div className="filters-grid">
        <label style={{ opacity: yearSupport === false ? 0.5 : 1 }}>
          <FilterLabel support={yearSupport}>Annee debut</FilterLabel>
          <input
            type="number"
            min={YEAR_MIN}
            max={YEAR_MAX}
            value={filters.yearFrom}
            onChange={(event) => setFilter("yearFrom", event.target.value)}
            disabled={loading || yearSupport === false}
          />
        </label>

        <label style={{ opacity: yearSupport === false ? 0.5 : 1 }}>
          <FilterLabel support={yearSupport}>Annee fin</FilterLabel>
          <input
            type="number"
            min={YEAR_MIN}
            max={YEAR_MAX}
            value={filters.yearTo}
            onChange={(event) => setFilter("yearTo", event.target.value)}
            disabled={loading || yearSupport === false}
          />
        </label>

        <label>
          <FilterLabel support={branchSupport}>Branche</FilterLabel>
          <select
            value={filters.branch}
            onChange={(event) => setFilter("branch", event.target.value)}
            disabled={loading}
          >
            {BRANCH_OPTIONS.map((option) => (
              <option
                key={option}
                value={option}
                disabled={branchSupport === "auto" && option !== "ALL" && option !== "AUTO"}
              >
                {option}
              </option>
            ))}
          </select>
        </label>

        <label style={{ opacity: monthSupport === false ? 0.5 : 1 }}>
          <FilterLabel support={monthSupport}>Mois</FilterLabel>
          <select
            value={filters.month}
            onChange={(event) => setFilter("month", event.target.value)}
            disabled={loading || monthSupport === false}
          >
            {MONTH_OPTIONS.map((monthOption) => (
              <option key={monthOption.value} value={monthOption.value}>
                {monthOption.label}
              </option>
            ))}
          </select>
        </label>

        <label style={{ opacity: govSupport === false ? 0.5 : 1 }}>
          <FilterLabel support={govSupport}>Gouvernorat</FilterLabel>
          <select
            value={filters.gouvernorat}
            onChange={(event) => setFilter("gouvernorat", event.target.value)}
            disabled={loading || govSupport === false}
          >
            <option value="ALL">Tous</option>
            {governorates.map((gov) => (
              <option key={gov} value={gov}>
                {gov}
              </option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}
