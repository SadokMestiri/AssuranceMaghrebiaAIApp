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

export default function FiltersBar({ governorates, loading }) {
  const { filters, setFilter, resetFilters } = useFilters();

  return (
    <section className="panel filters-panel">
      <div className="panel-headline">
        <h2>Filtres analytiques</h2>
        <button type="button" className="ghost-btn" onClick={resetFilters}>
          Reinitialiser
        </button>
      </div>

      <div className="filters-grid">
        <label>
          Annee debut
          <input
            type="number"
            min={YEAR_MIN}
            max={YEAR_MAX}
            value={filters.yearFrom}
            onChange={(event) => setFilter("yearFrom", event.target.value)}
            disabled={loading}
          />
        </label>

        <label>
          Annee fin
          <input
            type="number"
            min={YEAR_MIN}
            max={YEAR_MAX}
            value={filters.yearTo}
            onChange={(event) => setFilter("yearTo", event.target.value)}
            disabled={loading}
          />
        </label>

        <label>
          Branche
          <select
            value={filters.branch}
            onChange={(event) => setFilter("branch", event.target.value)}
            disabled={loading}
          >
            {BRANCH_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label>
          Mois
          <select
            value={filters.month}
            onChange={(event) => setFilter("month", event.target.value)}
            disabled={loading}
          >
            {MONTH_OPTIONS.map((monthOption) => (
              <option key={monthOption.value} value={monthOption.value}>
                {monthOption.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          Gouvernorat
          <select
            value={filters.gouvernorat}
            onChange={(event) => setFilter("gouvernorat", event.target.value)}
            disabled={loading}
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
