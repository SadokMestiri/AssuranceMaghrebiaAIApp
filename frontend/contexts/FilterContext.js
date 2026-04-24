import { createContext, useContext, useMemo, useState } from "react";

const FilterContext = createContext(null);

export const YEAR_MIN = 2019;
export const YEAR_MAX = 2026;

const initialFilters = {
  yearFrom: YEAR_MIN,
  yearTo: YEAR_MAX,
  month: "ALL",
  branch: "ALL",
  gouvernorat: "ALL",
};

function clampYear(value, fallback) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return fallback;
  }

  const year = Math.trunc(numericValue);
  return Math.min(YEAR_MAX, Math.max(YEAR_MIN, year));
}

function normalizeYearRange(nextFilters) {
  const normalizedFrom = clampYear(nextFilters.yearFrom, initialFilters.yearFrom);
  const normalizedTo = clampYear(nextFilters.yearTo, initialFilters.yearTo);

  return {
    ...nextFilters,
    yearFrom: Math.min(normalizedFrom, normalizedTo),
    yearTo: Math.max(normalizedFrom, normalizedTo),
  };
}

export function FilterProvider({ children }) {
  const [filters, setFilters] = useState(initialFilters);

  const setFilter = (key, value) => {
    setFilters((prev) => normalizeYearRange({
      ...prev,
      [key]: value,
    }));
  };

  const resetFilters = () => {
    setFilters(initialFilters);
  };

  const value = useMemo(
    () => ({
      filters,
      setFilter,
      resetFilters,
    }),
    [filters]
  );

  return <FilterContext.Provider value={value}>{children}</FilterContext.Provider>;
}

export function useFilters() {
  const context = useContext(FilterContext);
  if (!context) {
    throw new Error("useFilters must be used inside FilterProvider");
  }
  return context;
}
