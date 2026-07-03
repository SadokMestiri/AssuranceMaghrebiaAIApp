"""
ml_services/prophet_service.py
────────────────────────────────
Forecast service — loads pre-trained models from forecast_model.pkl.

Best model for PRIMES_ACQUISES: XGBoost (MAPE 33.3% on held-out test set).
Other KPIs: Prophet (multi-KPI loop from the notebook).

Falls back to Holt-Winters if the pkl is unavailable.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PKL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "forecast_model.pkl"
)

# Frontend indicateur name → pkl KPI column
_IND_MAP: dict[str, str] = {
    "primes_acquises_tnd": "PRIMES_ACQUISES",
    "cout_sinistres_tnd":  "COUT_SINISTRES",
    "nb_sinistres":        "NB_SINISTRES",
    "taux_resiliation":    "TAUX_RESILIATION",
    "sp_ratio":            "SP_RATIO",
    "impayes_tnd":         "MT_IMPAYE",
}

_art_cache: dict | None = None


def _load_artifact() -> dict:
    global _art_cache
    if _art_cache is None:
        import joblib
        _art_cache = joblib.load(_PKL_PATH)
    return _art_cache


_BRANCHES = {"AUTO", "IRDS", "SANTE"}


def _resolve_scope(art: dict, departement: str) -> dict:
    """Return the branch-level sub-artifact if available, else company-wide."""
    if departement in _BRANCHES:
        branch_data = art.get("per_branch", {}).get(departement)
        if branch_data:
            return branch_data
    return art


def _hist_series(scope: dict, kpi_col: str) -> pd.Series:
    raw = scope.get("kpi_historical", {}).get(kpi_col, {})
    if not raw:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(k): v for k, v in raw.items()})
    return s.sort_index()


def _fc_series_and_ci(scope: dict, kpi_col: str, art: dict) -> tuple[pd.Series, pd.DataFrame | None]:
    """Return (forecast_series, ci_df_or_None) for the given KPI and scope."""
    if kpi_col == "PRIMES_ACQUISES":
        fc = scope.get("xgb_forecast")
        if fc is None:
            fc = art.get("xgb_forecast") or art.get("primary_forecast")
        ci = art.get("primary_forecast_ci")
        return fc, ci

    all_fc = scope.get("all_forecasts", {})
    fc_df = all_fc.get(kpi_col)
    if fc_df is None:
        # Fall back to company-wide forecast for this KPI
        fc_df = art.get("all_forecasts", {}).get(kpi_col)
    if fc_df is None:
        return pd.Series(dtype=float), None
    return fc_df["yhat"], fc_df[["yhat_lower", "yhat_upper"]]


def _mape_for(art: dict, kpi_col: str) -> float | None:
    # PRIMES_ACQUISES: show Ensemble MAPE (Prophet+SARIMA+XGBoost, 17%) — more representative
    if kpi_col == "PRIMES_ACQUISES":
        mape_dict = art.get("metrics", {}).get("MAPE", {})
        return mape_dict.get("Ensemble") or mape_dict.get("XGBoost")
    # All other KPIs: Prophet holdout MAPE stored by the notebook
    kpi_mapes = art.get("kpi_mapes", {})
    val = kpi_mapes.get(kpi_col)
    if val is None:
        return None
    # Suppress absurdly high MAPEs (sparse KPIs like MT_IMPAYE) — treat as N/D
    return float(val) if float(val) < 500 else None


# ── Public API ────────────────────────────────────────────────────────────────

def get_forecast(
    departement: str = "AUTO",
    indicateur:  str = "primes_acquises_tnd",
    nb_mois:     int = 6,
) -> dict[str, Any]:
    try:
        art = _load_artifact()
    except Exception as exc:
        logger.warning("forecast_model.pkl unavailable (%s) — falling back to Holt-Winters", exc)
        return _fallback_forecast(departement, indicateur, nb_mois)

    kpi_col = _IND_MAP.get(indicateur, "PRIMES_ACQUISES")
    methode = "XGBoost" if kpi_col == "PRIMES_ACQUISES" else "Prophet"

    # Resolve scope: branch-specific data if available, else company-wide
    scope = _resolve_scope(art, departement)

    # Historical — all available months for this branch
    hist = _hist_series(scope, kpi_col)
    historique = [
        {"periode": str(idx)[:7], "valeur": round(float(v), 2), "type": "reel"}
        for idx, v in hist.items()
        if float(v) > 0
    ]

    # Forecast
    fc, ci = _fc_series_and_ci(scope, kpi_col, art)
    if fc is None or len(fc) == 0:
        return {"error": f"No forecast available for {indicateur}"}

    # Start forecast strictly after the last historical data point for this KPI
    # (dynamic — avoids hardcoded dates breaking when data is updated)
    if historique:
        last_hist_date = pd.Timestamp(historique[-1]["periode"] + "-01")
        min_fc_date = last_hist_date + pd.DateOffset(months=1)
        if fc.index.min() < min_fc_date:
            fc = fc[fc.index >= min_fc_date]

    fc = fc.iloc[:nb_mois]

    previsions = []
    for idx, v in fc.items():
        # Clip to 0 — KPI values are never negative in practice
        v_clipped = max(0.0, float(v))
        entry: dict[str, Any] = {
            "periode": str(idx)[:7],
            "valeur":  round(v_clipped, 2),
            "type":    "prevision",
        }
        if ci is not None:
            try:
                if isinstance(ci, pd.DataFrame):
                    row = ci.loc[idx] if idx in ci.index else None
                    if row is not None:
                        entry["valeur_min"] = round(max(0.0, float(row.iloc[0])), 1)
                        entry["valeur_max"] = round(max(0.0, float(row.iloc[1])), 1)
            except Exception:
                pass
        previsions.append(entry)

    derniere_valeur  = historique[-1]["valeur"] if historique else 0.0
    prochaine_valeur = previsions[0]["valeur"]  if previsions  else derniere_valeur
    mape             = _mape_for(art, kpi_col)

    result: dict[str, Any] = {
        "departement":      departement,
        "indicateur":       indicateur,
        "methode":          methode,
        "historique":       historique,
        "previsions":       previsions,
        "derniere_valeur":  round(derniere_valeur,  2),
        "prochaine_valeur": round(prochaine_valeur, 2),
    }
    if mape is not None:
        result["mape"] = round(float(mape), 1)
    return result


def get_all_forecasts(nb_mois: int = 6) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for dept in ("AUTO", "IRDS", "SANTE"):
        results[dept] = {}
        for ind in _IND_MAP:
            try:
                results[dept][ind] = get_forecast(dept, ind, nb_mois)
            except Exception as exc:
                results[dept][ind] = {"error": str(exc)}
    return results


# ── Holt-Winters fallback (no pkl) ────────────────────────────────────────────

def _fallback_forecast(departement: str, indicateur: str, nb_mois: int) -> dict[str, Any]:
    """Minimal fallback using Holt-Winters when the pkl is absent."""
    import psycopg2
    from decimal import Decimal

    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://maghrebia:maghrebia@postgres:5432/maghrebia"
    )
    _ALIAS = {
        "taux_resiliation": "taux_resiliation_pct",
        "sp_ratio":         "ratio_combine_pct",
        "impayes_tnd":      "cout_sinistres_tnd",
    }
    _VALID = {
        "ratio_combine_pct", "primes_acquises_tnd", "cout_sinistres_tnd",
        "nb_sinistres", "taux_resiliation_pct", "provision_totale_tnd",
    }
    db_col = _ALIAS.get(indicateur, indicateur if indicateur in _VALID else "primes_acquises_tnd")

    conn = psycopg2.connect(DATABASE_URL.replace("+psycopg2", ""))
    cur  = conn.cursor()
    cur.execute(
        f"SELECT annee, mois, {db_col} FROM kpis_mensuels WHERE departement=%s ORDER BY annee, mois",
        [departement],
    )
    rows = cur.fetchall(); cur.close(); conn.close()
    if len(rows) < 12:
        return {"error": "Données insuffisantes"}

    df = pd.DataFrame(rows, columns=["annee", "mois", "y"])
    df = df[pd.to_numeric(df["annee"], errors="coerce").between(1900, 2261)].copy()
    df["ds"] = pd.to_datetime({"year": df["annee"].astype(int), "month": df["mois"].astype(int), "day": 1}, errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    df["y"] = df["y"].apply(lambda v: float(v) if v else 0.0)

    warnings.filterwarnings("ignore")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        n = len(df)
        m = ExponentialSmoothing(df["y"].values, seasonal_periods=min(12, n), trend="add", seasonal=None, damped_trend=True)
        res = m.fit(optimized=True, remove_bias=True)
        yhat = np.array(res.forecast(nb_mois))
        std  = np.std(df["y"].values[-12:] - res.fittedvalues[-12:])
        methode = "Holt-Winters"
    except Exception:
        from sklearn.linear_model import Ridge
        n = len(df)
        X = np.arange(n).reshape(-1, 1).astype(float)
        reg = Ridge(alpha=1.0).fit(X, df["y"].values)
        yhat = reg.predict(np.arange(n, n + nb_mois).reshape(-1, 1).astype(float))
        std  = np.std(df["y"].values - reg.predict(X))
        methode = "Régression"

    last = df["ds"].max()
    future = pd.date_range(start=last + pd.DateOffset(months=1), periods=nb_mois, freq="MS")

    historique = [{"periode": r.ds.strftime("%Y-%m"), "valeur": round(float(r.y), 2), "type": "reel"} for _, r in df.tail(24).iterrows()]
    previsions = [
        {"periode": d.strftime("%Y-%m"), "valeur": round(float(v), 2),
         "valeur_min": round(float(v - 1.96 * std), 1), "valeur_max": round(float(v + 1.96 * std), 1), "type": "prevision"}
        for d, v in zip(future, yhat)
    ]
    return {
        "departement":      departement,
        "indicateur":       indicateur,
        "methode":          methode,
        "historique":       historique,
        "previsions":       previsions,
        "derniere_valeur":  historique[-1]["valeur"] if historique else 0,
        "prochaine_valeur": previsions[0]["valeur"]  if previsions  else 0,
    }
