"""
Maghrebia â€” Service PrÃ©visions Temporelles (Prophet / Holt-Winters)
AdaptÃ© Ã  partir du projet InsureDecide.
"""

import os
import logging
import psycopg2
import pandas as pd
import numpy as np
import warnings
from decimal import Decimal

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://maghrebia:maghrebia@postgres:5432/maghrebia")

# Map virtual/computed indicateur names to actual kpis_mensuels columns (or None if computed)
_INDICATEUR_ALIAS = {
    "taux_resiliation":     "taux_resiliation_pct",   # stored as taux_resiliation_pct
    "sp_ratio":             "ratio_combine_pct",        # closest stored proxy
    "impayes_tnd":          "cout_sinistres_tnd",       # fallback proxy if not stored
}

# Canonical list of valid kpis_mensuels columns
_VALID_KPIS_COLS = {
    "ratio_combine_pct", "primes_acquises_tnd", "cout_sinistres_tnd",
    "nb_sinistres", "taux_resiliation_pct", "provision_totale_tnd",
    "nb_suspicions_fraude",
}

def _resolve_indicateur(indicateur: str) -> str:
    """Resolve an indicateur name to an actual DB column, applying aliases."""
    if indicateur in _VALID_KPIS_COLS:
        return indicateur
    if indicateur in _INDICATEUR_ALIAS:
        return _INDICATEUR_ALIAS[indicateur]
    # Default fallback
    return "primes_acquises_tnd"


# Bounds for each indicateur (for clipping forecasts)
BOUNDS = {
    "ratio_combine_pct":    (0.0,   200.0),
    "primes_acquises_tnd":  (10_000, None),
    "cout_sinistres_tnd":   (0,      None),
    "nb_sinistres":         (0,      None),
}

def _clean(val):
    if isinstance(val, Decimal): return float(val)
    return float(val) if val else 0.0

def _clip_yhat(values: np.ndarray, indicateur: str) -> np.ndarray:
    lo, hi = BOUNDS.get(indicateur, (0, None))
    result = np.maximum(values, lo) if lo is not None else values
    result = np.minimum(result, hi) if hi is not None else result
    return result

def _forecast_holtwinters(df: pd.DataFrame, nb_mois: int, indicateur: str):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        warnings.filterwarnings("ignore")

        y = df["y"].values
        n = len(y)
        if n < 12: raise ValueError(f"DonnÃ©es insuffisantes : {n} mois (min 12)")

        model = ExponentialSmoothing(y, seasonal_periods=min(12, n), trend="add", seasonal=None, damped_trend=True)
        result = model.fit(optimized=True, remove_bias=True)

        yhat_future = np.array(result.forecast(nb_mois))
        yhat_hist   = np.array(result.fittedvalues)
        resid       = y - yhat_hist
        std_recent = np.std(resid[-12:]) if n >= 12 else np.std(resid)

        last_date = df["ds"].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=nb_mois, freq="MS")

        hist_ci = 1.96 * std_recent
        future_ci = np.array([1.96 * std_recent * np.sqrt(1 + i * 0.15) for i in range(1, nb_mois + 1)])

        all_ds = pd.concat([df["ds"], pd.Series(future_dates)], ignore_index=True)
        all_yhat = np.concatenate([yhat_hist, yhat_future])
        all_lower = np.concatenate([yhat_hist - hist_ci, yhat_future - future_ci])
        all_upper = np.concatenate([yhat_hist + hist_ci, yhat_future + future_ci])

        fc = pd.DataFrame({
            "ds": all_ds,
            "yhat": _clip_yhat(all_yhat, indicateur),
            "yhat_lower": all_lower,
            "yhat_upper": all_upper,
        })
        return fc, "Holt-Winters"
    except Exception as e:
        logger.warning(f"HW failed: {e}")
        return _forecast_poly(df, nb_mois, indicateur)

def _forecast_poly(df: pd.DataFrame, nb_mois: int, indicateur: str):
    from sklearn.linear_model import Ridge
    warnings.filterwarnings("ignore")
    
    df_fit = df.tail(24).reset_index(drop=True).copy()
    n = len(df_fit)
    df_fit["t"] = np.arange(n, dtype=float)
    X = df_fit[["t"]].values
    y = df_fit["y"].values

    reg = Ridge(alpha=1.0).fit(X, y)
    resid = y - reg.predict(X)
    std_resid = np.std(resid)

    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=nb_mois, freq="MS")

    t_fut = np.arange(n, n + nb_mois, dtype=float)
    X_fut = t_fut.reshape(-1, 1)

    yhat_future = reg.predict(X_fut)
    yhat_hist = reg.predict(X)

    future_ci = np.array([1.96 * std_resid * np.sqrt(1 + i * 0.15) for i in range(1, nb_mois + 1)])

    n_full = len(df)
    n_ancien = n_full - n

    all_ds = pd.concat([df["ds"], pd.Series(future_dates)], ignore_index=True)
    all_yhat = np.concatenate([df["y"].values[:n_ancien], yhat_hist, yhat_future])
    all_lower = np.concatenate([df["y"].values[:n_ancien], yhat_hist - 1.96 * std_resid, yhat_future - future_ci])
    all_upper = np.concatenate([df["y"].values[:n_ancien], yhat_hist + 1.96 * std_resid, yhat_future + future_ci])

    fc = pd.DataFrame({
        "ds": all_ds,
        "yhat": _clip_yhat(all_yhat, indicateur),
        "yhat_lower": all_lower,
        "yhat_upper": all_upper,
    })
    return fc, "Regression Poly"

def get_forecast(departement: str = "AUTO", indicateur: str = "primes_acquises_tnd", nb_mois: int = 6) -> dict:
    db_col = _resolve_indicateur(indicateur)
    conn = psycopg2.connect(DATABASE_URL.replace("+psycopg2", ""))
    cur = conn.cursor()
    cur.execute(f"SELECT annee, mois, {db_col} FROM kpis_mensuels WHERE departement = %s ORDER BY annee, mois", [departement])
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < 12: return {"error": "DonnÃ©es insuffisantes"}

    df = pd.DataFrame(rows, columns=["annee", "mois", "y"])
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df["mois"] = pd.to_numeric(df["mois"], errors="coerce")
    # Filter corrupted periods (ex: year 9999) that crash pandas datetime conversion.
    df = df[
        df["annee"].between(1900, 2261)
        & df["mois"].between(1, 12)
    ].copy()
    if len(df) < 12:
        return {"error": "DonnÃ©es insuffisantes aprÃ¨s nettoyage des pÃ©riodes"}

    df["ds"] = pd.to_datetime(
        {
            "year": df["annee"].astype(int),
            "month": df["mois"].astype(int),
            "day": 1,
        },
        errors="coerce",
    )
    df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if len(df) < 12:
        return {"error": "DonnÃ©es insuffisantes aprÃ¨s conversion des dates"}

    df["y"] = df["y"].apply(_clean)

    forecast, methode = _forecast_holtwinters(df, nb_mois, indicateur)
    if forecast is None:
        return {"error": "Ã‰chec de prÃ©vision"}

    historique = [{"periode": row.ds.strftime("%Y-%m"), "valeur": round(float(row.y), 2), "type": "reel"} for _, row in df.iterrows()]
    future_rows = forecast[forecast["ds"] > df["ds"].max()]
    previsions = [
        {"periode": row.ds.strftime("%Y-%m"), "valeur": round(float(row.yhat), 2), "valeur_min": round(float(row.yhat_lower), 1), "valeur_max": round(float(row.yhat_upper), 1), "type": "prevision"}
        for _, row in future_rows.iterrows()
    ]

    dernier_reel = float(df["y"].iloc[-1])
    premiere_prev = previsions[0]["valeur"] if previsions else dernier_reel

    return {
        "departement": departement,
        "indicateur": indicateur,           # original name requested
        "db_col": db_col,                   # actual column used
        "methode": methode,
        "historique": historique[-24:],
        "previsions": previsions,
        "derniere_valeur": round(dernier_reel, 2),
        "prochaine_valeur": round(premiere_prev, 2)
    }

def get_all_forecasts(nb_mois: int = 6) -> dict:
    departements = ["AUTO", "IRDS", "SANTE"]
    indicateurs = ["primes_acquises_tnd", "cout_sinistres_tnd", "nb_sinistres"]
    results = {}
    for dept in departements:
        results[dept] = {}
        for ind in indicateurs:
            try:
                results[dept][ind] = get_forecast(dept, ind, nb_mois)
            except Exception as e:
                results[dept][ind] = {"error": str(e)}
    return results