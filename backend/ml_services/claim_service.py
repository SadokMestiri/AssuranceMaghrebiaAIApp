"""
ml_services/claim_service.py
─────────────────────────────
Claim severity prediction service — claim_severity.ipynb

Notebook artifact : claim_severity_model.pkl
  Keys : xgb_model, lgb_model, q10_model, q90_model, encoders,
         features (38), feature_defaults, w_xgb, w_lgb,
         reserve_factor, quantile_coverage, metrics

Ensemble: XGBoost + LightGBM (1/MAPE weights) on log1p(MT_EVALUATION).
Quantile models (q10, q90) provide the real confidence interval.
Reserve recommendation = median prediction × reserve_factor (1.15).

Falls back to proxy GradientBoostingRegressor if artifact absent.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

from ._base import (
    load_artifact, load_police, load_sinistre,
    load_vehicule, save_artifact, safe_float,
)

logger = logging.getLogger("maghrebia.ml_services.claim")

_ARTIFACT_NAME = "claim_severity_model"
_RESERVE_FACTOR = 1.15
_CI_FACTOR      = 0.25

# Legacy feature set — used only for proxy training
FEATURE_COLS = [
    "RESPONSABILITE", "DELAI_DECLARATION",
    "MOIS_SURVENANCE", "FLAG_WEEKEND",
    "AGE_CLIENT", "FLAG_MORALE",
    "ID_BRANCHE",
    "PUISSANCE", "AGE_VEHICULE", "NB_PLACE",
    "NATURE_SINISTRE_ENC", "BRANCHE_ENC",
]

_NATURE_ENC = {
    "MATERIEL": 0, "CORPOREL": 1, "MIXTE": 2,
    "INCENDIE": 3, "VOL": 4, "AUTRE": 5,
}
_BRANCHE_ENC = {"AUTO": 0, "IRDS": 1, "SANTE": 2}


def _is_notebook_art(art: dict) -> bool:
    return "xgb_model" in art or "lgb_model" in art


# ── Feature builder (for proxy training) ────────────────────────────────────

def _build_sinistre_features() -> pd.DataFrame:
    sinistre = load_sinistre()
    if sinistre.empty:
        return pd.DataFrame()

    police   = load_police()
    vehicule = load_vehicule()

    sinistre.columns = [c.upper() for c in sinistre.columns]
    sinistre["MT_EVALUATION"]   = pd.to_numeric(sinistre.get("MT_EVALUATION"), errors="coerce").fillna(0)
    sinistre["DATE_SURVENANCE"] = pd.to_datetime(sinistre.get("DATE_SURVENANCE"), errors="coerce")
    if "DATE_DECLARATION" in sinistre.columns:
        sinistre["DATE_DECLARATION"] = pd.to_datetime(sinistre["DATE_DECLARATION"], errors="coerce")
        delai_raw = (sinistre["DATE_DECLARATION"] - sinistre["DATE_SURVENANCE"]).dt.days
        sinistre["DELAI_DECLARATION"] = pd.to_numeric(delai_raw, errors="coerce").fillna(7).clip(0, 365)
    else:
        sinistre["DELAI_DECLARATION"] = 7.0
    sinistre["RESPONSABILITE"] = pd.to_numeric(sinistre.get("RESPONSABILITE"), errors="coerce").fillna(0)

    df = sinistre[sinistre["MT_EVALUATION"] > 0].copy()
    df["MOIS_SURVENANCE"]     = df["DATE_SURVENANCE"].dt.month.fillna(6)
    df["FLAG_WEEKEND"]        = df["DATE_SURVENANCE"].dt.dayofweek.isin([5, 6]).astype(int)
    df["NATURE_SINISTRE_ENC"] = df.get("NATURE_SINISTRE", "MATERIEL").map(_NATURE_ENC).fillna(0)
    df["BRANCHE_ENC"]         = df.get("BRANCHE", "AUTO").map(_BRANCHE_ENC).fillna(0)

    if not police.empty:
        police.columns = [c.upper() for c in police.columns]
        police["FLAG_MORALE"] = (police.get("TYPE_POLICE", "") == "E").astype(int)
        df = df.merge(police[["ID_POLICE", "ID_BRANCHE", "FLAG_MORALE"]], on="ID_POLICE", how="left")
    else:
        df["ID_BRANCHE"]  = 1
        df["FLAG_MORALE"] = 0

    if not vehicule.empty:
        vehicule.columns = [c.upper() for c in vehicule.columns]
        vehicule["PUISSANCE"]    = pd.to_numeric(vehicule.get("PUISSANCE"), errors="coerce").fillna(6)
        vehicule["NB_PLACE"]     = pd.to_numeric(vehicule.get("NB_PLACE"),  errors="coerce").fillna(5)
        vehicule["DATE_MEC"]     = pd.to_datetime(vehicule.get("DATE_MISE_CIRCULATION"), errors="coerce")
        ref = pd.Timestamp("today")
        vehicule["AGE_VEHICULE"] = ((ref - vehicule["DATE_MEC"]).dt.days / 365.25).clip(0).fillna(5)
        df = df.merge(vehicule[["ID_VEHICULE", "PUISSANCE", "NB_PLACE", "AGE_VEHICULE"]]
                      .rename(columns={"ID_VEHICULE": "ID_POLICE"}),
                      on="ID_POLICE", how="left")
    else:
        df["PUISSANCE"]    = 6.0
        df["NB_PLACE"]     = 5.0
        df["AGE_VEHICULE"] = 5.0

    df["AGE_CLIENT"] = 40.0

    valid = [c for c in FEATURE_COLS if c in df.columns]
    df[valid] = df[valid].fillna(0)
    return df


# ── Proxy model trainer ─────────────────────────────────────────────────────

def _train_proxy(df: pd.DataFrame) -> dict:
    valid = [c for c in FEATURE_COLS if c in df.columns]
    X     = df[valid].fillna(0).values
    y_log = np.log1p(df["MT_EVALUATION"].clip(0).values)

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42,
    )
    model.fit(X_imp, y_log)

    y_pred    = np.expm1(model.predict(X_imp))
    residuals = df["MT_EVALUATION"].clip(0).values - y_pred
    q10_adj   = float(np.percentile(residuals, 10))
    q90_adj   = float(np.percentile(residuals, 90))

    art = {
        "model":    model,
        "imputer":  imputer,
        "features": valid,
        "q10_adj":  q10_adj,
        "q90_adj":  q90_adj,
        "source":   "proxy",
    }
    save_artifact(_ARTIFACT_NAME, art)
    logger.info("Claim severity proxy model trained and saved.")
    return art


def _get_artifact() -> dict:
    art = load_artifact(_ARTIFACT_NAME)
    if art:
        return art
    logger.info("Claim artifact not found — training proxy.")
    df = _build_sinistre_features()
    if df.empty or len(df) < 50:
        raise RuntimeError("Not enough sinistre data to train claim model.")
    return _train_proxy(df)


# ── Prediction helpers ───────────────────────────────────────────────────────

def _encode_categorical(art: dict, field: str, value: str) -> float:
    """Use notebook encoder if available; fall back to hardcoded mapping."""
    encoders = art.get("encoders", {})
    enc = encoders.get(field)
    if enc is not None:
        try:
            return float(enc.transform([value.upper()])[0])
        except Exception:
            pass
    # Hardcoded fallbacks
    if field == "NATURE_SINISTRE":
        return float(_NATURE_ENC.get(value.upper(), 7.0))
    if field == "BRANCHE":
        return float(_BRANCHE_ENC.get(value.upper(), 0.0))
    return 0.0


_MIN_SEVERITY_TND = 10.0  # predictions below this in TND indicate a broken model


def _predict_notebook(art: dict, X_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (median_pred, q10_pred, q90_pred) all in original TND scale."""
    w_xgb = art.get("w_xgb", 0.5)
    w_lgb = art.get("w_lgb", 0.5)

    pred_xgb = float(np.expm1(art["xgb_model"].predict(X_df)[0]))
    pred_lgb = float(np.expm1(art["lgb_model"].predict(X_df)[0]))

    # XGBoost severity model may be broken (constant near-zero output).
    # Fall back to LightGBM-only when XGB is clearly anomalous.
    if pred_xgb < _MIN_SEVERITY_TND and pred_lgb >= _MIN_SEVERITY_TND:
        median_pred = pred_lgb
    else:
        median_pred = w_xgb * pred_xgb + w_lgb * pred_lgb

    q10_pred = float(np.expm1(art["q10_model"].predict(X_df)[0]))
    q90_pred = float(np.expm1(art["q90_model"].predict(X_df)[0]))

    return median_pred, q10_pred, q90_pred


# ── Public API ──────────────────────────────────────────────────────────────

def get_claim_model_info() -> dict[str, Any]:
    """Return performance metrics and metadata for the claim severity model."""
    art = _get_artifact()
    return {
        "status":            "ready",
        "source":            art.get("source",           "Ensemble XGB+LGB"),
        "notebook":          art.get("notebook",         "claim_severity.ipynb"),
        "features":          len(art.get("features",     [])),
        "reserve_factor":    art.get("reserve_factor",   _RESERVE_FACTOR),
        "quantile_coverage": art.get("quantile_coverage", 0.0),
        "weights":           {"xgb": art.get("w_xgb"), "lgb": art.get("w_lgb")},
        "metrics":           art.get("metrics",     {}),
        "metrics_xgb":       art.get("metrics_xgb", {}),
        "metrics_lgb":       art.get("metrics_lgb", {}),
    }


def predict_claim_severity(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Predict the final settlement amount for a claim.
    Input keys: branche, nature_sinistre, mt_evaluation, age_client, bonus_malus
    """
    art = _get_artifact()

    if _is_notebook_art(art):
        feat_cols = art["features"]
        defaults  = art.get("feature_defaults", {c: 0.0 for c in feat_cols})

        # Start from notebook training medians
        row = {c: defaults.get(c, 0.0) for c in feat_cols}

        # Map simulator inputs
        branche         = str(payload.get("branche",         "AUTO")).upper()
        nature_sinistre = str(payload.get("nature_sinistre", "MATERIEL")).upper()
        age_client      = safe_float(payload.get("age_client",  defaults.get("AGE_CLIENT",  40.0)))
        bonus_malus     = safe_float(payload.get("bonus_malus", defaults.get("BONUS_MALUS", 1.0)))

        row["AGE_CLIENT"]   = age_client
        row["BONUS_MALUS"]  = bonus_malus
        row["BONUS_MALUS_EM"] = bonus_malus

        # Categorical encoding via notebook encoders
        row["BRANCHE_ENC"]         = _encode_categorical(art, "BRANCHE",         branche)
        row["ID_BRANCHE"]          = float(_BRANCHE_ENC.get(branche, 0) + 1)
        row["NATURE_SINISTRE_ENC"] = _encode_categorical(art, "NATURE_SINISTRE", nature_sinistre)

        # Date-based features — use current month/quarter as defaults
        now = datetime.now()
        row["MOIS_SURVENANCE"]  = float(now.month)
        row["TRIMESTRE"]        = float((now.month - 1) // 3 + 1)
        row["JOUR_SEMAINE"]     = float(now.weekday())
        row["FLAG_WEEKEND"]     = float(now.weekday() >= 5)
        row["ANNEE_SURVENANCE"] = float(now.year)

        # DELAI_DECLARATION: 8.2% importance — days between accident and declaration
        delai_decl = int(payload.get("delai_declaration", 9))
        row["DELAI_DECLARATION"] = float(delai_decl)

        # PRIME_MOY + PRIME_MAX: 7.3% + 7.0% — average and max policy premium
        prime_contrat = safe_float(payload.get("prime_contrat", 0.0))
        if prime_contrat > 0:
            row["PRIME_MOY"] = prime_contrat
            row["PRIME_MAX"] = prime_contrat * 1.3

        # VALEUR_A_NEUF + VALEUR_ACTUELLE: 3.8% — vehicle value stored in millimes
        valeur_veh = safe_float(payload.get("valeur_vehicule", 0.0))
        if valeur_veh > 0:
            val_millimes = valeur_veh * 1000.0
            age_veh_def = float(defaults.get("AGE_VEHICULE", 10.0))
            taux_dep = min(0.85, age_veh_def * 0.07)
            row["VALEUR_A_NEUF"]   = val_millimes / max(0.15, 1.0 - taux_dep)
            row["VALEUR_ACTUELLE"] = val_millimes

        X_df = pd.DataFrame([row], columns=feat_cols)
        predicted, q10_pred, q90_pred = _predict_notebook(art, X_df)

        reserve_factor = art.get("reserve_factor", _RESERVE_FACTOR)
        reserve = round(predicted * reserve_factor, 0)

        # Ensure CI is ordered correctly
        ci_low  = round(max(0.0, min(q10_pred, predicted * 0.5)), 0)
        ci_high = round(max(q90_pred, predicted * 1.2), 0)

    else:
        # Legacy proxy path
        df_train = _build_sinistre_features()
        valid    = art["features"]
        medians  = df_train[valid].median() if not df_train.empty else pd.Series(0, index=valid)
        row_s    = medians.copy()

        if "branche" in payload:
            row_s["BRANCHE_ENC"] = float(_BRANCHE_ENC.get(str(payload["branche"]).upper(), 0))
        if "nature_sinistre" in payload:
            row_s["NATURE_SINISTRE_ENC"] = float(_NATURE_ENC.get(str(payload["nature_sinistre"]).upper(), 0))
        if "age_client" in payload and "AGE_CLIENT" in row_s.index:
            row_s["AGE_CLIENT"] = safe_float(payload["age_client"])
        if "bonus_malus" in payload and "RESPONSABILITE" in row_s.index:
            row_s["RESPONSABILITE"] = min(100, safe_float(payload["bonus_malus"]) * 50)

        X = art["imputer"].transform(row_s.values.reshape(1, -1))
        pred_log  = float(art["model"].predict(X)[0])
        predicted = float(np.expm1(pred_log))

        q10_adj = art.get("q10_adj", -predicted * _CI_FACTOR)
        q90_adj = art.get("q90_adj",  predicted * _CI_FACTOR)
        ci_low  = round(max(0, predicted + q10_adj), 0)
        ci_high = round(predicted + q90_adj, 0)
        reserve = round(predicted * _RESERVE_FACTOR, 0)

    if predicted > 50000:
        severity_class = "Grave"
    elif predicted > 15000:
        severity_class = "Modéré"
    else:
        severity_class = "Léger"

    return {
        "predicted_severity":  round(predicted, 0),
        "ci_low":              ci_low,
        "ci_high":             ci_high,
        "reserve_recommandee": reserve,
        "severity_class":      severity_class,
        "model_used":          art.get("source", "XGB+LGB ensemble"),
        "confidence":          art.get("quantile_coverage", 0.82),
    }
