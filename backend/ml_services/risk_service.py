"""
ml_services/risk_service.py
────────────────────────────
Risk scoring & pricing service — risk_scoring_pricing.ipynb

Architecture (frequency–severity):
  PRED_FREQ  : LightGBM (Poisson objective) — predicts claims frequency
  PRED_SEV   : LightGBM (regression on log(severity)) — predicts mean claim cost
  PRIME_PURE = PRED_FREQ × PRED_SEV
  PRIME_TECHNIQUE = PRIME_PURE × (1 + frais 20% + marge 5% + FGA 3%)
  RISK_SCORE [0–1000] = weighted combination of normalised freq + sev + taux_impaye

If the notebook artifact is absent the service trains a proxy using
GradientBoostingRegressor on DB/CSV data.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ._base import (
    load_artifact, load_emission, load_impaye, load_police,
    load_sinistre, load_vehicule, save_artifact, safe_float, normalize_0_1000,
)

logger = logging.getLogger("maghrebia.ml_services.risk")

_ARTIFACT_NAME = "risk_model"

# Chargement total from notebook
_CHARGEMENT_TOTAL = 1.28   # 1 + 0.20 + 0.05 + 0.03

FEATURE_COLS = [
    "BONUS_MALUS", "NB_QUITTANCES", "EXPOSITION_ANS",
    "PUISSANCE", "AGE_VEHICULE", "NB_PLACE",
    "AGE_CLIENT", "FLAG_MORALE", "ID_BRANCHE",
    "TAUX_IMPAYE", "NB_IMPAYES",
    "PRIME_PAR_AN",
]

_RISK_CLASSES = [
    (0,   200, "A — Excellent"),
    (200, 400, "B — Bon"),
    (400, 600, "C — Modéré"),
    (600, 800, "D — Élevé"),
    (800, 1001,"E — Critique"),
]


def _risk_class(score: float) -> str:
    for lo, hi, label in _RISK_CLASSES:
        if lo <= score < hi:
            return label
    return "E — Critique"


# ── Feature builder ─────────────────────────────────────────────────────────

def _build_portfolio() -> pd.DataFrame:
    police   = load_police()
    emission = load_emission()
    sinistre = load_sinistre()
    impaye   = load_impaye()
    vehicule = load_vehicule()

    if police.empty:
        return pd.DataFrame()

    police.columns   = [c.upper() for c in police.columns]
    emission.columns = [c.upper() for c in emission.columns] if not emission.empty else emission.columns
    sinistre.columns = [c.upper() for c in sinistre.columns] if not sinistre.empty else sinistre.columns
    impaye.columns   = [c.upper() for c in impaye.columns]   if not impaye.empty   else impaye.columns
    vehicule.columns = [c.upper() for c in vehicule.columns] if not vehicule.empty else vehicule.columns

    ref = pd.Timestamp("today")
    police["DATE_EFFET"]    = pd.to_datetime(police.get("DATE_EFFET"),    errors="coerce")
    police["DATE_ECHEANCE"] = pd.to_datetime(police.get("DATE_ECHEANCE"), errors="coerce")
    police["BONUS_MALUS"]   = pd.to_numeric(police.get("BONUS_MALUS"),    errors="coerce").fillna(1.0)
    police["FLAG_MORALE"]   = (police.get("TYPE_POLICE", "") == "E").astype(int)
    police["EXPOSITION_ANS"]= ((police["DATE_ECHEANCE"] - police["DATE_EFFET"]).dt.days / 365.25).clip(0).fillna(1)

    df = police[["ID_POLICE","ID_CLIENT","ID_BRANCHE","BONUS_MALUS",
                 "FLAG_MORALE","DATE_EFFET","EXPOSITION_ANS"]].copy()

    # ── Emission agg ──
    if not emission.empty:
        emission["MT_PNET"] = pd.to_numeric(emission.get("MT_PNET"), errors="coerce").fillna(0)
        em_agg = emission.groupby("ID_POLICE").agg(
            NB_QUITTANCES=("NUM_QUITTANCE","count"),
            PRIME_PAR_AN =("MT_PNET",      "mean"),
        ).reset_index()
        df = df.merge(em_agg, on="ID_POLICE", how="left")
    else:
        df["NB_QUITTANCES"] = 0
        df["PRIME_PAR_AN"]  = 0.0

    # ── Sinistre agg ──
    if not sinistre.empty:
        sinistre["MT_EVALUATION"] = pd.to_numeric(sinistre.get("MT_EVALUATION"), errors="coerce").fillna(0)
        sin_agg = sinistre.groupby("ID_POLICE").agg(
            NB_SINISTRES=("NUM_SINISTRE","count"),
            MT_EVAL_SUM =("MT_EVALUATION","sum"),
        ).reset_index()
        df = df.merge(sin_agg, on="ID_POLICE", how="left")
    else:
        df["NB_SINISTRES"] = 0
        df["MT_EVAL_SUM"]  = 0.0

    df["NB_SINISTRES"] = df.get("NB_SINISTRES", 0).fillna(0)
    df["MT_EVAL_SUM"]  = df.get("MT_EVAL_SUM",  0).fillna(0)
    df["FREQ_SINISTRE"] = df["NB_SINISTRES"] / df["EXPOSITION_ANS"].replace(0, 1)
    df["SEVERITE_MOY"]  = df["MT_EVAL_SUM"] / df["NB_SINISTRES"].replace(0, np.nan)
    df["SEVERITE_MOY"]  = df["SEVERITE_MOY"].fillna(0)
    df["SP_RATIO"]      = df["MT_EVAL_SUM"] / df["PRIME_PAR_AN"].replace(0, np.nan)
    df["SP_RATIO"]      = df["SP_RATIO"].fillna(0)

    # ── Impayé agg ──
    if not impaye.empty:
        imp_agg = impaye.groupby("ID_POLICE")["NUM_QUITTANCE"].count().reset_index(name="NB_IMPAYES")
        df = df.merge(imp_agg, on="ID_POLICE", how="left")
    df["NB_IMPAYES"] = df.get("NB_IMPAYES", 0).fillna(0)
    df["TAUX_IMPAYE"] = df["NB_IMPAYES"] / df["NB_QUITTANCES"].replace(0, 1)

    # ── Véhicule ──
    if not vehicule.empty:
        vehicule["PUISSANCE"]   = pd.to_numeric(vehicule.get("PUISSANCE"),   errors="coerce").fillna(6)
        vehicule["NB_PLACE"]    = pd.to_numeric(vehicule.get("NB_PLACE"),    errors="coerce").fillna(5)
        vehicule["DATE_MEC"]    = pd.to_datetime(vehicule.get("DATE_MISE_CIRCULATION"), errors="coerce")
        vehicule["AGE_VEHICULE"]= ((ref - vehicule["DATE_MEC"]).dt.days / 365.25).clip(0).fillna(5)
        df = df.merge(vehicule[["ID_VEHICULE","PUISSANCE","NB_PLACE","AGE_VEHICULE"]]
                      .rename(columns={"ID_VEHICULE":"ID_POLICE"}),
                      on="ID_POLICE", how="left")
    else:
        df["PUISSANCE"]    = 6.0
        df["NB_PLACE"]     = 5.0
        df["AGE_VEHICULE"] = 5.0

    df["AGE_CLIENT"] = 40.0  # no client join here — use placeholder

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


# ── Proxy model trainer ─────────────────────────────────────────────────────

def _train_proxy(df: pd.DataFrame) -> dict:
    valid = [c for c in FEATURE_COLS if c in df.columns]
    X = df[valid].fillna(0).values
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = imputer.fit_transform(X)
    X_sc    = scaler.fit_transform(X_imp)

    # Frequency model
    y_freq = df["FREQ_SINISTRE"].clip(0, 2).values
    freq_model = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                            learning_rate=0.1, random_state=42)
    freq_model.fit(X_sc, np.log1p(y_freq))

    # Severity model
    y_sev = df["SEVERITE_MOY"].clip(0).values
    sev_model = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                           learning_rate=0.1, random_state=42)
    sev_model.fit(X_sc, np.log1p(y_sev))

    # Normalisation anchors
    df["PRED_FREQ"] = np.expm1(freq_model.predict(X_sc)).clip(0)
    df["PRED_SEV"]  = np.expm1(sev_model.predict(X_sc)).clip(0)
    anchors = {
        "freq_p1":  float(df["PRED_FREQ"].quantile(0.01)),
        "freq_p99": float(df["PRED_FREQ"].quantile(0.99)),
        "sev_p1":   float(df["PRED_SEV"].quantile(0.01)),
        "sev_p99":  float(df["PRED_SEV"].quantile(0.99)),
    }

    art = {
        "freq_model":   freq_model,
        "sev_model":    sev_model,
        "imputer":      imputer,
        "scaler":       scaler,
        "features":     valid,
        "anchors":      anchors,
        "chargement":   _CHARGEMENT_TOTAL,
        "source":       "proxy",
    }
    save_artifact(_ARTIFACT_NAME, art)
    logger.info("Risk proxy model trained and saved.")
    return art, df


def _get_artifact_and_df() -> tuple[dict, pd.DataFrame]:
    art = load_artifact(_ARTIFACT_NAME)
    df  = _build_portfolio()
    if df.empty:
        raise RuntimeError("No data for risk service.")
    if art is None:
        return _train_proxy(df)
    return art, df


# ── Score computation ────────────────────────────────────────────────────────

def _compute_score(art: dict, row: pd.Series) -> dict:
    feats = art["features"]
    X = art["imputer"].transform(row[feats].fillna(0).values.reshape(1, -1))
    X = art["scaler"].transform(X)

    freq = float(np.expm1(art["freq_model"].predict(X)[0]))
    sev  = float(np.expm1(art["sev_model"].predict(X)[0]))

    prime_pure       = max(0, freq * sev)
    prime_technique  = round(prime_pure * art.get("chargement", _CHARGEMENT_TOTAL))
    loading_factor   = round(art.get("chargement", _CHARGEMENT_TOTAL) - 1, 3)

    anchors = art.get("anchors", {})
    score_freq = normalize_0_1000(freq, anchors.get("freq_p1",0), anchors.get("freq_p99",0.5))
    score_sev  = normalize_0_1000(sev,  anchors.get("sev_p1",0),  anchors.get("sev_p99",10000))
    taux_imp   = float(row.get("TAUX_IMPAYE", 0))
    score_imp  = min(1000, taux_imp * 2000)

    risk_score = int(0.45 * score_freq + 0.40 * score_sev + 0.15 * score_imp)
    return {
        "risk_score":       risk_score,
        "risk_label":       _risk_class(risk_score),
        "prime_technique":  prime_technique,
        "loading_factor":   loading_factor,
        "components": {
            "frequence_estimee": round(freq, 4),
            "severite_estimee":  round(sev,  0),
            "prime_pure":        round(prime_pure, 0),
        },
        "model": art.get("source", "LightGBM freq+sev"),
    }


# ── Public API ──────────────────────────────────────────────────────────────

def score_risk(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute risk score + prime technique for a single policy."""
    art, df = _get_artifact_and_df()
    medians  = df[art["features"]].median()
    row = medians.copy()
    map_keys = {
        "bonus_malus":       "BONUS_MALUS",
        "puissance":         "PUISSANCE",
        "age_vehicule":      "AGE_VEHICULE",
        "age_client":        "AGE_CLIENT",
        "nb_sinistres_hist": "NB_SINISTRES",
        "mt_pnet":           "PRIME_PAR_AN",
    }
    for src, dst in map_keys.items():
        if src in payload and dst in row.index:
            row[dst] = safe_float(payload[src])

    return _compute_score(art, row)


def get_risk_table() -> dict[str, Any]:
    """Return aggregated risk table Branche × Genre Véhicule."""
    _, df = _get_artifact_and_df()

    group_cols = ["ID_BRANCHE"]
    if "GENRE_VEHICULE" in df.columns:
        group_cols.append("GENRE_VEHICULE")

    agg = df.groupby(group_cols).agg(
        NB_POLICES  =("ID_POLICE",     "count"),
        FREQ_SIN_MOY=("FREQ_SINISTRE", "mean"),
        SEV_MOY     =("SEVERITE_MOY",  "mean"),
        SP_RATIO_MOY=("SP_RATIO",      "mean"),
        PRIME_MOY   =("PRIME_PAR_AN",  "mean"),
    ).reset_index()

    agg["PRIME_TECHNIQUE"] = (agg["FREQ_SIN_MOY"] * agg["SEV_MOY"] * _CHARGEMENT_TOTAL).round(0)

    # Map ID_BRANCHE to name if column present
    branche_map = {1:"AUTO", 2:"IRDS", 3:"SANTE"}
    if "ID_BRANCHE" in agg.columns:
        agg["BRANCHE"] = agg["ID_BRANCHE"].map(branche_map).fillna(agg["ID_BRANCHE"].astype(str))

    records = []
    for _, row in agg.iterrows():
        records.append({
            "branche":        str(row.get("BRANCHE", row.get("ID_BRANCHE","—"))),
            "genre_vehicule": str(row.get("GENRE_VEHICULE","—")),
            "freq_sin_moy":   round(float(row["FREQ_SIN_MOY"]),  4),
            "sev_moy":        round(float(row["SEV_MOY"]),        0),
            "sp_ratio_moy":   round(float(row["SP_RATIO_MOY"]),  3),
            "prime_technique":round(float(row["PRIME_TECHNIQUE"]),0),
        })

    return {"table": records}
