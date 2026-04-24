"""
ml_services/segmentation_service.py
────────────────────────────────────
Customer segmentation service — customer_segmentation.ipynb

Two-level segmentation:
  1. RFM scoring (Recency · Frequency · Monetary) — rule-based
  2. K-Means clustering (k=4) on 18 behavioural features

When the notebook artifact (segmentation_model.pkl) is absent the service
builds the features from DB/CSV and runs K-Means on-the-fly.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from ._base import (
    load_artifact, load_emission, load_impaye, load_police,
    load_sinistre, save_artifact, safe_float,
)

logger = logging.getLogger("maghrebia.ml_services.segmentation")

_ARTIFACT_NAME = "segmentation_model"
_K_OPTIMAL = 4

CLUSTER_FEATURES = [
    "RECENCY_DAYS", "NB_POLICES", "NB_QUITTANCES", "TOTAL_PRIMES",
    "PRIME_PAR_POLICE", "NB_BRANCHES", "NB_POLICES_ACTIVES",
    "TAUX_RESILIATION", "ANCIENNETE_JOURS", "BONUS_MALUS_MOY",
    "NB_SINISTRES", "SP_RATIO", "TAUX_IMPAYE", "NB_IMPAYES",
    "FLAG_MORALE", "FLAG_MULTI_BRANCHES",
]

# Log-transform columns (heavy right-skew)
_LOG_COLS = ["TOTAL_PRIMES", "PRIME_PAR_POLICE", "NB_QUITTANCES",
             "RECENCY_DAYS", "ANCIENNETE_JOURS", "NB_SINISTRES"]


# ── Feature builder ─────────────────────────────────────────────────────────

def _build_master() -> pd.DataFrame:
    police   = load_police()
    emission = load_emission()
    sinistre = load_sinistre()
    impaye   = load_impaye()

    if police.empty:
        return pd.DataFrame()

    police.columns  = [c.upper() for c in police.columns]
    police["DATE_EFFET"]    = pd.to_datetime(police.get("DATE_EFFET"),    errors="coerce")
    police["DATE_ECHEANCE"] = pd.to_datetime(police.get("DATE_ECHEANCE"), errors="coerce")
    police["BONUS_MALUS"]   = pd.to_numeric(police.get("BONUS_MALUS"),    errors="coerce").fillna(1.0)
    police["FLAG_MORALE"]   = (police.get("TYPE_POLICE", "") == "E").astype(int)

    ref = pd.Timestamp("today")

    # ── Client-level aggregation (group polices by client) ──
    # Per-police features first
    pol = police[["ID_POLICE", "ID_CLIENT", "ID_BRANCHE", "BRANCHE",
                  "SITUATION", "BONUS_MALUS", "FLAG_MORALE",
                  "DATE_EFFET", "DATE_ECHEANCE"]].copy()
    pol["ACTIVE"] = pol["SITUATION"].isin(["V", "S"]).astype(int)
    pol["ANCIENNETE_JOURS"] = (ref - pol["DATE_EFFET"]).dt.days.clip(0)

    # Emission aggregation per police
    if not emission.empty:
        emission.columns = [c.upper() for c in emission.columns]
        emission["MT_PNET"] = pd.to_numeric(emission.get("MT_PNET"), errors="coerce").fillna(0)
        em_pol = emission.groupby("ID_POLICE").agg(
            NB_QUITTANCES=("NUM_QUITTANCE","count"),
            TOTAL_PRIMES =("MT_PNET",      "sum"),
        ).reset_index()
        pol = pol.merge(em_pol, on="ID_POLICE", how="left")
    else:
        pol["NB_QUITTANCES"] = 0
        pol["TOTAL_PRIMES"]  = 0.0

    pol["NB_QUITTANCES"] = pol["NB_QUITTANCES"].fillna(0)
    pol["TOTAL_PRIMES"]  = pol["TOTAL_PRIMES"].fillna(0)

    # Sinistre aggregation per police
    if not sinistre.empty:
        sinistre.columns = [c.upper() for c in sinistre.columns]
        sinistre["MT_EVALUATION"] = pd.to_numeric(sinistre.get("MT_EVALUATION"), errors="coerce").fillna(0)
        sinistre["MT_PAYE"]       = pd.to_numeric(sinistre.get("MT_PAYE"),       errors="coerce").fillna(0)
        sin_pol = sinistre.groupby("ID_POLICE").agg(
            NB_SINISTRES=("NUM_SINISTRE",  "count"),
            MT_EVAL_SUM =("MT_EVALUATION", "sum"),
        ).reset_index()
        pol = pol.merge(sin_pol, on="ID_POLICE", how="left")
    else:
        pol["NB_SINISTRES"] = 0
        pol["MT_EVAL_SUM"]  = 0.0

    pol["NB_SINISTRES"] = pol["NB_SINISTRES"].fillna(0)
    pol["MT_EVAL_SUM"]  = pol["MT_EVAL_SUM"].fillna(0)
    pol["SP_RATIO"] = pol["MT_EVAL_SUM"] / pol["TOTAL_PRIMES"].replace(0, np.nan)

    # Impayé per police
    if not impaye.empty:
        impaye.columns = [c.upper() for c in impaye.columns]
        imp_pol = impaye.groupby("ID_POLICE")["NUM_QUITTANCE"].count().reset_index(name="NB_IMPAYES")
        pol = pol.merge(imp_pol, on="ID_POLICE", how="left")
    pol["NB_IMPAYES"] = pol.get("NB_IMPAYES", 0).fillna(0)
    pol["TAUX_IMPAYE"] = pol["NB_IMPAYES"] / pol["NB_QUITTANCES"].replace(0, 1)

    # ── Aggregate polices per client ──
    master = pol.groupby("ID_CLIENT").agg(
        NB_POLICES        =("ID_POLICE",       "count"),
        NB_POLICES_ACTIVES=("ACTIVE",          "sum"),
        TAUX_RESILIATION  =("ACTIVE",          lambda x: 1 - x.mean()),
        TOTAL_PRIMES      =("TOTAL_PRIMES",    "sum"),
        NB_QUITTANCES     =("NB_QUITTANCES",   "sum"),
        NB_BRANCHES       =("ID_BRANCHE",      "nunique"),
        ANCIENNETE_JOURS  =("ANCIENNETE_JOURS","max"),
        BONUS_MALUS_MOY   =("BONUS_MALUS",     "mean"),
        NB_SINISTRES      =("NB_SINISTRES",    "sum"),
        SP_RATIO          =("SP_RATIO",        "mean"),
        NB_IMPAYES        =("NB_IMPAYES",      "sum"),
        TAUX_IMPAYE       =("TAUX_IMPAYE",     "mean"),
        FLAG_MORALE       =("FLAG_MORALE",     "max"),
        LAST_DATE_EFFET   =("DATE_EFFET",      "max"),
    ).reset_index()

    master["RECENCY_DAYS"]      = (ref - master["LAST_DATE_EFFET"]).dt.days.clip(0).fillna(365)
    master["PRIME_PAR_POLICE"]  = master["TOTAL_PRIMES"] / master["NB_POLICES"].replace(0, 1)
    master["FLAG_MULTI_BRANCHES"] = (master["NB_BRANCHES"] > 1).astype(int)

    # CLV estimate (3-year horizon)
    prime_ann = master["TOTAL_PRIMES"] / (master["ANCIENNETE_JOURS"].clip(30) / 365.25)
    master["CLV_ESTIME"] = (
        prime_ann.fillna(0)
        * 3
        * (1 - master["TAUX_RESILIATION"].clip(0, 1).fillna(0.13))
        * (1 - master["SP_RATIO"].clip(0, 1).fillna(0))
        * (1 - master["TAUX_IMPAYE"].clip(0, 1).fillna(0))
    ).clip(0)

    master[CLUSTER_FEATURES] = master[CLUSTER_FEATURES].fillna(0)
    return master


# ── Cluster naming ──────────────────────────────────────────────────────────

def _name_cluster(row_norm: pd.Series, row_raw: pd.Series) -> str:
    valeur   = safe_float(row_norm.get("TOTAL_PRIMES", 0))
    recency  = 1 - safe_float(row_norm.get("RECENCY_DAYS", 0))
    risque   = safe_float(row_norm.get("SP_RATIO", 0))
    impaye   = safe_float(row_norm.get("TAUX_IMPAYE", 0))
    anciennt = safe_float(row_norm.get("ANCIENNETE_JOURS", 0))
    morale   = safe_float(row_raw.get("FLAG_MORALE", 0))

    if morale > 0.3:
        return "🏢 Entreprises / Flottes"
    if valeur > 0.70 and recency > 0.60:
        return "🏆 VIP — Haute Valeur"
    if anciennt > 0.65 and risque < 0.40:
        return "🎯 Fidèles Rentables"
    if impaye > 0.55 or risque > 0.65:
        return "⚠️ Clients à Risque"
    if recency < 0.30:
        return "😴 Clients Dormants"
    return "🌱 Clients Potentiels"


_ACTION_PLAN = {
    "🏆 VIP — Haute Valeur":    "Programme fidélité premium — upsell multi-branches & garanties exclusives",
    "🎯 Fidèles Rentables":     "Renouvellement proactif — offre ancienneté + extension de garanties",
    "🌱 Clients Potentiels":    "Campagne activation — améliorer engagement digital + offre découverte",
    "⚠️ Clients à Risque":     "Intervention urgente agent — plan de paiement & rétention ciblée",
    "😴 Clients Dormants":      "Campagne réactivation — offre remise spéciale retour portefeuille",
    "🏢 Entreprises / Flottes": "Contrat cadre entreprise — tarification flotte & gestionnaire dédié",
}

_RADAR_DIMS = [
    ("Valeur",      "TOTAL_PRIMES"),
    ("Fidélité",    "ANCIENNETE_JOURS"),
    ("Engagement",  "NB_POLICES"),
    ("Fiabilité",   "TAUX_IMPAYE"),       # inverted
    ("Rentabilité", "SP_RATIO"),          # inverted
]


def _build_artifact() -> dict:
    master = _build_master()
    if master.empty:
        raise RuntimeError("No data to build segmentation artifact.")

    valid_feats = [c for c in CLUSTER_FEATURES if c in master.columns]
    df_clust = master[valid_feats].copy()

    # Log-transform skewed cols
    for col in _LOG_COLS:
        if col in df_clust.columns:
            df_clust[col] = np.log1p(df_clust[col].clip(0))

    # Cap outliers at p99
    for col in df_clust.columns:
        p99 = df_clust[col].quantile(0.99)
        df_clust[col] = df_clust[col].clip(upper=p99)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_clust.fillna(0))

    kmeans = KMeans(n_clusters=_K_OPTIMAL, init="k-means++", n_init=20,
                    max_iter=500, random_state=42)
    master["CLUSTER"] = kmeans.fit_predict(X_scaled)

    # Name clusters
    cp      = master.groupby("CLUSTER")[valid_feats].mean()
    cp_norm = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)
    cluster_names = {
        c: _name_cluster(cp_norm.loc[c], cp.loc[c]) for c in cp.index
    }
    master["CLUSTER_NOM"] = master["CLUSTER"].map(cluster_names)

    art = {
        "kmeans":          kmeans,
        "scaler":          scaler,
        "cluster_features": valid_feats,
        "log_cols":        _LOG_COLS,
        "cluster_names":   cluster_names,
        "source":          "proxy",
    }
    save_artifact(_ARTIFACT_NAME, art)
    return art, master


def _get_artifact_and_master() -> tuple[dict, pd.DataFrame]:
    art    = load_artifact(_ARTIFACT_NAME)
    master = _build_master()
    if master.empty:
        raise RuntimeError("No data for segmentation service.")

    if art is None:
        return _build_artifact()

    # Apply stored model to current data
    valid_feats = art["cluster_features"]
    df_clust = master[[c for c in valid_feats if c in master.columns]].copy()
    for col in art.get("log_cols", []):
        if col in df_clust.columns:
            df_clust[col] = np.log1p(df_clust[col].clip(0))
    df_clust = df_clust.fillna(0)
    for col in df_clust.columns:
        p99 = df_clust[col].quantile(0.99)
        df_clust[col] = df_clust[col].clip(upper=p99)

    X_scaled = art["scaler"].transform(df_clust)
    master["CLUSTER"]     = art["kmeans"].predict(X_scaled)
    master["CLUSTER_NOM"] = master["CLUSTER"].map(art["cluster_names"])
    return art, master


# ── Radar builder ────────────────────────────────────────────────────────────

def _build_radar(row_norm: pd.Series) -> list[dict]:
    result = []
    for label, col in _RADAR_DIMS:
        val = float(row_norm.get(col, 0.5))
        if label in ("Fiabilité", "Rentabilité"):
            val = 1 - val       # lower = better for risk metrics
        result.append({"metric": label, "value": round(val, 3)})
    return result


# ── Public API ──────────────────────────────────────────────────────────────

def get_segmentation_summary() -> dict[str, Any]:
    art, master = _get_artifact_and_master()

    valid_feats = art["cluster_features"]
    cp     = master.groupby("CLUSTER_NOM")[
        [c for c in valid_feats if c in master.columns]
    ].mean()
    cp_norm = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)

    sizes = master["CLUSTER_NOM"].value_counts()
    total = len(master)

    segments = []
    for name in sizes.index:
        count = int(sizes[name])
        row_n = cp_norm.loc[name] if name in cp_norm.index else pd.Series(dtype=float)
        row_r = cp.loc[name]     if name in cp.index      else pd.Series(dtype=float)

        # LTV from master
        ltv = float(master.loc[master["CLUSTER_NOM"] == name, "CLV_ESTIME"].mean()) if "CLV_ESTIME" in master.columns else 0.0

        segments.append({
            "name":               name,
            "count":              count,
            "share_pct":          round(count / total * 100, 1),
            "avg_prime":          round(float(row_r.get("TOTAL_PRIMES",       0)), 0),
            "avg_anciennete_jours": round(float(row_r.get("ANCIENNETE_JOURS", 0)), 0),
            "avg_churn_risk":     round(float(row_r.get("TAUX_RESILIATION",   0)), 3),
            "avg_taux_impaye":    round(float(row_r.get("TAUX_IMPAYE",        0)), 3),
            "avg_sp_ratio":       round(float(row_r.get("SP_RATIO",           0)), 3),
            "avg_ltv":            round(ltv, 0),
            "action":             _ACTION_PLAN.get(name, "Surveillance standard"),
            "radar":              _build_radar(row_n),
        })

    # Sort by avg_prime descending
    segments.sort(key=lambda s: s["avg_prime"], reverse=True)

    return {
        "nb_clients":  total,
        "nb_clusters": _K_OPTIMAL,
        "segments":    segments,
        "model_source": art.get("source", "artifact"),
    }
