"""
ml_services/segmentation_service.py
────────────────────────────────────
Customer segmentation service — customer_segmentation.ipynb

Two-level segmentation:
  1. RFM scoring (Recency · Frequency · Monetary) — rule-based
  2. K-Means clustering (k=5) on 15 behavioural features

Preprocessing must mirror fix_segmentation.py exactly:
  - SP_RATIO excluded (zero-inflated → RobustScaler blows up)
  - p99 cap BEFORE log1p transform
  - log1p on: TOTAL_PRIMES, PRIME_PAR_POLICE, NB_QUITTANCES,
               NB_SINISTRES, NB_IMPAYES, ANCIENNETE_JOURS
  - FLAG_MORALE from DIM_CLIENT.TYPE_PERSONNE == 'M' (not TYPE_POLICE)
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
_K_OPTIMAL = 5

CLUSTER_FEATURES = [
    "RECENCY_DAYS",
    "NB_POLICES",
    "NB_QUITTANCES",
    "TOTAL_PRIMES",
    "PRIME_PAR_POLICE",
    "NB_BRANCHES",
    "NB_POLICES_ACTIVES",
    "TAUX_RESILIATION",
    "ANCIENNETE_JOURS",
    "BONUS_MALUS_MOY",
    "NB_SINISTRES",
    "TAUX_IMPAYE",
    "NB_IMPAYES",
    "FLAG_MORALE",
    "FLAG_MULTI_BRANCHES",
]

# Columns to log1p-transform — must match fix_segmentation.py LOG_COLS
_LOG_COLS = [
    "TOTAL_PRIMES", "PRIME_PAR_POLICE", "NB_QUITTANCES",
    "NB_SINISTRES", "NB_IMPAYES", "ANCIENNETE_JOURS",
]

_ACTION_PLAN = {
    "🏆 VIP — Haute Valeur":    "Programme fidélité premium — upsell multi-branches & garanties exclusives",
    "🎯 Fidèles Rentables":     "Renouvellement proactif — offre ancienneté + extension de garanties",
    "🌱 Clients Potentiels":    "Campagne activation — améliorer engagement digital + offre découverte",
    "⚠️ Clients à Risque":     "Intervention urgente agent — plan de paiement & rétention ciblée",
    "😴 Clients Dormants":      "Campagne réactivation — offre remise spéciale retour portefeuille",
    "🏢 Entreprises / Flottes": "Contrat cadre entreprise — tarification flotte & gestionnaire dédié",
}

_RADAR_DIMS = [
    ("Valeur",       "TOTAL_PRIMES"),
    ("Fidélité",     "ANCIENNETE_JOURS"),
    ("Engagement",   "NB_POLICES"),
    ("Fiabilité",    "TAUX_IMPAYE"),        # inverted
    ("Rentabilité",  "TAUX_RESILIATION"),   # inverted
]


# ── Feature builder ──────────────────────────────────────────────────────────

def _build_master(ref: pd.Timestamp | None = None) -> pd.DataFrame:
    if ref is None:
        ref = pd.Timestamp("today")
    police   = load_police()
    emission = load_emission()
    sinistre = load_sinistre()
    impaye   = load_impaye()

    if police.empty:
        return pd.DataFrame()

    police.columns  = [c.upper() for c in police.columns]
    police["DATE_EFFET"] = pd.to_datetime(police.get("DATE_EFFET"), errors="coerce")
    police["BONUS_MALUS"] = pd.to_numeric(police.get("BONUS_MALUS"), errors="coerce").fillna(1.0)

    # FLAG_MORALE: try to get from DIM_CLIENT (TYPE_PERSONNE == 'M')
    # _base.py may not have load_client, so we check and fallback gracefully
    police["FLAG_MORALE"] = 0
    try:
        from ._base import load_client
        client = load_client()
        if not client.empty:
            client.columns = [c.upper() for c in client.columns]
            if "TYPE_PERSONNE" in client.columns:
                cli_m = client[["ID_CLIENT", "TYPE_PERSONNE"]].copy()
                cli_m["FLAG_MORALE"] = (cli_m["TYPE_PERSONNE"].str.strip().str.upper() == "M").astype(int)
                police = police.merge(cli_m[["ID_CLIENT", "FLAG_MORALE"]], on="ID_CLIENT", how="left",
                                      suffixes=("", "_cli"))
                police["FLAG_MORALE"] = police.get("FLAG_MORALE_cli", police["FLAG_MORALE"]).fillna(0).astype(int)
                if "FLAG_MORALE_cli" in police.columns:
                    police.drop(columns=["FLAG_MORALE_cli"], inplace=True)
    except (ImportError, Exception) as exc:
        logger.debug("load_client unavailable (%s) — FLAG_MORALE set to 0", exc)

    pol = police[["ID_POLICE", "ID_CLIENT", "BRANCHE", "SITUATION",
                  "BONUS_MALUS", "FLAG_MORALE", "DATE_EFFET"]].copy()
    pol["ACTIVE"] = pol["SITUATION"].isin(["V", "S"]).astype(int)
    pol["ANCIENNETE_JOURS"] = (ref - pol["DATE_EFFET"]).dt.days.clip(0)

    # Emission per police → per client
    if not emission.empty:
        emission.columns = [c.upper() for c in emission.columns]
        mt_ptt  = pd.to_numeric(emission["MT_PTT"],  errors="coerce") if "MT_PTT"  in emission.columns else pd.Series(np.nan, index=emission.index)
        mt_pnet = pd.to_numeric(emission["MT_PNET"], errors="coerce") if "MT_PNET" in emission.columns else pd.Series(np.nan, index=emission.index)
        emission["PRIME"] = mt_ptt.fillna(mt_pnet).fillna(0)
        if "DATE_EMISSION" in emission.columns:
            emission["DATE_EMISSION"] = pd.to_datetime(emission["DATE_EMISSION"], errors="coerce")
            em_pol = emission.groupby("ID_POLICE").agg(
                NB_QUITTANCES=("NUM_QUITTANCE", "count"),
                TOTAL_PRIMES =("PRIME",         "sum"),
                LAST_EMISSION=("DATE_EMISSION", "max"),
            ).reset_index()
        else:
            em_pol = emission.groupby("ID_POLICE").agg(
                NB_QUITTANCES=("NUM_QUITTANCE", "count"),
                TOTAL_PRIMES =("PRIME",         "sum"),
            ).reset_index()
            em_pol["LAST_EMISSION"] = pd.NaT
        pol = pol.merge(em_pol, on="ID_POLICE", how="left")
    else:
        pol["NB_QUITTANCES"] = 0
        pol["TOTAL_PRIMES"]  = 0.0
        pol["LAST_EMISSION"] = pd.NaT

    pol["NB_QUITTANCES"] = pol["NB_QUITTANCES"].fillna(0)
    pol["TOTAL_PRIMES"]  = pol["TOTAL_PRIMES"].fillna(0)

    # Sinistre per police → per client (FACT_SINISTRE has ID_CLIENT directly)
    if not sinistre.empty:
        sinistre.columns = [c.upper() for c in sinistre.columns]
        sinistre["MT_PAYE"] = pd.to_numeric(sinistre.get("MT_PAYE"), errors="coerce").fillna(0)
        if "ID_CLIENT" in sinistre.columns:
            # Direct groupby on ID_CLIENT
            sin_cli = sinistre.groupby("ID_CLIENT").agg(
                NB_SINISTRES  =("NUM_SINISTRE", "count"),
                COUT_SINISTRES=("MT_PAYE",      "sum"),
            ).reset_index()
        else:
            sin_pol = sinistre.groupby("ID_POLICE").agg(
                NB_SINISTRES  =("NUM_SINISTRE", "count"),
                COUT_SINISTRES=("MT_PAYE",      "sum"),
            ).reset_index()
            sin_cli = (pol[["ID_POLICE", "ID_CLIENT"]]
                       .merge(sin_pol, on="ID_POLICE", how="left")
                       .groupby("ID_CLIENT").agg(
                           NB_SINISTRES  =("NB_SINISTRES",   "sum"),
                           COUT_SINISTRES=("COUT_SINISTRES",  "sum"),
                       ).reset_index())
    else:
        sin_cli = pd.DataFrame(columns=["ID_CLIENT", "NB_SINISTRES", "COUT_SINISTRES"])

    # Impayé per police → per client
    if not impaye.empty:
        impaye.columns = [c.upper() for c in impaye.columns]
        imp_pol = impaye.groupby("ID_POLICE")["NUM_QUITTANCE"].count().reset_index(name="NB_IMPAYES")
        pol = pol.merge(imp_pol, on="ID_POLICE", how="left")
    pol["NB_IMPAYES"] = pol.get("NB_IMPAYES", 0).fillna(0)

    # Aggregate polices per client
    master = pol.groupby("ID_CLIENT").agg(
        NB_POLICES         =("ID_POLICE",        "count"),
        NB_POLICES_ACTIVES =("ACTIVE",            "sum"),
        TOTAL_PRIMES       =("TOTAL_PRIMES",      "sum"),
        NB_QUITTANCES      =("NB_QUITTANCES",     "sum"),
        NB_BRANCHES        =("BRANCHE",           "nunique"),
        ANCIENNETE_JOURS   =("ANCIENNETE_JOURS",  "max"),
        BONUS_MALUS_MOY    =("BONUS_MALUS",       "mean"),
        NB_IMPAYES         =("NB_IMPAYES",        "sum"),
        FLAG_MORALE        =("FLAG_MORALE",       "max"),
        NB_POLICES_RESIL   =("ACTIVE",            lambda x: (x == 0).sum()),
        LAST_EMISSION      =("LAST_EMISSION",     "max"),
    ).reset_index()

    master["RECENCY_DAYS"]       = (ref - master["LAST_EMISSION"]).dt.days.clip(0).fillna(730)
    master["PRIME_PAR_POLICE"]   = master["TOTAL_PRIMES"] / master["NB_POLICES"].replace(0, 1)
    master["FLAG_MULTI_BRANCHES"]= (master["NB_BRANCHES"] > 1).astype(int)
    master["TAUX_RESILIATION"]   = master["NB_POLICES_RESIL"] / master["NB_POLICES"].replace(0, 1)
    master["TAUX_IMPAYE"]        = master["NB_IMPAYES"] / master["NB_QUITTANCES"].replace(0, 1)

    # Merge sinistre
    master = master.merge(sin_cli, on="ID_CLIENT", how="left")
    master["NB_SINISTRES"]   = master["NB_SINISTRES"].fillna(0)
    master["COUT_SINISTRES"] = master.get("COUT_SINISTRES", 0).fillna(0)

    # SP_RATIO (not used in clustering but kept for profiling)
    master["SP_RATIO"] = np.where(
        master["TOTAL_PRIMES"] > 0,
        master["COUT_SINISTRES"] / master["TOTAL_PRIMES"], 0
    ).clip(0, 1.5)

    # CLV estimate
    prime_ann = master["TOTAL_PRIMES"] / (master["ANCIENNETE_JOURS"].clip(30) / 365.25)
    master["CLV_ESTIME"] = (
        prime_ann.fillna(0)
        * 3
        * (1 - master["TAUX_RESILIATION"].clip(0, 1).fillna(0.13))
        * (1 - master["SP_RATIO"].clip(0, 1).fillna(0))
        * (1 - master["TAUX_IMPAYE"].clip(0, 1).fillna(0))
    ).clip(0)

    master[CLUSTER_FEATURES] = master[CLUSTER_FEATURES].fillna(0)
    return master[master["NB_POLICES"] > 0].reset_index(drop=True)


def _preprocess(master: pd.DataFrame, art: dict) -> np.ndarray:
    """Apply the same preprocessing as fix_segmentation.py to get X_scaled."""
    feats = art.get("cluster_features", CLUSTER_FEATURES)
    log_c = art.get("log_cols", _LOG_COLS)
    p99_caps = art.get("p99_caps", {})

    df = master[[c for c in feats if c in master.columns]].copy().fillna(0)

    # 1. Cap at training p99 values (use live p99 only as fallback)
    for col in df.columns:
        cap = p99_caps.get(col)
        if cap is None:
            cap = float(df[col].quantile(0.99))
        if cap > 0:
            df[col] = df[col].clip(upper=cap)

    # 2. Log1p transform heavy-skew columns
    for col in log_c:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(0))

    X = art["scaler"].transform(df.fillna(0))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ── Radar builder ────────────────────────────────────────────────────────────

def _build_radar(row_norm: pd.Series, art: dict) -> list[dict]:
    radar_dims = art.get("radar_dims", _RADAR_DIMS)
    result = []
    for label, col in radar_dims:
        val = float(row_norm.get(col, 0.5))
        if label in ("Fiabilité", "Rentabilité"):
            val = 1.0 - val
        result.append({"metric": label, "value": round(max(0.0, min(1.0, val)), 3)})
    return result


# ── Artifact builder (fallback when pkl absent) ───────────────────────────────

def _build_artifact() -> tuple[dict, pd.DataFrame]:
    master = _build_master()
    if master.empty:
        raise RuntimeError("No data to build segmentation artifact.")

    feats = [c for c in CLUSTER_FEATURES if c in master.columns]
    df = master[feats].copy().fillna(0)
    for col in feats:
        p99 = df[col].quantile(0.99)
        if p99 > 0:
            df[col] = df[col].clip(upper=p99)
    for col in _LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(0))

    scaler = RobustScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(df.fillna(0)), nan=0.0)

    kmeans = KMeans(n_clusters=_K_OPTIMAL, init="k-means++", n_init=20,
                    max_iter=500, random_state=42)
    master["CLUSTER"] = kmeans.fit_predict(X_scaled)

    cp               = master.groupby("CLUSTER")[feats].mean()
    cp["NB_CLIENTS"] = master.groupby("CLUSTER").size()
    cp_norm          = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)

    def _rank_assign(cp):
        rem = list(cp.index); out = {}
        out[cp.loc[rem, "TOTAL_PRIMES"].idxmax()]                          = "🏆 VIP — Haute Valeur";  rem.remove(list(out)[-1])
        out[cp.loc[rem, "RECENCY_DAYS"].idxmax()]                          = "😴 Clients Dormants";    rem.remove(list(out)[-1])
        anc = "ANCIENNETE_JOURS" if "ANCIENNETE_JOURS" in cp.columns else "TOTAL_PRIMES"
        out[cp.loc[rem, anc].idxmax()]                                     = "🎯 Fidèles Rentables";   rem.remove(list(out)[-1])
        out[cp.loc[rem, "NB_CLIENTS"].idxmin() if "NB_CLIENTS" in cp.columns else rem[0]] = "🏢 Entreprises / Flottes"; rem = [r for r in rem if r not in out]
        for r in rem: out[r] = "🌱 Clients Potentiels"
        return out
    cluster_names = _rank_assign(cp)

    master["CLUSTER_NOM"] = master["CLUSTER"].map(cluster_names)
    art = {
        "kmeans": kmeans, "scaler": scaler,
        "cluster_features": feats, "log_cols": _LOG_COLS,
        "cluster_names": cluster_names, "source": "proxy",
        "action_plan": _ACTION_PLAN, "radar_dims": _RADAR_DIMS,
    }
    save_artifact(_ARTIFACT_NAME, art)
    return art, master


def _get_artifact_and_master() -> tuple[dict, pd.DataFrame]:
    art = load_artifact(_ARTIFACT_NAME)
    if art is None:
        return _build_artifact()

    ref_date = art.get("ref_date")
    if ref_date is not None and not isinstance(ref_date, pd.Timestamp):
        ref_date = pd.Timestamp(ref_date)

    master = _build_master(ref=ref_date)
    if master.empty:
        raise RuntimeError("No data for segmentation service.")

    X_scaled = _preprocess(master, art)
    master["CLUSTER"]     = art["kmeans"].predict(X_scaled)
    master["CLUSTER_NOM"] = master["CLUSTER"].map(art["cluster_names"])
    return art, master


# ── Public API ───────────────────────────────────────────────────────────────

def get_segmentation_summary() -> dict[str, Any]:
    art, master = _get_artifact_and_master()

    feats = art.get("cluster_features", CLUSTER_FEATURES)
    profile_feats = [c for c in feats if c in master.columns]

    cp      = master.groupby("CLUSTER_NOM")[profile_feats].mean()
    cp_norm = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)

    sizes = master["CLUSTER_NOM"].value_counts()
    total = len(master)
    action_plan      = art.get("action_plan",    _ACTION_PLAN)
    radar_dims       = art.get("radar_dims",     _RADAR_DIMS)
    _raw_colors      = art.get("cluster_colors", {})
    _cluster_names   = art.get("cluster_names",  {})
    # cluster_colors may use int cluster IDs OR name strings as keys
    if _raw_colors and all(isinstance(k, str) for k in _raw_colors):
        cluster_colors = _raw_colors
    else:
        # id-keyed — invert via cluster_names to get name→color
        cluster_colors = {name: _raw_colors.get(cid, "#64748b")
                          for cid, name in _cluster_names.items()}

    segments = []
    for name in sizes.index:
        count  = int(sizes[name])
        row_n  = cp_norm.loc[name] if name in cp_norm.index else pd.Series(dtype=float)
        row_r  = cp.loc[name]      if name in cp.index      else pd.Series(dtype=float)
        ltv    = float(master.loc[master["CLUSTER_NOM"] == name, "CLV_ESTIME"].mean()) \
                 if "CLV_ESTIME" in master.columns else 0.0
        sp     = float(master.loc[master["CLUSTER_NOM"] == name, "SP_RATIO"].mean()) \
                 if "SP_RATIO" in master.columns else 0.0

        # Radar
        radar_result = []
        for label, col in radar_dims:
            val = float(row_n.get(col, 0.5))
            if label in ("Fiabilité", "Rentabilité"):
                val = 1.0 - val
            radar_result.append({"metric": label, "value": round(max(0.0, min(1.0, val)), 3)})

        # Resolve action
        action = action_plan.get(name)
        if action is None:
            for k, v in action_plan.items():
                if name.startswith(k):
                    action = v
                    break
        action = action or "Surveillance standard"

        color = cluster_colors.get(name, "#64748b")

        segments.append({
            "name":                 name,
            "count":                count,
            "share_pct":            round(count / total * 100, 1),
            "color":                color,
            "avg_prime":            round(float(safe_float(row_r.get("TOTAL_PRIMES",       0))), 0),
            "avg_anciennete_jours": round(float(safe_float(row_r.get("ANCIENNETE_JOURS",   0))), 0),
            "avg_churn_risk":       round(float(safe_float(row_r.get("TAUX_RESILIATION",   0))), 3),
            "avg_taux_impaye":      round(float(safe_float(row_r.get("TAUX_IMPAYE",        0))), 3),
            "avg_sp_ratio":         round(sp, 3),
            "avg_ltv":              round(ltv, 0),
            "action":               action,
            "radar":                radar_result,
        })

    segments.sort(key=lambda s: s["avg_prime"], reverse=True)

    model_metrics = art.get("metrics", {})
    ref_date = art.get("ref_date")
    return {
        "nb_clients":         total,
        "nb_clusters":        art.get("kmeans", {}).n_clusters if hasattr(art.get("kmeans"), "n_clusters") else _K_OPTIMAL,
        "segments":           segments,
        "model_source":       art.get("source", "proxy") + ":KMeans",
        "silhouette":         model_metrics.get("silhouette"),
        "davies_bouldin":     model_metrics.get("davies_bouldin"),
        "calinski_harabasz":  model_metrics.get("calinski_harabasz"),
        "k":                  model_metrics.get("k", _K_OPTIMAL),
        "ref_date":           str(ref_date)[:10] if ref_date is not None else None,
    }
