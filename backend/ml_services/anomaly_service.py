"""
ml_services/anomaly_service.py
──────────────────────────────
Contract-level anomaly detection — anomaly_detection.ipynb

4-algorithm consensus: IF · LOF · Autoencoder (PCA proxy) · DBSCAN
Score 0–4: number of models that flag the contract as anomalous.
High priority: score >= 3.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ._base import (
    load_artifact, load_emission, load_impaye, load_police,
    load_sinistre, load_table, save_artifact,
)

logger = logging.getLogger("maghrebia.ml_services.anomaly")

_ARTIFACT_NAME = "anomaly_model"

FEATURE_COLS = [
    "NB_QUITTANCES", "MT_PNET_TOTAL", "MT_PNET_MOY", "MT_PNET_STD",
    "RATIO_COM_MOY", "TAUX_IMPAYE", "TAUX_ANNULE", "LOSS_RATIO",
    "FREQ_SINISTRE", "MT_IMPAYE_TOT", "MT_EVAL_TOT", "DELAI_MOY",
    "BONUS_MALUS_MOY",
]


def _build_features() -> pd.DataFrame:
    emission = load_emission()
    sinistre = load_sinistre()
    impaye   = load_impaye()
    police   = load_police()

    if emission.empty:
        return pd.DataFrame()

    emission.columns = [c.upper() for c in emission.columns]

    def _col(df, col, default=0.0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    emission["MT_PNET"]     = _col(emission, "MT_PNET", 0.0)
    emission["MT_COM"]      = _col(emission, "MT_COM",  0.0)
    emission["BONUS_MALUS"] = _col(emission, "BONUS_MALUS", 1.0)

    # Contract-level aggregates from emission
    em_pol = emission.groupby("ID_POLICE").agg(
        NB_QUITTANCES   =("NUM_QUITTANCE", "count"),
        MT_PNET_TOTAL   =("MT_PNET",       "sum"),
        MT_PNET_MOY     =("MT_PNET",       "mean"),
        MT_PNET_STD     =("MT_PNET",       "std"),
        BONUS_MALUS_MOY =("BONUS_MALUS",   "mean"),
    ).reset_index()
    em_pol["MT_PNET_STD"] = em_pol["MT_PNET_STD"].fillna(0)

    # Commission ratio
    emission["RATIO_COM"] = emission["MT_COM"] / emission["MT_PNET"].replace(0, np.nan)
    ratio_agg = emission.groupby("ID_POLICE")["RATIO_COM"].mean().fillna(0).reset_index()
    ratio_agg.columns = ["ID_POLICE", "RATIO_COM_MOY"]
    em_pol = em_pol.merge(ratio_agg, on="ID_POLICE", how="left")

    # Annulation rate
    if "ETAT_QUITTANCE" in emission.columns:
        emission["IS_ANNULE"] = (emission["ETAT_QUITTANCE"].astype(str).str.upper() == "ANNULE").astype(int)
        annule_agg = emission.groupby("ID_POLICE").agg(
            NB_ANNULE=("IS_ANNULE",      "sum"),
            NB_TOTAL =("NUM_QUITTANCE",  "count"),
        ).reset_index()
        annule_agg["TAUX_ANNULE"] = annule_agg["NB_ANNULE"] / annule_agg["NB_TOTAL"].replace(0, np.nan)
        em_pol = em_pol.merge(annule_agg[["ID_POLICE", "TAUX_ANNULE"]], on="ID_POLICE", how="left")
    else:
        em_pol["TAUX_ANNULE"] = 0.0

    # Impayé aggregates
    if not impaye.empty:
        impaye.columns = [c.upper() for c in impaye.columns]
        impaye["MT_IMPAYE"] = (
            pd.to_numeric(impaye["MT_IMPAYE"], errors="coerce").fillna(0)
            if "MT_IMPAYE" in impaye.columns else pd.Series(0.0, index=impaye.index)
        )
        imp_pol = impaye.groupby("ID_POLICE").agg(
            NB_IMPAYES   =("NUM_QUITTANCE", "count"),
            MT_IMPAYE_TOT=("MT_IMPAYE",     "sum"),
        ).reset_index()
        em_pol = em_pol.merge(imp_pol, on="ID_POLICE", how="left")
        em_pol["NB_IMPAYES"]    = em_pol["NB_IMPAYES"].fillna(0)
        em_pol["MT_IMPAYE_TOT"] = em_pol["MT_IMPAYE_TOT"].fillna(0)
    else:
        em_pol["NB_IMPAYES"]    = 0
        em_pol["MT_IMPAYE_TOT"] = 0.0

    em_pol["TAUX_IMPAYE"] = em_pol["NB_IMPAYES"] / em_pol["NB_QUITTANCES"].replace(0, np.nan)

    # Sinistre aggregates
    if not sinistre.empty:
        sinistre.columns = [c.upper() for c in sinistre.columns]
        sinistre["MT_EVALUATION"] = pd.to_numeric(sinistre.get("MT_EVALUATION"), errors="coerce").fillna(0)
        if "DATE_DECLARATION" in sinistre.columns and "DATE_SURVENANCE" in sinistre.columns:
            sinistre["DATE_DECLARATION"] = pd.to_datetime(sinistre["DATE_DECLARATION"], errors="coerce")
            sinistre["DATE_SURVENANCE"]  = pd.to_datetime(sinistre["DATE_SURVENANCE"],  errors="coerce")
            sinistre["DELAI"] = (sinistre["DATE_DECLARATION"] - sinistre["DATE_SURVENANCE"]).dt.days.fillna(30)
        else:
            sinistre["DELAI"] = 30.0
        sin_pol = sinistre.groupby("ID_POLICE").agg(
            NB_SINISTRES=("NUM_SINISTRE",  "count"),
            MT_EVAL_TOT =("MT_EVALUATION", "sum"),
            DELAI_MOY   =("DELAI",         "mean"),
        ).reset_index()
        em_pol = em_pol.merge(sin_pol, on="ID_POLICE", how="left")
        em_pol["NB_SINISTRES"] = em_pol["NB_SINISTRES"].fillna(0)
        em_pol["MT_EVAL_TOT"]  = em_pol["MT_EVAL_TOT"].fillna(0)
        em_pol["DELAI_MOY"]    = em_pol["DELAI_MOY"].fillna(30)
    else:
        em_pol["NB_SINISTRES"] = 0
        em_pol["MT_EVAL_TOT"]  = 0.0
        em_pol["DELAI_MOY"]    = 30.0

    em_pol["LOSS_RATIO"]    = em_pol["MT_EVAL_TOT"] / em_pol["MT_PNET_TOTAL"].replace(0, np.nan)
    em_pol["FREQ_SINISTRE"] = em_pol["NB_SINISTRES"]

    # Branche + client from police
    if not police.empty:
        police.columns = [c.upper() for c in police.columns]
        pol_cols = ["ID_POLICE", "BRANCHE"]
        if "ID_CLIENT" in police.columns:
            pol_cols.append("ID_CLIENT")
        pol_slim = police[pol_cols].drop_duplicates("ID_POLICE")
        em_pol = em_pol.merge(pol_slim, on="ID_POLICE", how="left")
    else:
        em_pol["BRANCHE"]   = "—"
        em_pol["ID_CLIENT"] = None

    # Client name / ville
    try:
        clients = load_table(
            "SELECT id_client, nom, prenom, type_personne, ville FROM dim_client",
            "DIM_CLIENT",
        )
        if not clients.empty and "ID_CLIENT" in em_pol.columns:
            clients.columns = [c.upper() for c in clients.columns]
            em_pol = em_pol.merge(clients, on="ID_CLIENT", how="left")
    except Exception:
        pass

    return em_pol.fillna(0)


def _run_models(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    valid = [c for c in FEATURE_COLS if c in df.columns]
    X = df[valid].values
    if len(X) < 20:
        df["IF_ANOMALY"]     = False
        df["LOF_ANOMALY"]    = False
        df["AE_ANOMALY"]     = False
        df["DBSCAN_ANOMALY"] = False
        df["IF_SCORE"]       = 0.0
        df["ANOMALY_SCORE"]  = 0
        return df

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    raw_if = -iso.score_samples(X_scaled)
    df["IF_SCORE"]   = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-9)
    df["IF_ANOMALY"] = iso.predict(X_scaled) == -1

    # LOF
    lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1), contamination=contamination, n_jobs=-1)
    df["LOF_ANOMALY"] = lof.fit_predict(X_scaled) == -1

    # Autoencoder proxy: PCA reconstruction error
    n_comp = min(6, X_scaled.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    X_rec  = pca.inverse_transform(pca.fit_transform(X_scaled))
    recon_err    = np.mean((X_scaled - X_rec) ** 2, axis=1)
    ae_threshold = np.percentile(recon_err, (1 - contamination) * 100)
    df["AE_ANOMALY"] = recon_err > ae_threshold

    # DBSCAN on PCA 2D
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    db_labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(X_2d)
    df["DBSCAN_ANOMALY"] = db_labels == -1

    # Consensus score 0–4
    df["ANOMALY_SCORE"] = (
        df["IF_ANOMALY"].astype(int)    +
        df["LOF_ANOMALY"].astype(int)   +
        df["AE_ANOMALY"].astype(int)    +
        df["DBSCAN_ANOMALY"].astype(int)
    )
    return df


def detect_anomalies(
    departement:    Optional[str] = None,
    contamination:  float         = 0.05,
    nb_mois_recent: int           = 24,
    min_score:      int           = 1,
) -> dict[str, Any]:

    df = _build_features()
    if df.empty:
        return {"error": "Aucune donnée disponible"}

    # Score the FULL dataset first (cache is keyed on all contracts),
    # then apply the departement filter so the size check stays valid.
    use_cache = contamination == 0.05
    art = load_artifact(_ARTIFACT_NAME) if use_cache else None
    cached    = (art or {}).get("score_cache")
    cache_size = (art or {}).get("cache_size", 0)

    if cached and abs(len(df) - cache_size) <= max(50, cache_size * 0.02):
        ids = df["ID_POLICE"].astype(str)
        _def = (False, False, False, False, 0.0)
        df["IF_ANOMALY"]     = [cached.get(k, _def)[0] for k in ids]
        df["LOF_ANOMALY"]    = [cached.get(k, _def)[1] for k in ids]
        df["AE_ANOMALY"]     = [cached.get(k, _def)[2] for k in ids]
        df["DBSCAN_ANOMALY"] = [cached.get(k, _def)[3] for k in ids]
        df["IF_SCORE"]       = [float(cached.get(k, _def)[4]) for k in ids]
        df["ANOMALY_SCORE"]  = (
            df["IF_ANOMALY"].astype(int) + df["LOF_ANOMALY"].astype(int) +
            df["AE_ANOMALY"].astype(int) + df["DBSCAN_ANOMALY"].astype(int)
        )
    else:
        df = _run_models(df, contamination)
        if use_cache:
            score_cache = {
                str(row["ID_POLICE"]): (
                    bool(row["IF_ANOMALY"]), bool(row["LOF_ANOMALY"]),
                    bool(row["AE_ANOMALY"]), bool(row["DBSCAN_ANOMALY"]),
                    float(row["IF_SCORE"]),
                )
                for _, row in df.iterrows()
            }
            save_artifact(_ARTIFACT_NAME, {"score_cache": score_cache, "cache_size": len(df)})

    if departement and "BRANCHE" in df.columns:
        df = df[df["BRANCHE"].astype(str).str.upper() == departement.upper()]
        if df.empty:
            return {"error": f"Aucune donnée pour la branche {departement}"}

    min_score = max(1, min(4, min_score))
    anomalies_df = df[df["ANOMALY_SCORE"] >= min_score].sort_values("ANOMALY_SCORE", ascending=False)

    def _client_name(row) -> str:
        nom    = str(row.get("NOM",    "") or "").strip()
        prenom = str(row.get("PRENOM", "") or "").strip()
        if nom and prenom:
            return f"{prenom} {nom}"
        return nom or prenom or "—"

    records = []
    for _, row in anomalies_df.head(200).iterrows():
        records.append({
            "id_police":      str(row.get("ID_POLICE", "—")),
            "branche":        str(row.get("BRANCHE", "—")),
            "anomaly_score":  int(row["ANOMALY_SCORE"]),
            "if_anomaly":     bool(row["IF_ANOMALY"]),
            "lof_anomaly":    bool(row["LOF_ANOMALY"]),
            "ae_anomaly":     bool(row["AE_ANOMALY"]),
            "dbscan_anomaly": bool(row["DBSCAN_ANOMALY"]),
            "loss_ratio":     round(float(row.get("LOSS_RATIO", 0)), 4),
            "taux_impaye":    round(float(row.get("TAUX_IMPAYE", 0)), 4),
            "nb_sinistres":   int(row.get("NB_SINISTRES", 0)),
            "mt_pnet_total":  round(float(row.get("MT_PNET_TOTAL", 0)), 2),
            "if_score":       round(float(row.get("IF_SCORE", 0)), 4),
            "client_nom":     _client_name(row),
            "client_ville":   str(row.get("VILLE",        "") or "—"),
            "type_personne":  str(row.get("TYPE_PERSONNE","") or "—"),
            "id_client":      str(row.get("ID_CLIENT",    "—")),
        })

    score_counts = df["ANOMALY_SCORE"].value_counts().to_dict()

    return {
        "nb_contracts_analysed": len(df),
        "nb_anomalies":          int((df["ANOMALY_SCORE"] >= 1).sum()),
        "nb_normaux":            int((df["ANOMALY_SCORE"] == 0).sum()),
        "score_4":               int(score_counts.get(4, 0)),
        "score_3":               int(score_counts.get(3, 0)),
        "score_2":               int(score_counts.get(2, 0)),
        "score_1":               int(score_counts.get(1, 0)),
        "contamination":         contamination,
        "min_score":             min_score,
        "departement":           departement or "Tous",
        "anomalies":             records,
        "model_source":          "realtime_4algo",
    }
