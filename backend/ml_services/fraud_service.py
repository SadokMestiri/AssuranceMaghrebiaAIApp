"""
ml_services/fraud_service.py
─────────────────────────────
Fraud detection service — fraud_detection.ipynb

Ensemble scoring: IF (40%) + Autoencoder (40%) + LOF (20%).
Risk levels (thresholds are percentile-based, stored in the artifact):
  Normal       < p90
  Risque Modéré  p90–p95
  Risque Élevé   p95–p99
  Critique       > p99

When the trained artifact is absent, the service builds features from
DB/CSV and computes the ensemble score using sklearn proxies (no Keras).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ._base import (
    load_artifact, load_client, load_emission, load_impaye,
    load_police, load_sinistre, save_artifact, safe_float,
)

logger = logging.getLogger("maghrebia.ml_services.fraud")

_ARTIFACT_NAME = "fraud_model"

FEATURE_COLS = [
    # Sinistre
    "MT_EVALUATION", "MT_PAYE", "RATIO_PAYE_EVAL", "MT_EVAL_ZSCORE",
    "DELAI_DECLARATION", "RESPONSABILITE",
    # Client aggregates
    "CLIENT_NB_SINISTRES", "CLIENT_MOY_MONTANT", "CLIENT_MAX_MONTANT",
    "CLIENT_STD_MONTANT",
    # Agent aggregates
    "AGENT_NB_SINISTRES", "AGENT_MOY_MONTANT", "AGENT_TAUX_OUVERT",
    # Police aggregates
    "POLICE_NB_QUITTANCES", "POLICE_MT_PNET_MOY", "POLICE_BONUS_MALUS_MOY",
    "JOURS_DEPUIS_EFFET", "FLAG_SINISTRE_PRECOCE", "FLAG_SINISTRE_FIN_CONTRAT",
    "POLICE_NB_IMPAYES",
]


# ── Feature builder ─────────────────────────────────────────────────────────

def _build_fraud_features() -> pd.DataFrame:
    sinistre = load_sinistre()
    if sinistre.empty:
        return pd.DataFrame()

    emission = load_emission()
    impaye   = load_impaye()
    police   = load_police()

    df = sinistre.copy()
    df.columns = [c.upper() for c in df.columns]

    # ── Date parsing ──
    df["DATE_SURVENANCE"]  = pd.to_datetime(df.get("DATE_SURVENANCE"),  errors="coerce")
    # DATE_DECLARATION may not exist in all table versions
    if "DATE_DECLARATION" in df.columns:
        df["DATE_DECLARATION"] = pd.to_datetime(df["DATE_DECLARATION"], errors="coerce")
        delai_raw = (df["DATE_DECLARATION"] - df["DATE_SURVENANCE"]).dt.days
        df["DELAI_DECLARATION"] = pd.to_numeric(delai_raw, errors="coerce").fillna(30)
    else:
        df["DELAI_DECLARATION"] = 30.0
    df["MT_EVALUATION"]    = pd.to_numeric(df.get("MT_EVALUATION"), errors="coerce").fillna(0)
    df["MT_PAYE"]          = pd.to_numeric(df.get("MT_PAYE"),       errors="coerce").fillna(0)
    df["RESPONSABILITE"]   = pd.to_numeric(df.get("RESPONSABILITE"),errors="coerce").fillna(0)

    df["RATIO_PAYE_EVAL"]   = df["MT_PAYE"] / df["MT_EVALUATION"].replace(0, np.nan)
    df["RATIO_PAYE_EVAL"]   = df["RATIO_PAYE_EVAL"].fillna(0).clip(0, 5)

    # Z-score of evaluation amount
    mu  = df["MT_EVALUATION"].mean()
    std = df["MT_EVALUATION"].std() + 1e-9
    df["MT_EVAL_ZSCORE"] = ((df["MT_EVALUATION"] - mu) / std).clip(-5, 5)

    # ── Client aggregates ──
    client_agg = df.groupby("ID_CLIENT").agg(
        CLIENT_NB_SINISTRES=("NUM_SINISTRE",  "count"),
        CLIENT_MOY_MONTANT =("MT_EVALUATION", "mean"),
        CLIENT_MAX_MONTANT =("MT_EVALUATION", "max"),
        CLIENT_STD_MONTANT =("MT_EVALUATION", "std"),
    ).fillna(0).reset_index()
    df = df.merge(client_agg, on="ID_CLIENT", how="left")

    # ── Agent aggregates ──
    if "ID_AGENT" in df.columns:
        agent_agg = df.groupby("ID_AGENT").agg(
            AGENT_NB_SINISTRES=("NUM_SINISTRE",  "count"),
            AGENT_MOY_MONTANT =("MT_EVALUATION", "mean"),
            AGENT_TAUX_OUVERT =("ETAT_SINISTRE", lambda x: (x == "OUVERT").mean()),
        ).fillna(0).reset_index()
        df = df.merge(agent_agg, on="ID_AGENT", how="left")
    else:
        for c in ["AGENT_NB_SINISTRES", "AGENT_MOY_MONTANT", "AGENT_TAUX_OUVERT"]:
            df[c] = 0.0

    # ── Police aggregates ──
    if not police.empty:
        police.columns = [c.upper() for c in police.columns]
        police["DATE_EFFET"]    = pd.to_datetime(police.get("DATE_EFFET"),    errors="coerce")
        police["DATE_ECHEANCE"] = pd.to_datetime(police.get("DATE_ECHEANCE"), errors="coerce")
        police["BONUS_MALUS"]   = pd.to_numeric(police.get("BONUS_MALUS"),    errors="coerce").fillna(1.0)

        if not emission.empty:
            emission.columns = [c.upper() for c in emission.columns]
            emission["MT_PNET"] = pd.to_numeric(emission.get("MT_PNET"), errors="coerce").fillna(0)
            pol_em = emission.groupby("ID_POLICE").agg(
                POLICE_NB_QUITTANCES=("NUM_QUITTANCE","count"),
                POLICE_MT_PNET_MOY  =("MT_PNET",      "mean"),
                POLICE_BONUS_MALUS_MOY=("BONUS_MALUS", "mean"),
            ).reset_index()
            police = police.merge(pol_em, on="ID_POLICE", how="left")
        else:
            police["POLICE_NB_QUITTANCES"]   = 1
            police["POLICE_MT_PNET_MOY"]     = 1000.0
            police["POLICE_BONUS_MALUS_MOY"] = 1.0

        if not impaye.empty:
            impaye.columns = [c.upper() for c in impaye.columns]
            pol_imp = impaye.groupby("ID_POLICE")["NUM_QUITTANCE"].count().reset_index(name="POLICE_NB_IMPAYES")
            police = police.merge(pol_imp, on="ID_POLICE", how="left")
        police["POLICE_NB_IMPAYES"] = police.get("POLICE_NB_IMPAYES", 0).fillna(0)

        df = df.merge(
            police[["ID_POLICE", "DATE_EFFET", "DATE_ECHEANCE",
                    "POLICE_NB_QUITTANCES", "POLICE_MT_PNET_MOY",
                    "POLICE_BONUS_MALUS_MOY", "POLICE_NB_IMPAYES"]],
            on="ID_POLICE", how="left",
        )

        df["JOURS_DEPUIS_EFFET"] = (df["DATE_SURVENANCE"] - df["DATE_EFFET"]).dt.days.fillna(180)
        df["FLAG_SINISTRE_PRECOCE"]      = (df["JOURS_DEPUIS_EFFET"] < 30).astype(int)
        df["FLAG_SINISTRE_FIN_CONTRAT"]  = (
            (df["DATE_ECHEANCE"] - df["DATE_SURVENANCE"]).dt.days.fillna(365) < 30
        ).astype(int)
    else:
        for c in ["POLICE_NB_QUITTANCES","POLICE_MT_PNET_MOY","POLICE_BONUS_MALUS_MOY",
                  "POLICE_NB_IMPAYES","JOURS_DEPUIS_EFFET",
                  "FLAG_SINISTRE_PRECOCE","FLAG_SINISTRE_FIN_CONTRAT"]:
            df[c] = 0

    valid_cols = [c for c in FEATURE_COLS if c in df.columns]
    df[valid_cols] = df[valid_cols].fillna(0)
    return df


# ── Ensemble scorer ─────────────────────────────────────────────────────────

def _compute_ensemble_scores(df: pd.DataFrame, scaler: StandardScaler | None = None
                              ) -> tuple[np.ndarray, StandardScaler]:
    valid_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[valid_cols].fillna(0).values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Isolation Forest score → normalised [0,1]
    iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    raw_if  = -iso.score_samples(X_scaled)                   # higher = more anomalous
    s_if    = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-9)

    # LOF score → normalised [0,1]
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
    lof.fit(X_scaled)
    raw_lof = -lof.negative_outlier_factor_
    s_lof   = (raw_lof - raw_lof.min()) / (raw_lof.max() - raw_lof.min() + 1e-9)

    # Autoencoder proxy via reconstruction error with PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(8, X_scaled.shape[1]), random_state=42)
    X_reduced    = pca.fit_transform(X_scaled)
    X_recon      = pca.inverse_transform(X_reduced)
    recon_errors = np.mean((X_scaled - X_recon) ** 2, axis=1)
    s_ae = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-9)

    # Weighted ensemble: IF 40% + AE 40% + LOF 20%
    fraud_score = 0.40 * s_if + 0.40 * s_ae + 0.20 * s_lof
    return fraud_score, scaler


def _build_artifact() -> dict:
    df = _build_fraud_features()
    if df.empty:
        raise RuntimeError("No sinistre data available for fraud model.")

    fraud_score, scaler = _compute_ensemble_scores(df)
    df["FRAUD_SCORE"] = fraud_score

    p90 = float(np.percentile(fraud_score, 90))
    p95 = float(np.percentile(fraud_score, 95))
    p99 = float(np.percentile(fraud_score, 99))

    art = {
        "scaler":   scaler,
        "features": [c for c in FEATURE_COLS if c in df.columns],
        "p90": p90, "p95": p95, "p99": p99,
        "source": "proxy",
    }
    save_artifact(_ARTIFACT_NAME, art)
    return art, df


def _get_artifact_and_df() -> tuple[dict, pd.DataFrame]:
    art = load_artifact(_ARTIFACT_NAME)
    df  = _build_fraud_features()
    if df.empty:
        raise RuntimeError("No data for fraud service.")
    if art is None:
        art, df = _build_artifact()
        return art, df

    fraud_score, _ = _compute_ensemble_scores(df, art["scaler"])
    df["FRAUD_SCORE"] = fraud_score
    return art, df


def _risk_label(score: float, p90: float, p95: float, p99: float) -> str:
    if score >= p99:
        return "Critique"
    if score >= p95:
        return "Risque Élevé"
    if score >= p90:
        return "Risque Modéré"
    return "Normal"


# ── Public API ──────────────────────────────────────────────────────────────

def get_fraud_summary(top_n: int = 10) -> dict[str, Any]:
    art, df = _get_artifact_and_df()
    p90, p95, p99 = art["p90"], art["p95"], art["p99"]

    df["RISK_LEVEL"] = df["FRAUD_SCORE"].apply(lambda s: _risk_label(s, p90, p95, p99))

    counts = df["RISK_LEVEL"].value_counts().to_dict()

    # Build score distribution (10 bins)
    hist, edges = np.histogram(df["FRAUD_SCORE"], bins=10, range=(0, 1))
    score_dist = [
        {"bin": f"{edges[i]:.1f}–{edges[i+1]:.1f}", "count": int(hist[i])}
        for i in range(len(hist))
    ]

    # Top suspects
    top_df = df[df["RISK_LEVEL"].isin(["Critique", "Risque Élevé"])].copy()
    top_df = top_df.sort_values("FRAUD_SCORE", ascending=False).head(top_n)
    top_records = []
    for _, row in top_df.iterrows():
        top_records.append({
            "num_sinistre":  str(row.get("NUM_SINISTRE", "—")),
            "branche":       str(row.get("BRANCHE", "—")),
            "nature_sinistre": str(row.get("NATURE_SINISTRE", "—")),
            "mt_evaluation": float(row.get("MT_EVALUATION", 0)),
            "fraud_score":   round(float(row["FRAUD_SCORE"]), 4),
            "risk_level":    str(row["RISK_LEVEL"]),
        })

    return {
        "nb_critique": counts.get("Critique",      0),
        "nb_eleve":    counts.get("Risque Élevé",  0),
        "nb_modere":   counts.get("Risque Modéré", 0),
        "nb_normal":   counts.get("Normal",        0),
        "thresholds":  {"p90": round(p90, 4), "p95": round(p95, 4), "p99": round(p99, 4)},
        "top_fraud":         top_records,
        "score_distribution": score_dist,
        "model_source":       art.get("source", "artifact"),
    }