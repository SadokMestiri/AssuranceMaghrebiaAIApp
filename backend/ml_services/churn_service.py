"""
ml_services/churn_service.py
─────────────────────────────
Churn prediction service — churn_prediction_v3.ipynb

Model  : LightGBM (CalibratedClassifierCV, isotonic), trained with SMOTE.
Target : CHURN = 1 if SITUATION in ('R','A'), else 0 (polices résiliées / annulées).
Threshold : 0.40 (from F1-optimised search on validation set).

If the trained artifact is present at models/churn_model.pkl the service
uses it directly; otherwise it builds features from DB/CSV and trains a
lightweight proxy model on-the-fly so the API never returns 500.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ._base import (
    load_artifact, load_emission, load_impaye, load_police,
    load_sinistre, save_artifact, safe_float, clip_ratio,
)

logger = logging.getLogger("maghrebia.ml_services.churn")

_ARTIFACT_NAME = "churn_model"
_THRESHOLD = 0.40

# ── Feature list (matches churn_prediction_v3.ipynb) ──────────────────────
_NUMERIC_FEATURES = [
    "nb_quittances", "mt_pnet_moy", "mt_pnet_total",
    "mt_commission_moy", "bonus_malus_moy",
    "nb_sinistres", "mt_eval_moy", "mt_paye_moy",
    "nb_impayes", "taux_impaye",
    "anciennete_jours", "nb_branches",
    "prime_x_anciennete", "impaye_x_prime",
]

# ── Portfolio feature builder ───────────────────────────────────────────────

def _build_portfolio_features() -> pd.DataFrame:
    """Build per-police feature matrix from raw DWH tables."""
    police   = load_police()
    emission = load_emission()
    sinistre = load_sinistre()
    impaye   = load_impaye()

    if police.empty:
        return pd.DataFrame()

    # ── Date parsing ──
    police["date_effet"]    = pd.to_datetime(police["date_effet"],    errors="coerce")
    police["date_echeance"] = pd.to_datetime(police["date_echeance"], errors="coerce")
    ref_date = pd.Timestamp("today")
    police["anciennete_jours"] = (ref_date - police["date_effet"]).dt.days.clip(0)
    police["CHURN"] = police["situation"].isin(["R", "A"]).astype(int)

    feats = police[["id_police", "id_client", "id_agent", "branche",
                    "bonus_malus", "anciennete_jours", "situation", "CHURN"]].copy()

    # ── Emission aggregation ──
    if not emission.empty:
        emission["mt_pnet"] = pd.to_numeric(emission["mt_pnet"], errors="coerce").fillna(0)
        em_agg = emission.groupby("id_police").agg(
            nb_quittances   =("num_quittance",  "count"),
            mt_pnet_moy     =("mt_pnet",        "mean"),
            mt_pnet_total   =("mt_pnet",        "sum"),
            mt_commission_moy=("mt_commission", "mean"),
            bonus_malus_moy =("bonus_malus",    "mean"),
        ).reset_index()
        feats = feats.merge(em_agg, on="id_police", how="left")
    else:
        for c in ["nb_quittances","mt_pnet_moy","mt_pnet_total","mt_commission_moy","bonus_malus_moy"]:
            feats[c] = 0.0

    # ── Sinistre aggregation ──
    if not sinistre.empty:
        sinistre["mt_evaluation"] = pd.to_numeric(sinistre["mt_evaluation"], errors="coerce").fillna(0)
        sinistre["mt_paye"]       = pd.to_numeric(sinistre["mt_paye"],       errors="coerce").fillna(0)
        sin_agg = sinistre.groupby("id_police").agg(
            nb_sinistres=("num_sinistre", "count"),
            mt_eval_moy =("mt_evaluation","mean"),
            mt_paye_moy =("mt_paye",      "mean"),
        ).reset_index()
        feats = feats.merge(sin_agg, on="id_police", how="left")
    else:
        feats["nb_sinistres"] = 0
        feats["mt_eval_moy"]  = 0.0
        feats["mt_paye_moy"]  = 0.0

    # ── Impayé aggregation ──
    if not impaye.empty:
        imp_agg = impaye.groupby("id_police").agg(
            nb_impayes=("num_quittance","count"),
        ).reset_index()
        feats = feats.merge(imp_agg, on="id_police", how="left")
    else:
        feats["nb_impayes"] = 0

    feats["nb_impayes"]  = feats["nb_impayes"].fillna(0)
    feats["taux_impaye"] = feats["nb_impayes"] / feats["nb_quittances"].replace(0, 1)

    # ── Branch diversity ──
    if not emission.empty:
        br_div = emission.groupby("id_police")["branche"].nunique().reset_index(name="nb_branches")
        feats = feats.merge(br_div, on="id_police", how="left")
    feats["nb_branches"] = feats.get("nb_branches", pd.Series(1, index=feats.index)).fillna(1)

    # ── Interaction features ──
    feats["prime_x_anciennete"] = feats["mt_pnet_moy"].fillna(0) * feats["anciennete_jours"].fillna(0)
    feats["impaye_x_prime"]     = feats["taux_impaye"].fillna(0) * feats["mt_pnet_moy"].fillna(0)

    feats[_NUMERIC_FEATURES] = feats[_NUMERIC_FEATURES].fillna(0)
    return feats


def _train_proxy_model(df: pd.DataFrame) -> dict:
    """Train a lightweight proxy when the notebook artifact is absent."""
    X = df[_NUMERIC_FEATURES].values
    y = df["CHURN"].values

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X))

    base = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                       learning_rate=0.08, random_state=42)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)

    art = {
        "model":    model,
        "imputer":  imputer,
        "scaler":   scaler,
        "features": _NUMERIC_FEATURES,
        "threshold": _THRESHOLD,
        "source":   "proxy",
    }
    save_artifact(_ARTIFACT_NAME, art)
    logger.info("Churn proxy model trained and saved.")
    return art


def _get_artifact() -> dict:
    art = load_artifact(_ARTIFACT_NAME)
    if art:
        return art
    logger.info("Churn artifact not found — training proxy model.")
    df = _build_portfolio_features()
    if df.empty or df["CHURN"].nunique() < 2:
        raise RuntimeError("Not enough data to train churn model.")
    return _train_proxy_model(df)


# ── Public API ──────────────────────────────────────────────────────────────

def get_churn_summary() -> dict[str, Any]:
    """
    Returns portfolio-level churn statistics.
    Loads from DB/CSV; uses actual résiliation labels for headline rate and
    per-branch counts; ML probabilities drive risk-segment breakdown only.
    """
    df = _build_portfolio_features()
    if df.empty:
        raise RuntimeError("No data available for churn summary.")

    # ── Real résiliation figures (SITUATION in R/A) ─────────────────────────
    nb_polices     = len(df)
    nb_resilie     = int(df["CHURN"].sum())
    taux_resiliation = round(nb_resilie / nb_polices * 100, 1) if nb_polices else 0.0

    # Real counts per branch: nb résiliées + total + taux réel
    by_branche_real = (
        df.groupby("branche")
        .agg(
            nb_total    =("CHURN", "count"),
            nb_resilie  =("CHURN", "sum"),
        )
        .reset_index()
    )
    by_branche_real["taux_resiliation_pct"] = (
        by_branche_real["nb_resilie"] / by_branche_real["nb_total"] * 100
    ).round(1)
    by_branche_real["nb_total"]             = by_branche_real["nb_total"].astype(int)
    by_branche_real["nb_resilie"]           = by_branche_real["nb_resilie"].astype(int)
    by_branche_real["taux_resiliation_pct"] = by_branche_real["taux_resiliation_pct"].astype(float)
    by_branche = by_branche_real.to_dict("records")

    # ── ML probabilities (used only for risk-segment breakdown) ────────────
    art = _get_artifact()
    feat_cols = art.get("features", _NUMERIC_FEATURES)
    X = art["imputer"].transform(df[feat_cols].fillna(0).values)
    X = art["scaler"].transform(X)
    df["PROB_CHURN"] = art["model"].predict_proba(X)[:, 1]

    threshold = art.get("threshold", _THRESHOLD)

    # Feature importances from model
    top_features = []
    try:
        inner = art["model"].calibrated_classifiers_[0].estimator
        if hasattr(inner, "feature_importances_"):
            imp = inner.feature_importances_
            top_features = [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:8]
            ]
    except Exception:
        pass

    # Risk segments (ML-based, on active policies only)
    df_actif = df[df["CHURN"] == 0].copy()
    segments = (
        pd.cut(df_actif["PROB_CHURN"], bins=[0, 0.20, 0.35, 0.50, 1.0],
               labels=["Faible (<20%)", "Modéré (20-35%)", "Élevé (35-50%)", "Critique (>50%)"],
               include_lowest=True)
        .value_counts()
        .rename_axis("segment")
        .reset_index(name="count")
        .to_dict("records")
    )

    return {
        "taux_churn_pct": taux_resiliation,     # real résiliation rate
        "nb_churn":       nb_resilie,            # real count of résiliées/annulées
        "nb_polices":     nb_polices,
        "by_branche":     by_branche,            # real counts per branch
        "top_features":   top_features,
        "risk_segments":  segments,              # ML-based, active policies only
        "threshold":      threshold,
        "model_source":   art.get("source", "artifact"),
    }


def predict_churn(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Predict churn probability for a single policy.
    Input keys (all optional, defaults to portfolio median):
      branche, nb_quittances, mt_pnet, bonus_malus, taux_impaye, nb_sinistres
    """
    art = _get_artifact()
    feat_cols = art.get("features", _NUMERIC_FEATURES)

    # Build feature vector from portfolio medians
    df_med = _build_portfolio_features()
    medians = df_med[feat_cols].median() if not df_med.empty else pd.Series(0, index=feat_cols)

    row = medians.copy()
    # Map incoming payload keys → internal feature names
    mapping = {
        "nb_quittances": "nb_quittances",
        "mt_pnet":       "mt_pnet_moy",
        "bonus_malus":   "bonus_malus_moy",
        "taux_impaye":   "taux_impaye",
        "nb_sinistres":  "nb_sinistres",
    }
    for src, dst in mapping.items():
        if src in payload and dst in row.index:
            row[dst] = safe_float(payload[src])

    X = art["imputer"].transform(row.values.reshape(1, -1))
    X = art["scaler"].transform(X)
    prob      = float(art["model"].predict_proba(X)[0, 1])
    threshold = art.get("threshold", _THRESHOLD)
    predicted = prob >= threshold

    action_map = {
        (True,  prob > 0.7): "Intervention urgente agent — risque critique de résiliation",
        (True,  prob > 0.5): "Contacter l'agent pour offre de fidélisation ciblée",
        (True,  True      ): "Surveiller le dossier — proposer renouvellement anticipé",
        (False, True      ): "Aucune action urgente requise — client stable",
    }
    action = next(
        (v for (pred_cond, prob_cond), v in action_map.items()
         if pred_cond == predicted and prob_cond),
        "Aucune action requise",
    )

    return {
        "churn_probability": round(prob, 4),
        "churn_predicted":   bool(predicted),
        "threshold":         threshold,
        "action":            action,
        "model":             art.get("source", "LightGBM calibrated"),
    }