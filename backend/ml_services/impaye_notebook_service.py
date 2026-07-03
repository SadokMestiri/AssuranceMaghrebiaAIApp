"""
ml_services/impaye_notebook_service.py
───────────────────────────────────────
Service impayé connecté au modèle issu du notebook impaye_risk_scoring.ipynb.

Modele retenu  : LightGBM calibre (CalibratedClassifierCV isotonic)
Seuil optimal  : 0.150 (optimise cout metier FN=10xFP)
Metriques test : AUC=0.9923 | AP=0.7178
Metriques seuil 0.150 : Accuracy=98.6% | F1=67.9% | Prec=52.5% | Recall=96.0%

Features (29) construites depuis les tables DWH :
  LOG_MT_PTT, PERIODICITE_RISK, MOIS_EMISSION, TRIMESTRE_EMISSION,
  JOUR_SEMAINE, FLAG_FIN_MOIS, FLAG_AVN, MOIS_ECHEANCE, BONUS_MALUS,
  ID_BRANCHE, BRANCHE_ENC, TYPE_POLICE_ENC,
  POLICE_NB_IMPAYES_HIST, POLICE_TAUX_IMPAYE_HIST, POLICE_SEQ_QUIT,
  POLICE_NB_ANNULATIONS, POLICE_NB_SINISTRES, POLICE_SP_RATIO,
  CLIENT_NB_IMPAYES, CLIENT_MT_IMPAYE_TOT, AGE_CLIENT,
  TYPE_PERSONNE_ENC, SEXE_ENC, PUISSANCE, LOG_VALEUR_VEH, CODE_USAGE,
  MT_COMMISSION, MT_FGA, MT_TIMBRE
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._base import (
    load_emission, load_impaye, load_police,
    load_sinistre, load_client, load_vehicule,
    safe_float,
)

logger = logging.getLogger("maghrebia.ml_services.impaye_notebook")

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PKL_NAME = "impaye_risk_model.pkl"  # fichier sauvegardé par le notebook Cell 34

# ── Mappings encodage (reproduire les LabelEncoders du notebook) ─────────────
_BRANCHE_ENC = {"AUTO": 0, "IRDS": 1, "SANTE": 2}
_TYPE_POLICE_ENC = {"I": 0, "E": 1}          # individuelle / entreprise
_TYPE_PERSONNE_ENC = {"P": 0, "M": 1, "U": 0}
_SEXE_ENC = {"M": 0, "F": 1, "U": 0, "": 0}
_PERIODICITE_RISK = {"A": 1, "S": 2, "T": 3, "C": 4}

FEATURE_COLS = [
    "LOG_MT_PTT", "PERIODICITE_RISK", "MOIS_EMISSION", "TRIMESTRE_EMISSION",
    "JOUR_SEMAINE", "FLAG_FIN_MOIS", "FLAG_AVN", "MOIS_ECHEANCE",
    "BONUS_MALUS", "ID_BRANCHE", "BRANCHE_ENC", "TYPE_POLICE_ENC",
    "POLICE_NB_IMPAYES_HIST", "POLICE_TAUX_IMPAYE_HIST", "POLICE_SEQ_QUIT",
    "POLICE_NB_ANNULATIONS", "POLICE_NB_SINISTRES", "POLICE_SP_RATIO",
    "CLIENT_NB_IMPAYES", "CLIENT_MT_IMPAYE_TOT",
    "AGE_CLIENT", "TYPE_PERSONNE_ENC", "SEXE_ENC",
    "PUISSANCE", "LOG_VALEUR_VEH", "CODE_USAGE",
    "MT_COMMISSION", "MT_FGA", "MT_TIMBRE",
]

# Médianes de repli calculées depuis les sorties du notebook (valeurs typiques)
_DEFAULT_MEDIANS: dict[str, float] = {
    "LOG_MT_PTT": 6.0,          # ~exp(6)=403 TND
    "PERIODICITE_RISK": 1.0,    # annuel
    "MOIS_EMISSION": 6.0,
    "TRIMESTRE_EMISSION": 2.0,
    "JOUR_SEMAINE": 2.0,
    "FLAG_FIN_MOIS": 0.0,
    "FLAG_AVN": 0.0,
    "MOIS_ECHEANCE": 6.0,
    "BONUS_MALUS": 1.0,
    "ID_BRANCHE": 1.0,
    "BRANCHE_ENC": 0.0,
    "TYPE_POLICE_ENC": 0.0,
    "POLICE_NB_IMPAYES_HIST": 0.0,
    "POLICE_TAUX_IMPAYE_HIST": 0.0,
    "POLICE_SEQ_QUIT": 1.0,
    "POLICE_NB_ANNULATIONS": 0.0,
    "POLICE_NB_SINISTRES": 0.0,
    "POLICE_SP_RATIO": 0.0,
    "CLIENT_NB_IMPAYES": 0.0,
    "CLIENT_MT_IMPAYE_TOT": 0.0,
    "AGE_CLIENT": 40.0,
    "TYPE_PERSONNE_ENC": 0.0,
    "SEXE_ENC": 0.0,
    "PUISSANCE": 6.0,
    "LOG_VALEUR_VEH": 9.0,      # ~exp(9)=8000 TND
    "CODE_USAGE": 0.0,
    "MT_COMMISSION": 150.0,
    "MT_FGA": 0.0,
    "MT_TIMBRE": 0.0,
}


# ── Artifact loader ──────────────────────────────────────────────────────────

_cached_artifact: dict | None = None
_artifact_load_error: str | None = None


def _load_artifact() -> dict | None:
    global _cached_artifact, _artifact_load_error
    if _cached_artifact is not None:
        return _cached_artifact

    pkl_path = _MODELS_DIR / _PKL_NAME
    if not pkl_path.exists():
        _artifact_load_error = f"Fichier {_PKL_NAME} introuvable dans {_MODELS_DIR}"
        logger.warning("Notebook pkl not found at %s", pkl_path)
        return None

    try:
        import pickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # ignore sklearn version mismatch
            with open(pkl_path, "rb") as f:
                art = pickle.load(f)
        _cached_artifact = art
        _artifact_load_error = None
        logger.info("Notebook impayé model loaded from %s", pkl_path)
        return art
    except Exception as exc:
        _artifact_load_error = f"Impossible de charger {_PKL_NAME}: {exc}"
        logger.error("Cannot load notebook pkl: %s", exc)
        return None


def get_artifact_info() -> dict[str, Any]:
    """Return metadata about the loaded notebook artifact."""
    art = _load_artifact()
    if art is None:
        return {"status": "unavailable", "error": _artifact_load_error or f"Fichier {_PKL_NAME} indisponible"}

    feature_cols = art.get("feature_cols", FEATURE_COLS)
    return {
        "status": "ready",
        "source": "notebook_impaye_risk_scoring",
        "model": "LightGBM calibré (isotonic)",
        "seuil_optimal": art.get("seuil_optimal", 0.15),
        "nb_features": len(feature_cols),
        "feature_cols": feature_cols,
        "metrics": {
            "AUC":             0.9923,
            "AP_PRAUC":        0.7178,
            "Accuracy":        0.986,
            "Precision":       0.525,
            "Recall":          0.960,
            "F1":              0.679,
            "metrics_threshold": 0.150,
        },
    }


# ── Portfolio stats loader ───────────────────────────────────────────────────

def _build_portfolio_medians() -> dict[str, float]:
    """
    Compute per-feature medians from the DWH for use as imputation defaults
    when the simulator provides only a subset of features.
    Returns default medians if DB is unavailable.
    """
    try:
        emission = load_emission()
        if emission.empty:
            return _DEFAULT_MEDIANS.copy()

        emission.columns = [c.upper() for c in emission.columns]
        medians: dict[str, float] = {}

        # Primitive features from emission table
        if "MT_PTT" in emission.columns:
            mt_ptt = pd.to_numeric(emission["MT_PTT"], errors="coerce").clip(0)
            medians["LOG_MT_PTT"] = float(np.log1p(mt_ptt.median()))
        if "MT_PNET" in emission.columns and "LOG_MT_PTT" not in medians:
            mt_pnet = pd.to_numeric(emission["MT_PNET"], errors="coerce").clip(0)
            medians["LOG_MT_PTT"] = float(np.log1p(mt_pnet.median()))
        if "MT_COMMISSION" in emission.columns:
            medians["MT_COMMISSION"] = float(pd.to_numeric(emission["MT_COMMISSION"], errors="coerce").median())
        if "MT_FGA" in emission.columns:
            medians["MT_FGA"] = float(pd.to_numeric(emission["MT_FGA"], errors="coerce").fillna(0).median())
        if "MT_TIMBRE" in emission.columns:
            medians["MT_TIMBRE"] = float(pd.to_numeric(emission["MT_TIMBRE"], errors="coerce").fillna(0).median())
        if "BONUS_MALUS" in emission.columns:
            medians["BONUS_MALUS"] = float(pd.to_numeric(emission["BONUS_MALUS"], errors="coerce").fillna(1.0).median())
        if "MOIS_ECHEANCE" in emission.columns:
            medians["MOIS_ECHEANCE"] = float(pd.to_numeric(emission["MOIS_ECHEANCE"], errors="coerce").median())

        # Fill remaining with defaults
        for k, v in _DEFAULT_MEDIANS.items():
            medians.setdefault(k, v)

        return medians
    except Exception as exc:
        logger.warning("Could not compute portfolio medians from DB: %s — using defaults", exc)
        return _DEFAULT_MEDIANS.copy()


# ── Feature vector builder for inference ─────────────────────────────────────

def _build_inference_vector(payload: dict[str, Any], medians: dict[str, float]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from a user-supplied payload.
    All missing features are filled from portfolio medians.

    Payload keys (all optional):
      mt_ptt         : float  — prime totale en TND
      periodicite    : str    — 'A'|'S'|'T'|'C'
      mois_emission  : int    — mois d'émission (1-12)
      mois_echeance  : int    — mois d'échéance (1-12)
      flag_avn       : int    — 0/1 (avenant ?)
      bonus_malus    : float  — ex: 1.0, 1.25
      branche        : str    — 'AUTO'|'IRDS'|'SANTE'
      type_police    : str    — 'I'|'E'
      police_nb_impayes_hist  : int
      police_taux_impaye_hist : float
      police_seq_quit         : int
      police_nb_annulations   : int
      police_nb_sinistres     : int
      police_sp_ratio         : float
      client_nb_impayes       : int
      client_mt_impaye_tot    : float
      age_client              : float
      type_personne           : str    — 'P'|'M'
      sexe                    : str    — 'M'|'F'
      puissance               : float  — puissance véhicule
      valeur_vehicule         : float  — valeur actuelle
      code_usage              : int    — ex: 0,1,2
      mt_commission           : float
      mt_fga                  : float
      mt_timbre               : float
    """
    import datetime
    now = datetime.date.today()
    mois_em = int(payload.get("mois_emission", medians.get("MOIS_EMISSION", 6)))

    branche = str(payload.get("branche", "AUTO")).strip().upper()
    periodicite = str(payload.get("periodicite", "A")).strip().upper()
    type_police = str(payload.get("type_police", "I")).strip().upper()
    type_personne = str(payload.get("type_personne", "P")).strip().upper()
    sexe = str(payload.get("sexe", "M")).strip().upper()

    mt_ptt = safe_float(payload.get("mt_ptt", medians.get("LOG_MT_PTT", 6.0)))
    # If user passes mt_ptt as actual value, log-transform it
    # If it's already a log, keep it — we detect by magnitude
    log_mt_ptt = float(np.log1p(max(0.0, mt_ptt))) if mt_ptt > 10 else mt_ptt

    valeur_veh = safe_float(payload.get("valeur_vehicule", 0.0))
    log_valeur_veh = float(np.log1p(max(0.0, valeur_veh))) if valeur_veh > 0 else float(medians.get("LOG_VALEUR_VEH", 9.0))

    row = {
        "LOG_MT_PTT":               log_mt_ptt,
        "PERIODICITE_RISK":         float(_PERIODICITE_RISK.get(periodicite, 1)),
        "MOIS_EMISSION":            float(mois_em),
        "TRIMESTRE_EMISSION":       float((mois_em - 1) // 3 + 1),
        "JOUR_SEMAINE":             float(medians.get("JOUR_SEMAINE", 2.0)),
        "FLAG_FIN_MOIS":            float(payload.get("flag_fin_mois", medians.get("FLAG_FIN_MOIS", 0.0))),
        "FLAG_AVN":                 float(int(payload.get("flag_avn", 0))),
        "MOIS_ECHEANCE":            float(int(payload.get("mois_echeance", medians.get("MOIS_ECHEANCE", 6)))),
        "BONUS_MALUS":              safe_float(payload.get("bonus_malus", medians.get("BONUS_MALUS", 1.0))),
        "ID_BRANCHE":               float(_BRANCHE_ENC.get(branche, 0) + 1),
        "BRANCHE_ENC":              float(_BRANCHE_ENC.get(branche, 0)),
        "TYPE_POLICE_ENC":          float(_TYPE_POLICE_ENC.get(type_police, 0)),
        "POLICE_NB_IMPAYES_HIST":   safe_float(payload.get("police_nb_impayes_hist", medians.get("POLICE_NB_IMPAYES_HIST", 0.0))),
        "POLICE_TAUX_IMPAYE_HIST":  safe_float(payload.get("police_taux_impaye_hist", medians.get("POLICE_TAUX_IMPAYE_HIST", 0.0))),
        "POLICE_SEQ_QUIT":          safe_float(payload.get("police_seq_quit", medians.get("POLICE_SEQ_QUIT", 1.0))),
        "POLICE_NB_ANNULATIONS":    safe_float(payload.get("police_nb_annulations", medians.get("POLICE_NB_ANNULATIONS", 0.0))),
        "POLICE_NB_SINISTRES":      safe_float(payload.get("police_nb_sinistres", medians.get("POLICE_NB_SINISTRES", 0.0))),
        "POLICE_SP_RATIO":          safe_float(payload.get("police_sp_ratio", medians.get("POLICE_SP_RATIO", 0.0))),
        "CLIENT_NB_IMPAYES":        safe_float(payload.get("client_nb_impayes", medians.get("CLIENT_NB_IMPAYES", 0.0))),
        "CLIENT_MT_IMPAYE_TOT":     safe_float(payload.get("client_mt_impaye_tot", medians.get("CLIENT_MT_IMPAYE_TOT", 0.0))),
        "AGE_CLIENT":               safe_float(payload.get("age_client", medians.get("AGE_CLIENT", 40.0))),
        "TYPE_PERSONNE_ENC":        float(_TYPE_PERSONNE_ENC.get(type_personne, 0)),
        "SEXE_ENC":                 float(_SEXE_ENC.get(sexe, 0)),
        "PUISSANCE":                safe_float(payload.get("puissance", medians.get("PUISSANCE", 6.0))),
        "LOG_VALEUR_VEH":           log_valeur_veh,
        "CODE_USAGE":               safe_float(payload.get("code_usage", medians.get("CODE_USAGE", 0.0))),
        "MT_COMMISSION":            safe_float(payload.get("mt_commission", medians.get("MT_COMMISSION", 150.0))),
        "MT_FGA":                   safe_float(payload.get("mt_fga", medians.get("MT_FGA", 0.0))),
        "MT_TIMBRE":                safe_float(payload.get("mt_timbre", medians.get("MT_TIMBRE", 0.0))),
    }

    return pd.DataFrame([row])


# ── Public API ───────────────────────────────────────────────────────────────

def predict_impaye_notebook(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Predict impayé probability using the LightGBM calibrated model from the notebook.

    Returns:
      probability_impaye  : float [0, 1]
      predicted_label     : int   0 = Payée / 1 = Impayée
      threshold           : float (seuil optimal coût-métier)
      risk_level          : str   Faible / Modéré / Élevé / Critique
      model               : str
      feature_importance  : dict (top features from lgb_tuned)
      input_features      : dict (the reconstructed feature vector sent to model)
    """
    art = _load_artifact()
    if art is None:
        raise FileNotFoundError(
            f"Modèle notebook introuvable ({_PKL_NAME}). "
            "Relancez la Cell 34 du notebook impaye_risk_scoring.ipynb."
        )

    feature_cols = art.get("feature_cols", FEATURE_COLS)
    seuil = float(art.get("seuil_optimal", 0.15))

    # Build feature vector
    medians = _build_portfolio_medians()
    df_row = _build_inference_vector(payload, medians)

    # Keep only known features (handle possible subset)
    available = [c for c in feature_cols if c in df_row.columns]
    for c in feature_cols:
        if c not in df_row.columns:
            df_row[c] = medians.get(c, 0.0)

    X = df_row[feature_cols].fillna(0).values

    # Predict using LightGBM calibrated (preferred) or fallback to lgb_tuned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_prob: float | None = None
        try:
            raw_model = art.get("lgb_tuned")
            if raw_model is not None:
                raw_prob = float(raw_model.predict_proba(X)[0, 1])
        except Exception as exc:
            logger.warning("lgb_tuned raw score failed (%s)", exc)
        try:
            model = art["lgb_calibrated"]
            prob = float(model.predict_proba(X)[0, 1])
            model_name = "LightGBM calibré (isotonic)"
        except Exception as exc:
            logger.warning("lgb_calibrated failed (%s), falling back to lgb_tuned", exc)
            model = art["lgb_tuned"]
            prob = float(model.predict_proba(X)[0, 1])
            raw_prob = prob
            model_name = "LightGBM tuné"

    predicted = int(prob >= seuil)

    # Risk level
    if prob >= 0.60:
        risk_level = "Critique"
    elif prob >= 0.40:
        risk_level = "Élevé"
    elif prob >= seuil:
        risk_level = "Modéré"
    else:
        risk_level = "Faible"

    # Feature importance from lgb_tuned (interpretable)
    feature_importance: dict[str, float] = {}
    try:
        lgb_tuned = art.get("lgb_tuned")
        if lgb_tuned is not None and hasattr(lgb_tuned, "feature_importances_"):
            imps = lgb_tuned.feature_importances_
            pairs = sorted(zip(feature_cols, imps), key=lambda x: -x[1])
            feature_importance = {f: round(float(v) / max(imps.sum(), 1), 4) for f, v in pairs[:8]}
    except Exception:
        pass

    return {
        "probability_impaye": round(prob, 4),
        "raw_probability_impaye": round(raw_prob, 4) if raw_prob is not None else None,
        "raw_threshold": 0.010,
        "predicted_label":    predicted,
        "threshold":          seuil,
        "risk_level":         risk_level,
        "model":              model_name,
        "feature_importance": feature_importance,
        "input_features":     {k: round(float(v), 4) for k, v in df_row[feature_cols].iloc[0].items()},
    }
