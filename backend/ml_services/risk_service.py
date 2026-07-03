"""
ml_services/risk_service.py
────────────────────────────
Risk scoring & pricing service — risk_scoring_pricing.ipynb

Notebook artifact  : risk_scoring_model.pkl
  Keys : lgb_freq, lgb_sev, xgb_freq, xgb_sev, encoders,
         features (30), feature_defaults, anchors, chargement,
         w_xgb_freq, w_lgb_freq, w_xgb_sev, w_lgb_sev

Architecture (frequency–severity ensemble):
  PRED_FREQ  = w_xgb_freq * XGB_freq.predict() + w_lgb_freq * LGB_freq.predict()
  PRED_SEV   = w_xgb_sev  * XGB_sev.predict()  + w_lgb_sev  * LGB_sev.predict()
  Models trained on log1p(target) → outputs need np.expm1()
  PRIME_PURE      = PRED_FREQ × PRED_SEV
  PRIME_TECHNIQUE = PRIME_PURE × chargement (1.28)
  RISK_SCORE [0–1000] from FREQ + SEV + IMPAYE anchors

Falls back to legacy proxy (risk_model.pkl) if notebook artifact is absent.
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

_ARTIFACT_NOTEBOOK = "risk_scoring_model"
_ARTIFACT_LEGACY   = "risk_model"
_CHARGEMENT_TOTAL  = 1.28

# Legacy feature set (12 cols) — used only for proxy training
FEATURE_COLS = [
    "BONUS_MALUS", "NB_QUITTANCES", "EXPOSITION_ANS",
    "PUISSANCE", "AGE_VEHICULE", "NB_PLACE",
    "AGE_CLIENT", "FLAG_MORALE", "ID_BRANCHE",
    "TAUX_IMPAYE", "NB_IMPAYES", "PRIME_PAR_AN",
]

_RISK_CLASSES = [
    (0,   200, "A — Excellent"),
    (200, 400, "B — Bon"),
    (400, 600, "C — Modéré"),
    (600, 800, "D — Élevé"),
    (800, 1001,"E — Critique"),
]

_BRANCHE_ENC = {"AUTO": 0, "IRDS": 1, "SANTE": 2}


def _risk_class(score: float) -> str:
    for lo, hi, label in _RISK_CLASSES:
        if lo <= score < hi:
            return label
    return "E — Critique"


def _is_notebook_art(art: dict) -> bool:
    return "lgb_freq" in art or "xgb_freq" in art


# ── Feature builder (for portfolio table + proxy training) ──────────────────

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

    if not emission.empty:
        emission["MT_PNET"] = pd.to_numeric(emission.get("MT_PNET"), errors="coerce").fillna(0)
        em_agg = emission.groupby("ID_POLICE").agg(
            NB_QUITTANCES=("NUM_QUITTANCE","count"),
            PRIME_PAR_AN =("MT_PNET",      "sum"),   # total annual premium, not quarterly avg
        ).reset_index()
        df = df.merge(em_agg, on="ID_POLICE", how="left")
    else:
        df["NB_QUITTANCES"] = 0
        df["PRIME_PAR_AN"]  = 0.0

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

    if not impaye.empty:
        imp_agg = impaye.groupby("ID_POLICE")["NUM_QUITTANCE"].count().reset_index(name="NB_IMPAYES")
        df = df.merge(imp_agg, on="ID_POLICE", how="left")
    df["NB_IMPAYES"] = df.get("NB_IMPAYES", 0).fillna(0)
    df["TAUX_IMPAYE"] = df["NB_IMPAYES"] / df["NB_QUITTANCES"].replace(0, 1)

    if not vehicule.empty:
        vehicule["PUISSANCE"]    = pd.to_numeric(vehicule.get("PUISSANCE"),   errors="coerce").fillna(6)
        vehicule["NB_PLACE"]     = pd.to_numeric(vehicule.get("NB_PLACE"),    errors="coerce").fillna(5)
        vehicule["DATE_MEC"]     = pd.to_datetime(vehicule.get("DATE_MEC"), errors="coerce")
        vehicule["AGE_VEHICULE"] = ((ref - vehicule["DATE_MEC"]).dt.days / 365.25).clip(0).fillna(5)
        df = df.merge(vehicule[["ID_VEHICULE","PUISSANCE","NB_PLACE","AGE_VEHICULE"]]
                      .rename(columns={"ID_VEHICULE":"ID_POLICE"}),
                      on="ID_POLICE", how="left")
    else:
        df["PUISSANCE"]    = 6.0
        df["NB_PLACE"]     = 5.0
        df["AGE_VEHICULE"] = 5.0

    df["AGE_CLIENT"] = 40.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


# ── Proxy model trainer ─────────────────────────────────────────────────────

def _train_proxy(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    valid   = [c for c in FEATURE_COLS if c in df.columns]
    X       = df[valid].fillna(0).values
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(imputer.fit_transform(X))

    y_freq = df["FREQ_SINISTRE"].clip(0, 2).values
    freq_model = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                            learning_rate=0.1, random_state=42)
    freq_model.fit(X_sc, np.log1p(y_freq))

    y_sev = df["SEVERITE_MOY"].clip(0).values
    sev_model = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                           learning_rate=0.1, random_state=42)
    sev_model.fit(X_sc, np.log1p(y_sev))

    df["PRED_FREQ"] = np.expm1(freq_model.predict(X_sc)).clip(0)
    df["PRED_SEV"]  = np.expm1(sev_model.predict(X_sc)).clip(0)
    anchors = {
        "freq_p1":  float(df["PRED_FREQ"].quantile(0.01)),
        "freq_p99": float(df["PRED_FREQ"].quantile(0.99)),
        "sev_p1":   float(df["PRED_SEV"].quantile(0.01)),
        "sev_p99":  float(df["PRED_SEV"].quantile(0.99)),
    }

    art = {
        "freq_model": freq_model,
        "sev_model":  sev_model,
        "imputer":    imputer,
        "scaler":     scaler,
        "features":   valid,
        "anchors":    anchors,
        "chargement": _CHARGEMENT_TOTAL,
        "source":     "proxy",
    }
    save_artifact(_ARTIFACT_LEGACY, art)
    logger.info("Risk proxy model trained and saved.")
    return art, df


def _get_artifact_and_df() -> tuple[dict, pd.DataFrame]:
    # Prefer notebook artifact; fall back to legacy proxy
    art = load_artifact(_ARTIFACT_NOTEBOOK) or load_artifact(_ARTIFACT_LEGACY)
    df  = _build_portfolio()
    if df.empty:
        raise RuntimeError("No data for risk service.")
    if art is None:
        return _train_proxy(df)
    return art, df


# ── Score computation ────────────────────────────────────────────────────────

_MIN_SEV_TND = 10.0  # predictions below this in TND indicate a broken model


def _compute_score_notebook(art: dict, row: dict) -> dict:
    """Compute risk score using the ensemble notebook artifact."""
    feat_cols = art["features"]
    X = pd.DataFrame([row], columns=feat_cols)

    # Frequency ensemble — both XGB and LGB contribute
    freq_xgb = float(np.expm1(art["xgb_freq"].predict(X)[0]))
    freq_lgb = float(np.expm1(art["lgb_freq"].predict(X)[0]))
    freq = (
        art["w_xgb_freq"] * freq_xgb +
        art["w_lgb_freq"] * freq_lgb
    )

    # Severity ensemble — XGBoost severity model may be broken (constant near-zero).
    # Fall back to LightGBM-only when XGB is anomalous.
    sev_xgb = float(np.expm1(art["xgb_sev"].predict(X)[0]))
    sev_lgb = float(np.expm1(art["lgb_sev"].predict(X)[0]))
    if sev_xgb < _MIN_SEV_TND and sev_lgb >= _MIN_SEV_TND:
        sev = sev_lgb
    else:
        sev = art["w_xgb_sev"] * sev_xgb + art["w_lgb_sev"] * sev_lgb

    freq = max(0.0, freq)
    sev  = max(0.0, sev)

    prime_pure      = freq * sev
    chargement      = art.get("chargement", _CHARGEMENT_TOTAL)
    prime_technique = round(prime_pure * chargement)
    loading_factor  = round(chargement - 1, 3)

    # Risk score from anchors — notebook format: anchors["FREQ"] = (p1, p99)
    anchors   = art.get("anchors", {})
    freq_anch = anchors.get("FREQ",   (0.0, 0.5))
    sev_anch  = anchors.get("SEV",    (0.0, 10000.0))
    imp_anch  = anchors.get("IMPAYE", (0.0, 0.5))

    score_freq = normalize_0_1000(freq, freq_anch[0], freq_anch[1])
    score_sev  = normalize_0_1000(sev,  sev_anch[0],  sev_anch[1])
    taux_imp   = float(row.get("TAUX_IMPAYE", 0))
    score_imp  = normalize_0_1000(taux_imp, imp_anch[0], imp_anch[1])

    risk_score = int(0.45 * score_freq + 0.40 * score_sev + 0.15 * score_imp)
    return {
        "risk_score":      risk_score,
        "risk_label":      _risk_class(risk_score),
        "prime_technique": prime_technique,
        "loading_factor":  loading_factor,
        "components": {
            "frequence_estimee": round(freq, 4),
            "severite_estimee":  round(sev,  0),
            "prime_pure":        round(prime_pure, 0),
        },
        "model": art.get("source", "Ensemble XGB+LGB"),
    }


def _compute_score_legacy(art: dict, row: pd.Series) -> dict:
    """Compute risk score using the legacy proxy artifact."""
    feats = art["features"]
    X = art["imputer"].transform(row[feats].fillna(0).values.reshape(1, -1))
    X = art["scaler"].transform(X)

    freq = float(np.expm1(art["freq_model"].predict(X)[0]))
    sev  = float(np.expm1(art["sev_model"].predict(X)[0]))

    prime_pure      = max(0, freq * sev)
    chargement      = art.get("chargement", _CHARGEMENT_TOTAL)
    prime_technique = round(prime_pure * chargement)
    loading_factor  = round(chargement - 1, 3)

    anchors    = art.get("anchors", {})
    score_freq = normalize_0_1000(freq, anchors.get("freq_p1", 0), anchors.get("freq_p99", 0.5))
    score_sev  = normalize_0_1000(sev,  anchors.get("sev_p1",  0), anchors.get("sev_p99",  10000))
    taux_imp   = float(row.get("TAUX_IMPAYE", 0))
    score_imp  = min(1000, taux_imp * 2000)

    risk_score = int(0.45 * score_freq + 0.40 * score_sev + 0.15 * score_imp)
    return {
        "risk_score":      risk_score,
        "risk_label":      _risk_class(risk_score),
        "prime_technique": prime_technique,
        "loading_factor":  loading_factor,
        "components": {
            "frequence_estimee": round(freq, 4),
            "severite_estimee":  round(sev,  0),
            "prime_pure":        round(prime_pure, 0),
        },
        "model": art.get("source", "proxy"),
    }


# ── Public API ──────────────────────────────────────────────────────────────

def score_risk(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute risk score + prime technique for a single policy."""
    art, _ = _get_artifact_and_df()

    if _is_notebook_art(art):
        feat_cols = art["features"]
        defaults  = art.get("feature_defaults", {c: 0.0 for c in feat_cols})

        # Start from notebook training medians
        row = {c: defaults.get(c, 0.0) for c in feat_cols}

        # Map simulator inputs
        bm         = safe_float(payload.get("bonus_malus",       defaults.get("BONUS_MALUS",    1.0)))
        puissance  = safe_float(payload.get("puissance",         defaults.get("PUISSANCE",      6.0)))
        age_veh    = safe_float(payload.get("age_vehicule",      defaults.get("AGE_VEHICULE",   5.0)))
        age_cli    = safe_float(payload.get("age_client",        defaults.get("AGE_CLIENT",     40.0)))
        mt_pnet    = safe_float(payload.get("mt_pnet",           defaults.get("PRIME_PAR_AN",   0.0)))
        nb_quit    = safe_float(payload.get("nb_quittances",     defaults.get("NB_QUITTANCES",  2.0)))
        taux_imp   = safe_float(payload.get("taux_impaye",       defaults.get("TAUX_IMPAYE",    0.0)))
        nb_imp     = safe_float(payload.get("nb_impayes",        defaults.get("NB_IMPAYES",     0.0)))
        branche    = str(payload.get("branche", "AUTO")).upper()

        row["BONUS_MALUS"]     = bm
        row["PUISSANCE"]       = puissance
        row["AGE_VEHICULE"]    = age_veh
        row["AGE_CLIENT"]      = age_cli
        row["PRIME_PAR_AN"]    = mt_pnet
        row["PRIME_MOY"]       = mt_pnet / max(1, nb_quit)
        row["NB_QUITTANCES"]   = nb_quit
        row["TAUX_IMPAYE"]     = taux_imp
        row["NB_IMPAYES"]      = nb_imp

        # Branche encoding
        branche_enc = _BRANCHE_ENC.get(branche, 0)
        row["BRANCHE_ENC"] = float(branche_enc)
        row["ID_BRANCHE"]  = float(branche_enc + 1)

        # Derived: puissance category (matches notebook feature CAT_PUISSANCE)
        if puissance < 6:
            cat_p = 1
        elif puissance < 10:
            cat_p = 2
        elif puissance < 15:
            cat_p = 3
        elif puissance < 20:
            cat_p = 4
        else:
            cat_p = 5
        row["CAT_PUISSANCE"] = float(cat_p)

        # Vehicle value — important features: LOG_VALEUR_VEH (7.5%), LOG_VALEUR_NEUF (7.9%)
        # Database stores values in millimes; user input is in TND → multiply by 1000.
        # TAUX_DEPRECIATION estimated from age at 7%/year (typical for Maghreb market).
        valeur_veh = safe_float(payload.get("valeur_vehicule", 0.0))
        if valeur_veh > 0:
            val_millimes = valeur_veh * 1000.0
            taux_dep = min(0.85, age_veh * 0.07)
            val_neuf_millimes = val_millimes / max(0.15, 1.0 - taux_dep)
            row["LOG_VALEUR_VEH"]    = float(np.log(max(1.0, val_millimes)))
            row["LOG_VALEUR_NEUF"]   = float(np.log(max(1.0, val_neuf_millimes)))
            row["TAUX_DEPRECIATION"] = taux_dep

        return _compute_score_notebook(art, row)

    else:
        # Legacy proxy path
        df = _build_portfolio()
        medians = df[art["features"]].median() if not df.empty else pd.Series(0, index=art["features"])
        row = medians.copy()
        map_keys = {
            "bonus_malus":       "BONUS_MALUS",
            "puissance":         "PUISSANCE",
            "age_vehicule":      "AGE_VEHICULE",
            "age_client":        "AGE_CLIENT",
            "mt_pnet":           "PRIME_PAR_AN",
        }
        for src, dst in map_keys.items():
            if src in payload and dst in row.index:
                row[dst] = safe_float(payload[src])
        return _compute_score_legacy(art, row)


def get_risk_model_info() -> dict[str, Any]:
    """Return performance metrics and metadata for the risk scoring model."""
    art = load_artifact(_ARTIFACT_NOTEBOOK) or load_artifact(_ARTIFACT_LEGACY)
    if art is None:
        return {"status": "unavailable"}

    return {
        "status":    "ready",
        "source":    art.get("source",   "Ensemble XGB+LGB"),
        "notebook":  art.get("notebook", "risk_scoring_pricing.ipynb"),
        "features":  len(art.get("features", [])),
        "chargement": art.get("chargement", _CHARGEMENT_TOTAL),
        "weights": {
            "freq": {"xgb": art.get("w_xgb_freq"), "lgb": art.get("w_lgb_freq")},
            "sev":  {"xgb": art.get("w_xgb_sev"),  "lgb": art.get("w_lgb_sev")},
        },
        "metrics_freq": art.get("metrics_freq", {}),
        "metrics_sev":  art.get("metrics_sev",  {}),
    }


_VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}


def get_risk_table() -> dict[str, Any]:
    """Return aggregated risk statistics per branch (AUTO / IRDS / SANTE)."""
    _, df = _get_artifact_and_df()

    agg = df.groupby("ID_BRANCHE").agg(
        NB_POLICES    =("ID_POLICE",     "count"),
        FREQ_SIN_MOY  =("FREQ_SINISTRE", "mean"),
        SEV_MOY       =("SEVERITE_MOY",  "mean"),
        MT_EVAL_TOTAL =("MT_EVAL_SUM",   "sum"),   # aggregate claims
        PRIME_TOTAL   =("PRIME_PAR_AN",  "sum"),   # aggregate premiums
    ).reset_index()

    # Aggregate S/P ratio: total claims / total premiums (correct actuarial formula)
    agg["SP_RATIO_MOY"] = agg["MT_EVAL_TOTAL"] / agg["PRIME_TOTAL"].replace(0, float("nan"))
    agg["PRIME_TECHNIQUE"] = (agg["FREQ_SIN_MOY"] * agg["SEV_MOY"] * _CHARGEMENT_TOTAL).round(0)

    branche_map = {1: "AUTO", 2: "IRDS", 3: "SANTE"}
    agg["BRANCHE"] = agg["ID_BRANCHE"].map(branche_map)

    # Keep only known branches — filter out unmapped IDs (0, NaN, etc.)
    agg = agg[agg["BRANCHE"].isin(_VALID_BRANCHES)]

    records = []
    for _, row in agg.iterrows():
        records.append({
            "branche":         str(row["BRANCHE"]),
            "nb_polices":      int(row["NB_POLICES"]),
            "freq_sin_moy":    round(float(row["FREQ_SIN_MOY"]),  4),
            "sev_moy":         round(float(row["SEV_MOY"]),        0),
            "sp_ratio_moy":    round(float(row["SP_RATIO_MOY"]),  3),
            "prime_technique": round(float(row["PRIME_TECHNIQUE"]), 0),
        })

    return {"table": records}
