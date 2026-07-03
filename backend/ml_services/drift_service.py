"""
ml_services/drift_service.py
────────────────────────────
Data drift detection — data_drift_evidently.ipynb

Source data: DWH_FACT_EMISSION (62,961 rows, 2017–2025).
Temporal split:
  Reference : [max_date - nb_mois_reference - nb_mois_courant, max_date - nb_mois_courant)
  Current   : [max_date - nb_mois_courant, max_date]

Metrics:
  1. Evidently AI DataDriftPreset  → per-feature p-value (KS / Chi²)
  2. PSI (Population Stability Index) → per-feature stability score
  3. Mean comparison → % variation reference → current
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._base import load_emission

logger = logging.getLogger("maghrebia.ml_services.drift")

FEATURE_NUMERICAL   = ["MT_PNET", "MT_COMMISSION", "RATIO_COMMISSION", "BONUS_MALUS"]
FEATURE_CATEGORICAL = ["ETAT_QUIT", "BRANCHE", "PERIODICITE"]
FEATURES = FEATURE_NUMERICAL + FEATURE_CATEGORICAL


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_features() -> pd.DataFrame:
    em = load_emission()
    if em.empty:
        return pd.DataFrame()

    em.columns = [c.upper() for c in em.columns]
    em["DATE_EMISSION"] = pd.to_datetime(em.get("DATE_EMISSION"), errors="coerce")

    def _num(col, default=0.0):
        if col in em.columns:
            return pd.to_numeric(em[col], errors="coerce").fillna(default)
        return pd.Series(default, index=em.index)

    em["MT_PNET"]       = _num("MT_PNET",      0.0)
    em["MT_COMMISSION"] = _num("MT_COMMISSION", 0.0)
    em["BONUS_MALUS"]   = _num("BONUS_MALUS",   1.0)
    em["RATIO_COMMISSION"] = (
        em["MT_COMMISSION"] / em["MT_PNET"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    for col in FEATURE_CATEGORICAL:
        if col in em.columns:
            em[col] = em[col].fillna("INCONNU").astype(str).str.upper()
        else:
            em[col] = "INCONNU"

    return em.dropna(subset=["DATE_EMISSION"])


# ── PSI ───────────────────────────────────────────────────────────────────────

def _psi_numerical(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    bp = np.percentile(expected, np.linspace(0, 100, bins + 1))
    bp[0] -= 1e-9
    bp[-1] += 1e-9
    exp_pct = (np.histogram(expected, bins=bp)[0] + 1e-9) / len(expected)
    act_pct = (np.histogram(actual,   bins=bp)[0] + 1e-9) / len(actual)
    return round(float(max(0.0, np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))), 6)


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = set(expected.unique()) | set(actual.unique())
    psi  = 0.0
    for c in cats:
        e = max(float((expected == c).mean()), 1e-9)
        a = max(float((actual   == c).mean()), 1e-9)
        psi += (a - e) * np.log(a / e)
    return round(float(max(0.0, psi)), 6)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_drift(
    departement:       Optional[str] = None,
    nb_mois_reference: int           = 12,
    nb_mois_courant:   int           = 6,
) -> dict[str, Any]:

    em = _build_features()
    if em.empty:
        return {"error": "Aucune donnée disponible"}

    if departement and "BRANCHE" in em.columns:
        em = em[em["BRANCHE"] == departement.upper()]
        if em.empty:
            return {"error": f"Aucune donnée pour la branche {departement}"}

    # ── Temporal split ────────────────────────────────────────────────────────
    max_date  = em["DATE_EMISSION"].max()
    cur_start = max_date - pd.DateOffset(months=nb_mois_courant)
    ref_end   = cur_start
    ref_start = ref_end - pd.DateOffset(months=nb_mois_reference)

    ref_df = em[(em["DATE_EMISSION"] >= ref_start) & (em["DATE_EMISSION"] < ref_end)].copy()
    cur_df = em[ em["DATE_EMISSION"] >= cur_start].copy()

    if len(ref_df) < 50 or len(cur_df) < 50:
        return {
            "error": (
                f"Données insuffisantes — référence : {len(ref_df)} lignes, "
                f"courant : {len(cur_df)} lignes (min 50 requis par fenêtre)"
            )
        }

    avail_num = [f for f in FEATURE_NUMERICAL   if f in ref_df.columns]
    avail_cat = [f for f in FEATURE_CATEGORICAL if f in ref_df.columns]
    all_feats = avail_num + avail_cat

    # ── Drift detection (Evidently first, scipy fallback) ─────────────────────
    drift_results: list[dict] = []

    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        col_map = ColumnMapping(numerical_features=avail_num, categorical_features=avail_cat)
        report  = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_df[all_feats],
            current_data=cur_df[all_feats],
            column_mapping=col_map,
        )
        result = report.as_dict()

        for metric in result.get("metrics", []):
            if metric.get("metric") == "DataDriftTable":
                for col, info in metric.get("result", {}).get("drift_by_columns", {}).items():
                    drift_results.append({
                        "feature":       col,
                        "drift_detecte": bool(info.get("drift_detected", False)),
                        "p_value":       round(float(info.get("drift_score", 1.0)), 4),
                        "methode":       str(info.get("stattest_name", "KS")),
                    })

    except Exception as ev_err:
        logger.warning("Evidently failed (%s) — using scipy fallback", ev_err)
        from scipy.stats import chi2_contingency, ks_2samp

        for col in avail_num:
            _, p = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
            drift_results.append({
                "feature": col, "drift_detecte": bool(p < 0.05),
                "p_value": round(float(p), 4), "methode": "KS",
            })
        for col in avail_cat:
            cats = list(set(ref_df[col].unique()) | set(cur_df[col].unique()))
            obs  = np.array([
                [(ref_df[col] == c).sum() for c in cats],
                [(cur_df[col] == c).sum() for c in cats],
            ])
            try:
                _, p, _, _ = chi2_contingency(obs)
            except Exception:
                p = 1.0
            drift_results.append({
                "feature": col, "drift_detecte": bool(p < 0.05),
                "p_value": round(float(p), 4), "methode": "Chi²",
            })

    # ── PSI ───────────────────────────────────────────────────────────────────
    psi_results: list[dict] = []
    for col in avail_num:
        psi = _psi_numerical(ref_df[col].dropna().values, cur_df[col].dropna().values)
        psi_results.append({
            "feature": col, "psi": psi,
            "niveau": "critique" if psi >= 0.2 else "modere" if psi >= 0.1 else "stable",
        })
    for col in avail_cat:
        psi = _psi_categorical(ref_df[col], cur_df[col])
        psi_results.append({
            "feature": col, "psi": psi,
            "niveau": "critique" if psi >= 0.2 else "modere" if psi >= 0.1 else "stable",
        })
    psi_results.sort(key=lambda x: x["psi"], reverse=True)

    # ── Comparison means (numerical only) ─────────────────────────────────────
    comparison: list[dict] = []
    for col in avail_num:
        m_ref = float(ref_df[col].mean())
        m_cur = float(cur_df[col].mean())
        var   = ((m_cur - m_ref) / m_ref * 100) if m_ref != 0 else 0.0
        comparison.append({
            "feature":       col,
            "moyenne_ref":   round(m_ref, 4),
            "moyenne_cur":   round(m_cur, 4),
            "variation_pct": round(var, 2),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    nb_drifted  = sum(1 for d in drift_results if d["drift_detecte"])
    nb_total    = len(drift_results)
    share_drift = nb_drifted / nb_total if nb_total else 0.0

    if share_drift >= 0.5:
        niveau, message = "critique", f"Drift majeur — {nb_drifted}/{nb_total} features ont dérivé"
    elif share_drift >= 0.3:
        niveau, message = "warning",  f"Drift modéré — {nb_drifted}/{nb_total} features ont dérivé"
    else:
        niveau, message = "normal",   f"Distribution stable — {nb_drifted}/{nb_total} features ont dérivé"

    return {
        "departement":       departement or "Tous",
        "nb_mois_reference": nb_mois_reference,
        "nb_mois_courant":   nb_mois_courant,
        "nb_ref":            len(ref_df),
        "nb_courant":        len(cur_df),
        "date_ref_debut":    str(ref_start.date()),
        "date_ref_fin":      str(ref_end.date()),
        "date_cur_debut":    str(cur_start.date()),
        "date_max":          str(max_date.date()),
        "dataset_drift":     share_drift >= 0.5,
        "nb_drifted":        nb_drifted,
        "nb_features":       nb_total,
        "share_drift":       round(share_drift, 4),
        "niveau":            niveau,
        "message":           message,
        "features":          drift_results,
        "comparaison":       comparison,
        "psi_features":      psi_results,
    }
