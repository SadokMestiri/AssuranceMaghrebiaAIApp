"""
Maghrebia â€” Service evidently Data Drift
DÃ©tection de drift sur les branches AUTO, IRDS, SANTE.
"""

import os
import logging
import psycopg2
import pandas as pd
from decimal import Decimal
from typing import Optional
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://maghrebia:maghrebia@postgres:5432/maghrebia")

FEATURES = [
    "ratio_combine_pct",
    "primes_acquises_tnd",
    "cout_sinistres_tnd",
    "nb_sinistres",
    "taux_resiliation_pct",
    "provision_totale_tnd",
    "nb_suspicions_fraude",
]

def _clean(val):
    if isinstance(val, Decimal): return float(val)
    return float(val) if val else 0.0

def detect_drift(departement: Optional[str] = None, nb_mois_reference: int = 12, nb_mois_courant: int = 6) -> dict:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        return {"error": "Evidently non installÃ© sur le backend Maghrebia"}

    conn = psycopg2.connect(DATABASE_URL.replace("+psycopg2", ""))
    cur = conn.cursor()

    conditions = ["annee >= 2018"]
    params = []
    if departement:
        conditions.append("departement = %s")
        params.append(departement)

    where = "WHERE " + " AND ".join(conditions)
    cur.execute(f"SELECT annee, mois, departement, " + ", ".join(FEATURES) + 
                f" FROM kpis_mensuels {where} ORDER BY annee, mois", params)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    cols = ["annee", "mois", "departement"] + FEATURES
    df = pd.DataFrame(rows, columns=cols)
    for f in FEATURES: df[f] = df[f].apply(_clean)

    if not departement:
        df = df.groupby(["annee", "mois"])[FEATURES].mean().reset_index()
        df = df.sort_values(["annee", "mois"])

    if len(df) < nb_mois_reference + nb_mois_courant:
        return {"error": f"Donnees insuffisantes ({len(df)} mois, besoin de {nb_mois_reference + nb_mois_courant} mois)"}

    reference = df.head(nb_mois_reference)[FEATURES].reset_index(drop=True)
    courant = df.tail(nb_mois_courant)[FEATURES].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=courant)
    ev_result = report.as_dict()

    dataset_drifted = False
    nb_drifted = 0
    nb_total = len(FEATURES)
    share_drift = 0.0

    for metric in ev_result.get("metrics", []):
        r = metric.get("result", {})
        if "dataset_drift" in r:
            dataset_drifted = r.get("dataset_drift", False)
            nb_drifted = r.get("number_of_drifted_columns", 0)
            nb_total = r.get("number_of_columns", len(FEATURES))
            share_drift = r.get("share_of_drifted_columns", 0.0)
            break

    drift_results = []
    for feat in FEATURES:
        ref_vals = reference[feat].dropna().values
        cur_vals = courant[feat].dropna().values
        if len(ref_vals) < 2 or len(cur_vals) < 2:
            drift_results.append({"feature": feat, "drift_detecte": False, "p_value": None, "methode": "KS"})
            continue
        stat, pval = ks_2samp(ref_vals, cur_vals)
        drift_results.append({
            "feature": feat, "drift_detecte": bool(pval < 0.05),
            "p_value": round(float(pval), 4), "methode": "KS",
        })

    nb_drifted = sum(1 for d in drift_results if d["drift_detecte"])
    dataset_drifted = nb_drifted > nb_total * 0.5
    share_drift = nb_drifted / nb_total if nb_total > 0 else 0

    if share_drift >= 0.5: niveau, message = "critique", "Drift majeur"
    elif share_drift >= 0.3: niveau, message = "warning", "Drift modÃ©rÃ©"
    else: niveau, message = "normal", "Pas de drift significatif"

    comparaison = []
    for f in FEATURES:
        mean_ref = reference[f].mean()
        mean_cur = courant[f].mean()
        variation = ((mean_cur - mean_ref) / mean_ref * 100 if mean_ref != 0 else 0)
        comparaison.append({
            "feature": f, "moyenne_ref": round(mean_ref, 2),
            "moyenne_cur": round(mean_cur, 2), "variation_pct": round(variation, 1),
        })

    return {
        "departement": departement or "Tous", "nb_mois_reference": nb_mois_reference,
        "nb_mois_courant": nb_mois_courant, "dataset_drift": dataset_drifted,
        "nb_features_drift": nb_drifted, "nb_features_total": nb_total,
        "share_drift": round(share_drift, 3), "niveau": niveau,
        "message": message, "features": drift_results, "comparaison": comparaison,
    }
