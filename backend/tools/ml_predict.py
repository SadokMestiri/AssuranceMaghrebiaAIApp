from __future__ import annotations

import importlib
import pathlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from db import query_dataframe as _query_dataframe
from ml_pipeline import FEATURE_COLUMNS, load_model_metadata, load_training_dataset
from utils import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    normalize_text as _normalize_text,
    format_metric_value as _format_metric_value,
)
from tools._shared import _normalize_branch, _resolve_period_context, _to_markdown_table

def ml_predict_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    import joblib
    import pathlib
    year_from, year_to = _resolve_period_context(context)
    branch = _normalize_branch(context.get("branch"))

    # Guard: if no year context is provided, ask for clarification
    if "year_from" not in context and "year_to" not in context:
        return {
            "tool": "ml_predict_tool",
            "summary": (
                "Précisez la période d'analyse pour lancer le modèle ML. "
                "Exemple : 'Lance le modèle ML sur les données 2024' ou 'Prédis le risque fraude/résiliation pour 2024'."
            ),
            "payload": {
                "needs_clarification": True,
                "question": "Sur quelle période souhaitez-vous lancer la prédiction ? (ex: 2023, 2024, 2023-2024)",
            },
            "charts": [],
        }

    sql_query = """
        WITH emission AS (
            SELECT annee_echeance as annee, mois_echeance as mois, branche as departement,
                   COALESCE(SUM(mt_pnet), 0) as primes_acquises_tnd,
                   COUNT(DISTINCT id_police) as nb_polices
            FROM dwh_fact_emission
            WHERE annee_echeance BETWEEN :year_from AND :year_to AND (:branch IS NULL OR branche = :branch)
              AND etat_quit IN ('E','P','A')
            GROUP BY annee_echeance, mois_echeance, branche
        ),
        impaye AS (
            SELECT annee_echeance as annee, mois_echeance as mois, branche as departement,
                   COALESCE(SUM(mt_acp), 0) as cout_sinistres_tnd,
                   COUNT(*) as nb_sinistres
            FROM dwh_fact_impaye
            WHERE annee_echeance BETWEEN :year_from AND :year_to AND (:branch IS NULL OR branche = :branch)
            GROUP BY annee_echeance, mois_echeance, branche
        )
        SELECT e.annee, e.mois, e.departement, e.primes_acquises_tnd, e.nb_polices,
               COALESCE(i.cout_sinistres_tnd, 0) AS cout_sinistres_tnd,
               COALESCE(i.nb_sinistres, 0) AS nb_sinistres
        FROM emission e
        LEFT JOIN impaye i ON e.annee = i.annee AND e.mois = i.mois AND e.departement = i.departement
        ORDER BY annee, mois
    """

    try:
        df = _query_dataframe(sql_query, {"year_from": year_from, "year_to": year_to, "branch": branch})
        if df.empty:
            return {
                "tool": "ml_predict_tool",
                "summary": f"Aucune donnée disponible pour la période {year_from}-{year_to}. Vérifiez les filtres.",
                "payload": {"needs_clarification": True},
                "charts": [],
            }

        df["ratio_combine_pct"] = (df["cout_sinistres_tnd"] / df["primes_acquises_tnd"].replace(0, float("nan"))) * 100.0
        df["ratio_combine_pct"] = df["ratio_combine_pct"].replace([float("inf"), -float("inf")], 0).fillna(0)
        df["provision_totale_tnd"] = df["cout_sinistres_tnd"] * 1.5

        models_dir = pathlib.Path(__file__).parent / "models"

        # Check all model files exist before loading
        required_files = ["features.pkl", "scaler.pkl", "rf_model_resiliation.pkl", "gb_model_fraude.pkl"]
        missing = [f for f in required_files if not (models_dir / f).exists()]
        if missing:
            return {
                "tool": "ml_predict_tool",
                "summary": f"Modèles ML non disponibles ({', '.join(missing)}). Entraînez les modèles d'abord via /api/v1/ml/train.",
                "payload": {"error": f"Fichiers manquants: {missing}", "needs_clarification": False},
                "charts": [],
            }

        features = joblib.load(models_dir / "features.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        rf_model = joblib.load(models_dir / "rf_model_resiliation.pkl")
        gb_model = joblib.load(models_dir / "gb_model_fraude.pkl")

        # Validate features exist in df
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return {
                "tool": "ml_predict_tool",
                "summary": f"Colonnes manquantes pour le modèle ML: {missing_features}. Vérifiez la compatibilité du modèle avec les données.",
                "payload": {"error": f"Features manquantes: {missing_features}", "needs_clarification": True},
                "charts": [],
            }

        idx = df[features].index
        X_scaled = scaler.transform(df[features])
        df.loc[idx, "pred_resiliation"] = rf_model.predict(X_scaled)
        df.loc[idx, "pred_fraude"] = gb_model.predict(X_scaled)

        high_resil = int(df["pred_resiliation"].sum())
        high_fraud = int(df["pred_fraude"].sum())
        res_rows = df.tail(12).to_dict(orient="records")

        # Build a chart showing predicted risk per month/branch
        chart_items = [
            {
                "period": f"{int(row.get('annee', 0))}-{int(row.get('mois', 0)):02d}",
                "departement": str(row.get("departement", "N/A")),
                "pred_resiliation": int(row.get("pred_resiliation", 0)),
                "pred_fraude": int(row.get("pred_fraude", 0)),
            }
            for row in res_rows
            if isinstance(row, dict)
        ]

        summary = (
            f"Prédictions ML ({year_from}-{year_to}, "
            f"{'toutes branches' if not branch else f'branche {branch}'}): "
            f"{high_resil} mois avec risque élevé de résiliation, "
            f"{high_fraud} mois avec risque de fraude."
        )

        return {
            "tool": "ml_predict_tool",
            "summary": summary,
            "payload": {
                "branch": branch or "ALL",
                "year_from": year_from,
                "year_to": year_to,
                "predictions": res_rows,
                "total_high_resiliation": high_resil,
                "total_high_fraud": high_fraud,
            },
            "charts": [
                _build_chart_payload(
                    chart_type="bar",
                    title="Prédiction risque résiliation par mois",
                    x_key="period",
                    y_key="pred_resiliation",
                    items=chart_items,
                )
            ],
        }
    except Exception as e:
        return {
            "tool": "ml_predict_tool",
            "summary": f"Erreur lors de la prédiction ML: {str(e)}",
            "payload": {"error": str(e), "year_from": year_from, "year_to": year_to},
            "charts": [],
        }

