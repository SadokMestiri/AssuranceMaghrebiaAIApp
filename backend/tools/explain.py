from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from db import query_dataframe as _query_dataframe
from ml_pipeline import FEATURE_COLUMNS, load_model_metadata, load_training_dataset
from utils import safe_float as _safe_float
from tools._shared import (
    _resolve_period_context,
    _build_chart_payload,
)


def explain_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    metadata = load_model_metadata()
    metrics = metadata.get("metrics", {})
    year_from, year_to = _resolve_period_context(context)

    shap_features: list[dict[str, Any]] = []
    engine = "statistical_fallback"
    explain_error = ""

    try:
        joblib = importlib.import_module("joblib")
        shap = importlib.import_module("shap")

        model_path = metadata.get("model_path")
        if not model_path:
            raise ValueError("model_path missing in metadata")

        bundle = joblib.load(model_path)
        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        if not hasattr(model, "named_steps"):
            raise ValueError("Unsupported model format for SHAP")

        dataset = load_training_dataset(year_from=year_from, year_to=year_to)
        if dataset.empty:
            raise ValueError("Empty dataset for SHAP")

        sample = dataset[FEATURE_COLUMNS].head(min(180, len(dataset))).copy()
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(sample)
        if hasattr(transformed, "toarray") and transformed.shape[1] <= 1200:
            transformed = transformed.toarray()

        feature_names = list(preprocessor.get_feature_names_out())
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            class_values = np.array(shap_values[-1])
        else:

            class_values = np.array(shap_values)

        mean_abs = np.abs(class_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:12]
        shap_features = [
            {
                "feature": feature_names[idx],
                "importance": float(mean_abs[idx]),
            }
            for idx in top_idx
        ]
        engine = "shap"
    except Exception as exc:
        explain_error = str(exc)

    if not shap_features:
        sql_query = """
            WITH labeled AS (
                SELECT
                    e.mt_pnet,
                    e.mt_commission,
                    e.bonus_malus,
                    CASE WHEN i.num_quittance IS NULL THEN 0 ELSE 1 END AS is_impaye
                FROM dwh_fact_emission e
                LEFT JOIN (SELECT DISTINCT num_quittance FROM dwh_fact_impaye) i
                  ON i.num_quittance = e.num_quittance
                WHERE e.etat_quit IN ('E','P','A')
                ORDER BY e.annee_echeance DESC, e.mois_echeance DESC
                LIMIT 30000
            )
            SELECT
                AVG(mt_pnet) FILTER (WHERE is_impaye = 1) AS impaye_avg_pnet,
                AVG(mt_pnet) FILTER (WHERE is_impaye = 0) AS non_impaye_avg_pnet,
                AVG(mt_commission) FILTER (WHERE is_impaye = 1) AS impaye_avg_commission,
                AVG(mt_commission) FILTER (WHERE is_impaye = 0) AS non_impaye_avg_commission,
                AVG(bonus_malus) FILTER (WHERE is_impaye = 1) AS impaye_avg_bonus_malus,
                AVG(bonus_malus) FILTER (WHERE is_impaye = 0) AS non_impaye_avg_bonus_malus
            FROM labeled
        """

        stats = _query_dataframe(sql_query).iloc[0]
        shap_features = [
            {
                "feature": "bonus_malus",
                "importance": abs(_safe_float(stats["impaye_avg_bonus_malus"]) - _safe_float(stats["non_impaye_avg_bonus_malus"])),
            },
            {
                "feature": "mt_commission",
                "importance": abs(_safe_float(stats["impaye_avg_commission"]) - _safe_float(stats["non_impaye_avg_commission"])),
            },
            {
                "feature": "mt_pnet",
                "importance": abs(_safe_float(stats["impaye_avg_pnet"]) - _safe_float(stats["non_impaye_avg_pnet"])),
            },
        ]

    summary = (
        f"Explain tool ({engine}): top facteurs identifies, recall {_safe_float(metrics.get('recall')):.3f}, "
        f"avg_precision {_safe_float(metrics.get('avg_precision')):.3f}."
    )

    payload = {
        "engine": engine,
        "model_role": metadata.get("model_role", "champion"),
        "run_id": metadata.get("run_id"),
        "metrics": metrics,
        "feature_importance": shap_features,
    }
    if explain_error and engine != "shap":
        payload["fallback_reason"] = explain_error

    return {
        "tool": "explain_tool",
        "summary": summary,
        "payload": payload,
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Feature importance",
                x_key="feature",
                y_key="importance",
                items=shap_features,
            )
        ],
    }
