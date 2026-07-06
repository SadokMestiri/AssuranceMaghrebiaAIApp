from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from db import query_dataframe as _query_dataframe
from utils import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    normalize_text as _normalize_text,
    format_metric_value as _format_metric_value,
    infer_report_mode as _infer_report_mode,
)
from tools._shared import (
    _normalize_branch,
    _resolve_period_context,
    _to_markdown_table,
    _build_chart_payload,
)


def _infer_forecast_report_mode(question: str) -> str:
    return _infer_report_mode(question)


def _detect_forecast_target(normalized_question: str) -> dict[str, Any]:
    count_signal = any(token in normalized_question for token in ["nombre", "nb", "count", "volume", "combien"])
    amount_signal = any(token in normalized_question for token in ["montant", "somme", "total", "valeur", "tnd", "mt_"])

    if any(token in normalized_question for token in ["ratio", "combine", "combin", "s/p", "ratio combine"]):
        return {
            "metric": "sp_ratio_reel_pct",
            "label": "Ratio S/P réel (sinistres/prime nette)",
            "unit": "%",
            "source_kind": "ratio_reel",
            "value_expr": "",
            "proxy_note": "",
        }

    if any(token in normalized_question for token in ["taux resiliation", "taux de resiliation", "churn", "resiliation"]):
        return {
            "metric": "taux_resiliation",
            "label": "Taux de resiliation",
            "unit": "%",
            "source_kind": "churn_rate",
            "value_expr": "",
            "proxy_note": "Calcul du taux de resiliation par mois.",
        }

    if "sinistre" in normalized_question:
        # dwh_fact_sinistre existe — on l'utilise directement
        if amount_signal and not count_signal:
            return {
                "metric": "total_mt_paye_sinistres",
                "label": "Montant sinistres payes",
                "unit": "TND",
                "source_kind": "sinistre",
                "value_expr": "COALESCE(SUM(mt_paye), 0)",
                "proxy_note": "",
            }
        return {
            "metric": "nb_sinistres",
            "label": "Nombre de sinistres",
            "unit": "count",
            "source_kind": "sinistre",
            "value_expr": "COUNT(*)",
            "proxy_note": "",
        }

    if "impaye" in normalized_question:
        if amount_signal and not count_signal:
            return {
                "metric": "total_impaye",
                "label": "Montant impaye",
                "unit": "TND",
                "source_kind": "impaye",
                "value_expr": "COALESCE(SUM(mt_acp), 0)",
                "proxy_note": "",
            }
        return {
            "metric": "nb_impayes",
            "label": "Nombre d impayes",
            "unit": "count",
            "source_kind": "impaye",
            "value_expr": "COUNT(*)",
            "proxy_note": "",
        }

    if "annulation" in normalized_question:
        if amount_signal and not count_signal:
            return {
                "metric": "total_annulation",
                "label": "Montant annulation",
                "unit": "TND",
                "source_kind": "annulation",
                "value_expr": "COALESCE(SUM(mt_ptt_ann), 0)",
                "proxy_note": "",
            }
        return {
            "metric": "nb_annulations",
            "label": "Nombre d annulations",
            "unit": "count",
            "source_kind": "annulation",
            "value_expr": "COUNT(*)",
            "proxy_note": "",
        }

    return {
        "metric": "total_pnet",
        "label": "Prime nette",
        "unit": "TND",
        "source_kind": "emission",
        "value_expr": "COALESCE(SUM(mt_pnet), 0)",
        "proxy_note": "",
    }


def _build_forecast_report_details(
    *,
    target: dict[str, Any],
    branch: str | None,
    year_from: int,
    year_to: int,
    horizon: int,
    trend_pct: float,
    latest_observed: float,
    latest_projected: float,
    predictions: list[dict[str, Any]],
    prediction_key: str,
) -> dict[str, Any]:
    scope_label = "toutes les branches" if not branch else f"la branche {branch}"
    period_label = str(year_from) if year_from == year_to else f"{year_from}-{year_to}"
    unit = str(target.get("unit", "TND"))
    target_label = str(target.get("label", "metrique"))

    context_line = f"Projection {target_label} sur {horizon} mois pour {scope_label} (historique {period_label})."

    if not predictions:
        return {
            "context": context_line,
            "analysis": "Aucune projection disponible sur ce perimetre.",
            "decision": "Impossible de conclure sans projection fiable.",
            "actions": [
                "Elargir l historique ou verifier la disponibilite des donnees mensuelles.",
                "Relancer une projection avec une metrique mieux alimentee.",
            ],
        }

    first_period = str(predictions[0].get("period", ""))
    last_period = str(predictions[-1].get("period", ""))
    analysis_line = (
        f"Scenario central {first_period} a {last_period}: valeur projetee { _format_metric_value(latest_projected, unit) }, "
        f"variation estimee {trend_pct:.2f}% vs dernier observe ({_format_metric_value(latest_observed, unit)})."
    )

    risk_metrics = {"nb_impayes", "total_impaye", "nb_sinistres", "total_mt_paye_sinistres", "nb_annulations", "total_annulation"}
    is_risk_metric = str(target.get("metric", "")) in risk_metrics

    if is_risk_metric:
        if trend_pct >= 5.0:
            decision_line = "Hausse projetee du risque: renforcer la prevention et le recouvrement."
            actions = [
                "Prioriser les plans recouvrement sur les segments les plus contributes.",
                "Renforcer la surveillance hebdomadaire des indicateurs de risque.",
            ]
        elif trend_pct <= -5.0:
            decision_line = "Baisse projetee du risque: consolider les actions deja efficaces."
            actions = [
                "Maintenir les actions de prevention ayant produit la baisse.",
                "Suivre la stabilite mensuelle pour eviter un rebond.",
            ]
        else:

            decision_line = "Risque projete globalement stable: pilotage mensuel a maintenir."
            actions = ["Maintenir un suivi mensuel cible sur les principaux contributeurs au risque."]
    else:

        if trend_pct >= 5.0:
            decision_line = "Croissance projetee de la production: opportunite a capter commercialement."
            actions = [
                "Ajuster les objectifs commerciaux sur la periode projetee.",
                "Verifier la capacite operationnelle pour soutenir la croissance.",
            ]
        elif trend_pct <= -5.0:
            decision_line = "Contraction projetee: un plan de relance est recommande."
            actions = [
                "Lancer des actions de relance sur les branches en recul.",
                "Analyser les causes de baisse sur les mois precedant la projection.",
            ]
        else:

            decision_line = "Trajectoire projetee stable: maintien du pilotage operationnel."
            actions = ["Maintenir un suivi budgetaire mensuel pour detecter rapidement tout ecart."]

    return {
        "context": context_line,
        "analysis": analysis_line,
        "decision": decision_line,
        "actions": actions,
    }


# Maps agent metric keys → prophet_service indicateur names.
_METRIC_TO_INDICATEUR: dict[str, str] = {
    "total_pnet":              "primes_acquises_tnd",
    "total_mt_paye_sinistres": "cout_sinistres_tnd",
    "nb_sinistres":            "nb_sinistres",
    "taux_resiliation":        "taux_resiliation",
    "sp_ratio_reel_pct":       "sp_ratio",
    "total_impaye":            "impayes_tnd",
    "nb_impayes":              "impayes_tnd",
}

_PROPHET_BRANCHES = {"AUTO", "IRDS", "SANTE"}


def _try_prophet_service(
    branch: str | None,
    target: dict[str, Any],
    horizon: int,
) -> dict[str, Any] | None:
    """
    Attempt to get the forecast from the pre-trained prophet_service pkl.
    Returns the prophet_service result dict on success, None if unavailable.
    Uses the same model as the MLOps forecast tab.
    """
    indicateur = _METRIC_TO_INDICATEUR.get(target.get("metric", ""))
    if not indicateur:
        return None

    dept = branch if branch in _PROPHET_BRANCHES else "AUTO"
    try:
        from ml_services.prophet_service import get_forecast
        result = get_forecast(departement=dept, indicateur=indicateur, nb_mois=horizon)
        if "error" in result:
            return None
        return result
    except Exception:
        return None


def _format_prophet_result(
    ps_result: dict[str, Any],
    target: dict[str, Any],
    branch: str | None,
    horizon: int,
) -> dict[str, Any]:
    """Convert a prophet_service result dict into the agent tool output format."""
    historique = ps_result.get("historique", [])
    previsions = ps_result.get("previsions", [])
    methode    = ps_result.get("methode", "prophet_service")
    mape       = ps_result.get("mape")
    target_unit = str(target.get("unit", "TND"))
    prediction_key = f"{target['metric']}_pred"

    derniere   = float(ps_result.get("derniere_valeur",  0.0))
    prochaine  = float(ps_result.get("prochaine_valeur", 0.0))
    trend_pct  = ((prochaine - derniere) / derniere * 100) if derniere > 0 else 0.0
    total_fc   = sum(float(p.get("valeur", 0)) for p in previsions)
    avg_fc     = total_fc / len(previsions) if previsions else 0.0

    # Build unified chart items: historical + forecast
    hist_items = [
        {
            "period": h.get("periode", ""),
            "actual": _safe_float(h.get("valeur"), 0.0),
            prediction_key: None,
            "combined_value": _safe_float(h.get("valeur"), 0.0),
        }
        for h in historique
    ]
    fc_items = [
        {
            "period": p.get("periode", ""),
            "actual": None,
            prediction_key: _safe_float(p.get("valeur"), 0.0),
            "combined_value": _safe_float(p.get("valeur"), 0.0),
        }
        for p in previsions
    ]
    # Anchor forecast line at last historical point for visual continuity
    if hist_items and fc_items:
        hist_items[-1][prediction_key] = hist_items[-1]["actual"]

    combined = hist_items + fc_items

    predictions = [
        {
            "period": p.get("periode", ""),
            prediction_key: _safe_float(p.get("valeur"), 0.0),
        }
        for p in previsions
    ]

    report = _build_forecast_report_details(
        target=target,
        branch=branch,
        year_from=0,
        year_to=0,
        horizon=horizon,
        trend_pct=trend_pct,
        latest_observed=derniere,
        latest_projected=prochaine,
        predictions=predictions,
        prediction_key=prediction_key,
    )

    mape_str = f", MAPE={mape:.1f}%" if mape is not None else ""
    kpis = [
        {"key": "projection_totale",      "label": "Projection cumulee",               "value": total_fc, "unit": target_unit},
        {"key": "projection_moyenne",     "label": "Projection moyenne mensuelle",      "value": avg_fc,   "unit": target_unit},
        {"key": "variation_projection_pct","label": "Variation projetee vs dernier obs","value": trend_pct,"unit": "%"},
        {"key": "horizon_mois",           "label": "Horizon de projection",             "value": float(horizon), "unit": "count"},
    ]

    chart_payload = _build_chart_payload(
        chart_type="line",
        title=f"Forecast {target['label']} ({methode}{mape_str})",
        x_key="period",
        y_key="combined_value",
        items=combined,
    )
    chart_payload["series"] = [
        {"key": "actual",        "label": "Historique", "color": "#0f766e",  "strokeWidth": 2.2},
        {"key": prediction_key,  "label": "Prevision",  "color": "#dc2626",  "strokeDasharray": "8 5", "strokeWidth": 2.8, "dot": True},
    ]
    chart_payload["forecast_start_period"] = fc_items[0]["period"] if fc_items else None

    table_cols = ["period", prediction_key]

    return {
        "tool": "forecast_tool",
        "summary": report["analysis"],
        "payload": {
            "branch":        branch or "ALL",
            "horizon_months": horizon,
            "engine":        methode,
            "target_metric": target["metric"],
            "target_label":  target["label"],
            "target_unit":   target_unit,
            "trend_pct":     trend_pct,
            "kpis":          kpis,
            "context":       report.get("context", ""),
            "analysis":      report.get("analysis", ""),
            "decision":      report.get("decision", ""),
            "actions":       report.get("actions", []),
            "history":       hist_items,
            "predictions":   predictions,
        },
        "charts": [chart_payload],
        "tables": [
            {
                "title": f"Projection {target['label']}",
                "columns": table_cols,
                "rows": predictions,
                "markdown": _to_markdown_table(table_cols, predictions),
            }
        ],
    }


def forecast_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)

    # Assurer un historique suffisant pour la prediction (remonter a 2019)
    if "year_from" not in context:
        year_from = min(year_from, 2019)

    horizon = max(1, min(_safe_int(context.get("horizon_months"), 3), 12))
    report_mode = _infer_forecast_report_mode(question)

    normalized_question = _normalize_text(question)
    target = _detect_forecast_target(normalized_question)

    prediction_key = f"{target['metric']}_pred"

    # --- Fast path: delegate to pre-trained prophet_service pkl (same as MLOps tab) ---
    ps_result = _try_prophet_service(branch, target, horizon)
    if ps_result is not None:
        return _format_prophet_result(ps_result, target, branch, horizon)
    # --- Fallback: DWH → Prophet / linear regression (used for annulations and
    #     any metric not covered by the pkl, or when the pkl is unavailable) ---

    if target["source_kind"] == "sinistre":
        sql_query = f"""
            SELECT
                annee_survenance AS annee_echeance,
                mois_survenance  AS mois_echeance,
                make_date(annee_survenance, mois_survenance, 1) AS period,
                {target['value_expr']} AS metric_value
            FROM dwh_fact_sinistre
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_survenance BETWEEN :year_from AND :year_to
              AND annee_survenance BETWEEN 1900 AND 2100
              AND mois_survenance BETWEEN 1 AND 12
            GROUP BY annee_survenance, mois_survenance,
                     make_date(annee_survenance, mois_survenance, 1)
            ORDER BY period
        """

    elif target["source_kind"] == "impaye":
        sql_query = f"""
            SELECT
                annee_echeance,
                mois_echeance,
                make_date(annee_echeance, mois_echeance, 1) AS period,
                {target['value_expr']} AS metric_value
            FROM dwh_fact_impaye
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN :year_from AND :year_to
              AND annee_echeance BETWEEN 1900 AND 2100
              AND mois_echeance BETWEEN 1 AND 12
            GROUP BY annee_echeance, mois_echeance, make_date(annee_echeance, mois_echeance, 1)
            ORDER BY period
        """
    elif target["source_kind"] == "annulation":
        sql_query = f"""
            SELECT
                annee_annulation AS annee_echeance,
                mois_annulation AS mois_echeance,
                make_date(annee_annulation, mois_annulation, 1) AS period,
                {target['value_expr']} AS metric_value
            FROM dwh_fact_annulation
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_annulation BETWEEN :year_from AND :year_to
              AND annee_annulation BETWEEN 1900 AND 2100
              AND mois_annulation BETWEEN 1 AND 12
            GROUP BY annee_annulation, mois_annulation, make_date(annee_annulation, mois_annulation, 1)
            ORDER BY period
        """
    elif target["source_kind"] == "churn_rate":
        sql_query = """
            WITH monthly_emission AS (
                SELECT
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COUNT(DISTINCT id_police) AS total_polices
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND annee_echeance BETWEEN :year_from AND :year_to
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
            ),
            monthly_resiliation AS (
                SELECT
                    make_date(annee_annulation, mois_annulation, 1) AS period,
                    COUNT(DISTINCT id_police) AS polices_resiliees
                FROM dwh_fact_annulation
                WHERE (:branch IS NULL OR branche = :branch)
                  AND annee_annulation BETWEEN :year_from AND :year_to
                  AND annee_annulation BETWEEN 1900 AND 2100
                  AND mois_annulation BETWEEN 1 AND 12
                GROUP BY make_date(annee_annulation, mois_annulation, 1)
            )
            SELECT
                COALESCE(e.period, r.period) AS period,
                CASE WHEN COALESCE(e.total_polices, 0) > 0
                     THEN (COALESCE(r.polices_resiliees, 0)::numeric / e.total_polices) * 100.0
                     ELSE 0.0 END AS metric_value
            FROM monthly_emission e
            FULL OUTER JOIN monthly_resiliation r ON e.period = r.period
            ORDER BY period
        """
    elif target["source_kind"] in {"ratio", "ratio_reel"}:
        # Ratio S/P RÉEL = sinistres payés / prime nette (par mois)
        sql_query = """
            WITH monthly_emission AS (
                SELECT
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND annee_echeance BETWEEN :year_from AND :year_to
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
            ),
            monthly_sinistre AS (
                SELECT
                    make_date(annee_survenance, mois_survenance, 1) AS period,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND annee_survenance BETWEEN :year_from AND :year_to
                  AND annee_survenance BETWEEN 1900 AND 2100
                  AND mois_survenance BETWEEN 1 AND 12
                GROUP BY make_date(annee_survenance, mois_survenance, 1)
            )
            SELECT
                COALESCE(e.period, s.period) AS period,
                CASE WHEN COALESCE(e.total_pnet, 0) > 0
                     THEN (COALESCE(s.total_mt_paye, 0) / e.total_pnet) * 100.0
                     ELSE 0.0
                END AS metric_value
            FROM monthly_emission e
            FULL OUTER JOIN monthly_sinistre s ON e.period = s.period
            ORDER BY period
        """
    else:

        sql_query = """
            SELECT
                annee_echeance,
                mois_echeance,
                make_date(annee_echeance, mois_echeance, 1) AS period,
                COALESCE(SUM(mt_pnet), 0) AS metric_value
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN :year_from AND :year_to
              AND annee_echeance BETWEEN 1900 AND 2100
              AND mois_echeance BETWEEN 1 AND 12
            GROUP BY annee_echeance, mois_echeance, make_date(annee_echeance, mois_echeance, 1)
            ORDER BY period
        """

    df = _query_dataframe(sql_query, {"branch": branch, "year_from": year_from, "year_to": year_to})
    result_kind = "timeseries"
    target_unit = str(target.get("unit", "TND"))

    if len(df) < 6:
        context_line = (
            f"Projection {target['label']} sur {horizon} mois pour "
            f"{'toutes les branches' if not branch else f'la branche {branch}'} "
            f"(historique {year_from}-{year_to})."
        )
        return {
            "tool": "forecast_tool",
            "summary": "Donnees insuffisantes pour une projection fiable (minimum 6 points mensuels).",
            "payload": {
                "predictions": [],
                "horizon_months": horizon,
                "engine": "none",
                "target_metric": target["metric"],
                "target_label": target["label"],
                "target_unit": target_unit,
                "proxy_note": target["proxy_note"],
                "report_mode": report_mode,
                "result_kind": result_kind,
                "kpis": [],
                "context": context_line,
                "analysis": "Donnees insuffisantes pour calculer une projection robuste sur ce perimetre.",
                "decision": "Projection non exploitable en l etat; consolidation de donnees requise.",
                "actions": [
                    "Elargir l historique de donnees mensuelles sur la metrique cible.",
                    "Relancer la projection apres verification de la qualite des donnees.",
                ],
            },
            "charts": [],
            "tables": [],
        }

    predictions: list[dict[str, Any]] = []
    engine = "linear_regression_fallback"

    try:
        prophet_module = importlib.import_module("prophet")
        Prophet = getattr(prophet_module, "Prophet")

        prophet_df = pd.DataFrame({"ds": pd.to_datetime(df["period"]), "y": df["metric_value"].astype(float)})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future).tail(horizon)
        predictions = [
            {
                "period": pd.Timestamp(row["ds"]).strftime("%Y-%m"),
                prediction_key: _safe_float(row["yhat"]),
            }
            for _, row in forecast.iterrows()
        ]
        engine = "prophet"
    except Exception:
        y_values = df["metric_value"].astype(float).to_numpy()
        x_values = np.arange(len(y_values), dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)

        last_period = pd.Timestamp(df.iloc[-1]["period"])
        for step in range(1, horizon + 1):
            projected = max(0.0, slope * (len(y_values) - 1 + step) + intercept)
            next_period = last_period + pd.DateOffset(months=step)
            predictions.append(
                {
                    "period": next_period.strftime("%Y-%m"),
                    prediction_key: float(projected),
                }
            )

    latest_observed = _safe_float(df.iloc[-1]["metric_value"], 0.0)
    latest_projected = _safe_float(predictions[-1][prediction_key], 0.0) if predictions else latest_observed
    trend_pct = (100.0 * (latest_projected - latest_observed) / latest_observed) if latest_observed > 0 else 0.0

    total_forecast = sum(_safe_float(item.get(prediction_key), 0.0) for item in predictions if isinstance(item, dict))
    average_forecast = (total_forecast / len(predictions)) if predictions else 0.0

    report_details = _build_forecast_report_details(
        target=target,
        branch=branch,
        year_from=year_from,
        year_to=year_to,
        horizon=horizon,
        trend_pct=trend_pct,
        latest_observed=latest_observed,
        latest_projected=latest_projected,
        predictions=predictions,
        prediction_key=prediction_key,
    )

    kpis = [
        {
            "key": "projection_totale",
            "label": "Projection cumulee",
            "value": total_forecast,
            "unit": target_unit,
        },
        {
            "key": "projection_moyenne",
            "label": "Projection moyenne mensuelle",
            "value": average_forecast,
            "unit": target_unit,
        },
        {
            "key": "variation_projection_pct",
            "label": "Variation projetee vs dernier observe",
            "value": trend_pct,
            "unit": "%",
        },
        {
            "key": "horizon_mois",
            "label": "Horizon de projection",
            "value": float(horizon),
            "unit": "count",
        },
    ]

    include_chart = report_mode in {"report", "graph_only", "graph_pref"}
    include_table = report_mode in {"report", "table_only", "table_pref"}

    historical_points = [
        {
            "period": pd.Timestamp(row["period"]).strftime("%Y-%m"),
            "actual": _safe_float(row.get("metric_value"), 0.0),
            prediction_key: None,
            "combined_value": _safe_float(row.get("metric_value"), 0.0),
        }
        for _, row in df.iterrows()
    ]

    forecast_points = [
        {
            "period": str(item.get("period", "")),
            "actual": None,
            prediction_key: _safe_float(item.get(prediction_key), 0.0),
            "combined_value": _safe_float(item.get(prediction_key), 0.0),
        }
        for item in predictions
        if isinstance(item, dict)
    ]

    # Anchor forecast line to the last observed value so the historical->forecast transition is continuous.
    if historical_points and forecast_points:
        historical_points[-1][prediction_key] = _safe_float(historical_points[-1].get("actual"), 0.0)

    combined_chart_items = historical_points + forecast_points

    charts: list[dict[str, Any]] = []
    if include_chart and predictions:
        chart_payload = _build_chart_payload(
            chart_type="line",
            title=f"Forecast {target['label']}",
            x_key="period",
            y_key="combined_value",
            items=combined_chart_items,
        )
        chart_payload["series"] = [
            {
                "key": "actual",
                "label": "Historique",
                "color": "#0f766e",
                "strokeWidth": 2.2,
            },
            {
                "key": prediction_key,
                "label": "Prevision",
                "color": "#dc2626",
                "strokeDasharray": "8 5",
                "strokeWidth": 2.8,
                "dot": True,
            },
        ]
        chart_payload["forecast_start_period"] = forecast_points[0]["period"] if forecast_points else None
        charts.append(chart_payload)

    tables: list[dict[str, Any]] = []
    if include_table and predictions:
        table_columns = ["period", prediction_key]
        tables.append(
            {
                "title": f"Projection {target['label']}",
                "columns": table_columns,
                "rows": predictions,
                "markdown": _to_markdown_table(table_columns, predictions),
            }
        )

    summary_parts = [report_details["analysis"]]
    if target["proxy_note"]:
        summary_parts.append(target["proxy_note"])
    summary = " ".join(summary_parts)

    return {
        "tool": "forecast_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "horizon_months": horizon,
            "engine": engine,
            "target_metric": target["metric"],
            "target_label": target["label"],
            "target_unit": target_unit,
            "proxy_note": target["proxy_note"],
            "trend_pct": trend_pct,
            "report_mode": report_mode,
            "result_kind": result_kind,
            "kpis": kpis,
            "context": report_details.get("context", ""),
            "analysis": report_details.get("analysis", ""),
            "decision": report_details.get("decision", ""),
            "actions": report_details.get("actions", []),
            "history": historical_points,
            "predictions": predictions,
        },
        "charts": charts,
        "tables": tables,
    }
