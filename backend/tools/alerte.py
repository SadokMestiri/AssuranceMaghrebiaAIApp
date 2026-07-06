from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from db import query_dataframe as _query_dataframe
from ml_pipeline import get_impaye_operations_readiness
from utils import safe_float as _safe_float
from config import ALERTE_IMPAYE_RATE_PCT, ALERTE_PRODUCTION_DROP_PCT, DATA_YEAR_TO
from tools._shared import (
    _normalize_branch,
    _to_markdown_table,
    _build_chart_payload,
)


def alerte_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    threshold_impaye_rate = ALERTE_IMPAYE_RATE_PCT
    threshold_drop_pct    = ALERTE_PRODUCTION_DROP_PCT

    # Anchor to the last month of available data, not today, so the 6-month
    # window always lands inside the dataset regardless of when the app runs.
    data_end = f"{DATA_YEAR_TO}-12-01"

    sql_query = """
        WITH monthly_emission AS (
            SELECT
                make_date(annee_echeance, mois_echeance, 1) AS period,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN 1900 AND 2100
              AND mois_echeance BETWEEN 1 AND 12
              AND make_date(annee_echeance, mois_echeance, 1)
                  BETWEEN CAST(:data_end AS date) - interval '6 months' AND CAST(:data_end AS date)
            GROUP BY make_date(annee_echeance, mois_echeance, 1)
        ),
        monthly_impaye AS (
            SELECT
                make_date(annee_echeance, mois_echeance, 1) AS period,
                COALESCE(SUM(mt_acp), 0) AS total_impaye
            FROM dwh_fact_impaye
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN 1900 AND 2100
              AND mois_echeance BETWEEN 1 AND 12
              AND make_date(annee_echeance, mois_echeance, 1)
                  BETWEEN CAST(:data_end AS date) - interval '6 months' AND CAST(:data_end AS date)
            GROUP BY make_date(annee_echeance, mois_echeance, 1)
        )
        SELECT
            COALESCE(e.period, i.period) AS period,
            COALESCE(e.total_pnet, 0) AS total_pnet,
            COALESCE(i.total_impaye, 0) AS total_impaye,
            ROUND(100.0 * COALESCE(i.total_impaye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0), 3) AS impaye_rate_pct
        FROM monthly_emission e
        FULL OUTER JOIN monthly_impaye i ON i.period = e.period
        ORDER BY period
    """

    monthly_df = _query_dataframe(sql_query, {"branch": branch, "data_end": data_end})
    monthly_items = [
        {
            "period": str(row["period"])[:10],
            "total_pnet": _safe_float(row["total_pnet"]),
            "total_impaye": _safe_float(row["total_impaye"]),
            "impaye_rate_pct": _safe_float(row["impaye_rate_pct"]),
        }
        for _, row in monthly_df.iterrows()
    ]

    alerts: list[dict[str, Any]] = []

    if monthly_items:
        latest = monthly_items[-1]
        if latest["impaye_rate_pct"] >= threshold_impaye_rate:
            alerts.append(
                {
                    "severity": "high",
                    "type": "impaye_rate",
                    "message": (
                        f"Taux impaye {latest['impaye_rate_pct']:.2f}% au-dessus du seuil "
                        f"{threshold_impaye_rate:.2f}%."
                    ),
                    "period": latest["period"],
                }
            )

        if len(monthly_items) >= 2:
            prev_values = [item["total_pnet"] for item in monthly_items[:-1] if item["total_pnet"] > 0]
            if prev_values and latest["total_pnet"] > 0:
                avg_prev = float(np.mean(prev_values))
                drop_pct = (100.0 * (avg_prev - latest["total_pnet"]) / avg_prev) if avg_prev > 0 else 0.0
                if drop_pct >= threshold_drop_pct:
                    alerts.append(
                        {
                            "severity": "medium",
                            "type": "production_drop",
                            "message": (
                                f"Baisse de production {drop_pct:.2f}% par rapport a la moyenne recente."
                            ),
                            "period": latest["period"],
                        }
                    )

    try:
        readiness = get_impaye_operations_readiness(months=6)
        readiness_status = str(readiness.get("readiness", {}).get("status", "unavailable")).lower()
        readiness_score = _safe_float(readiness.get("readiness", {}).get("score"), 0.0)
        if readiness_status in {"red", "amber"}:
            alerts.append(
                {
                    "severity": "high" if readiness_status == "red" else "medium",
                    "type": "ml_readiness",
                    "message": f"Readiness modele {readiness_status} (score {readiness_score:.1f}/100).",
                    "period": str(datetime.now(timezone.utc).date()),
                }
            )
    except Exception:
        pass

    summary = (
        f"Alerte tool: {len(alerts)} alertes detectees sur les 6 derniers mois."
        if alerts
        else "Alerte tool: aucun signal critique detecte sur les 6 derniers mois."
    )

    return {
        "tool": "alerte_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "alerts": alerts,
            "monthly_metrics": monthly_items,
            "thresholds": {
                "impaye_rate_pct": threshold_impaye_rate,
                "production_drop_pct": threshold_drop_pct,
            },
        },
        "charts": [
            _build_chart_payload(
                chart_type="line",
                title="Taux impaye recent",
                x_key="period",
                y_key="impaye_rate_pct",
                items=monthly_items,
            )
        ],
        "tables": [
            {
                "title": "Alertes recentes",
                "columns": ["severity", "type", "period", "message"],
                "rows": alerts,
                "markdown": _to_markdown_table(["severity", "type", "period", "message"], alerts),
            }
        ],
    }
