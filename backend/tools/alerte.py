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
from tools.kpi import _fetch_kpi_context_postgres
from auth import ROLES

# Same thresholds already narrated in agent_graph._compose_decision_answer
# ("Ratio combine au-dessus de 100%: activite deficitaire.", etc.) — reused
# here rather than re-invented, so the alert feed agrees with what the agent
# would say about the same numbers.
RATIO_COMBINE_CRITICAL_PCT = 100.0
RATIO_COMBINE_WARNING_PCT  = 80.0
RESILIATION_CRITICAL_PCT   = 8.0
RESILIATION_WARNING_PCT    = 4.0

# New-signup alerts are events, not threshold breaches — "info" severity,
# distinct from the "high"/"medium" business alerts above. 30 days rather than
# 7: "newly created" accounts should stay visible long enough to actually be
# noticed on an occasionally-checked dashboard (a 7-day window silently dropped
# accounts before anyone looked).
NEW_USER_ALERT_WINDOW_DAYS = 30


def compute_alerts(branch: str | None, months: int = 12) -> dict[str, Any]:
    """
    Core alert-detection logic — shared by alerte_tool (agent path, only
    reachable through a chat question) and the standalone GET /alerts route
    (agent_router-free, cheap enough to poll from the sidebar/alerts tab).

    Evaluates EVERY month in the window against each threshold (not just the
    latest one) so there's an actual history to filter/paginate in the UI,
    rather than at most one alert per signal type.
    """
    months = max(3, min(int(months), 36))
    threshold_impaye_rate = ALERTE_IMPAYE_RATE_PCT
    threshold_drop_pct    = ALERTE_PRODUCTION_DROP_PCT

    # Anchor to the last month of available data, not today, so the window
    # always lands inside the dataset regardless of when the app runs.
    data_end = f"{DATA_YEAR_TO}-12-01"

    sql_query = f"""
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
                  BETWEEN CAST(:data_end AS date) - interval '{months} months' AND CAST(:data_end AS date)
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
                  BETWEEN CAST(:data_end AS date) - interval '{months} months' AND CAST(:data_end AS date)
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

    # Impaye rate — one alert per month that breaches the threshold, not just
    # the latest one.
    for item in monthly_items:
        if item["impaye_rate_pct"] >= threshold_impaye_rate:
            alerts.append(
                {
                    "severity": "high",
                    "type": "impaye_rate",
                    "message": (
                        f"Taux impaye {item['impaye_rate_pct']:.2f}% au-dessus du seuil "
                        f"{threshold_impaye_rate:.2f}%."
                    ),
                    "period": item["period"],
                }
            )

    # Production drop — rolling comparison: each month vs the average of all
    # months before it in the window, so a drop anywhere in the history
    # surfaces, not only if it happened to be the most recent month.
    for i in range(1, len(monthly_items)):
        prev_values = [m["total_pnet"] for m in monthly_items[:i] if m["total_pnet"] > 0]
        current = monthly_items[i]
        if prev_values and current["total_pnet"] > 0:
            avg_prev = float(np.mean(prev_values))
            drop_pct = (100.0 * (avg_prev - current["total_pnet"]) / avg_prev) if avg_prev > 0 else 0.0
            if drop_pct >= threshold_drop_pct:
                alerts.append(
                    {
                        "severity": "medium",
                        "type": "production_drop",
                        "message": f"Baisse de production {drop_pct:.2f}% par rapport a la moyenne recente.",
                        "period": current["period"],
                    }
                )

    # Ratio combine + taux de resiliation — current-state snapshot on the
    # latest year of data (not a monthly history like impaye/production
    # above: these two are only ever computed over a period range, there's
    # no ready-made monthly breakdown to iterate the way _fetch_kpi_context_postgres
    # exposes them).
    try:
        kpi_context = _fetch_kpi_context_postgres(
            {"branch": branch, "year_from": DATA_YEAR_TO, "year_to": DATA_YEAR_TO}
        )
        ratio_combine = _safe_float(kpi_context.get("ratio_combine_pct"), 0.0)
        taux_resiliation = _safe_float(kpi_context.get("taux_resiliation_pct"), 0.0)
        snapshot_period = f"{DATA_YEAR_TO}-12-01"

        if ratio_combine >= RATIO_COMBINE_CRITICAL_PCT:
            alerts.append(
                {
                    "severity": "high",
                    "type": "ratio_combine",
                    "message": f"Ratio combine a {ratio_combine:.2f}% (>= {RATIO_COMBINE_CRITICAL_PCT:.0f}%) : activite deficitaire sur {DATA_YEAR_TO}.",
                    "period": snapshot_period,
                }
            )
        elif ratio_combine >= RATIO_COMBINE_WARNING_PCT:
            alerts.append(
                {
                    "severity": "medium",
                    "type": "ratio_combine",
                    "message": f"Ratio combine a {ratio_combine:.2f}% : marge technique sous pression sur {DATA_YEAR_TO}.",
                    "period": snapshot_period,
                }
            )

        if taux_resiliation >= RESILIATION_CRITICAL_PCT:
            alerts.append(
                {
                    "severity": "high",
                    "type": "resiliation",
                    "message": f"Taux de resiliation a {taux_resiliation:.2f}% (>= {RESILIATION_CRITICAL_PCT:.0f}%) : risque retention eleve.",
                    "period": snapshot_period,
                }
            )
        elif taux_resiliation >= RESILIATION_WARNING_PCT:
            alerts.append(
                {
                    "severity": "medium",
                    "type": "resiliation",
                    "message": f"Taux de resiliation a {taux_resiliation:.2f}% : surveillance requise.",
                    "period": snapshot_period,
                }
            )
    except Exception:
        pass

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

    # New signups — an event, not a threshold breach, so severity "info"
    # rather than "high"/"medium". Not branch-scoped (accounts aren't tied
    # to a branch), so this runs regardless of the branch filter.
    try:
        users_df = _query_dataframe(
            f"""
                SELECT email, nom, prenom, role, created_at
                FROM users
                WHERE created_at >= NOW() - INTERVAL '{NEW_USER_ALERT_WINDOW_DAYS} days'
                ORDER BY created_at DESC
            """
        )
        for _, row in users_df.iterrows():
            role_label = ROLES.get(str(row["role"]), str(row["role"]))
            alerts.append(
                {
                    "severity": "info",
                    "type": "new_user",
                    "message": f"Nouvel utilisateur inscrit : {row['prenom']} {row['nom']} ({role_label}).",
                    "period": str(row["created_at"])[:10],
                }
            )
    except Exception:
        pass

    alerts.sort(key=lambda a: a["period"], reverse=True)

    return {
        "branch": branch or "ALL",
        "alerts": alerts,
        "monthly_metrics": monthly_items,
        "thresholds": {
            "impaye_rate_pct": threshold_impaye_rate,
            "production_drop_pct": threshold_drop_pct,
        },
    }


def alerte_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    result = compute_alerts(branch)
    alerts = result["alerts"]
    monthly_items = result["monthly_metrics"]

    summary = (
        f"Alerte tool: {len(alerts)} alertes detectees sur les 6 derniers mois."
        if alerts
        else "Alerte tool: aucun signal critique detecte sur les 6 derniers mois."
    )

    return {
        "tool": "alerte_tool",
        "summary": summary,
        "payload": result,
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
