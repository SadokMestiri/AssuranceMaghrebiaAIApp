from __future__ import annotations

from typing import Any

from tools._shared import _normalize_branch, _build_chart_payload, _to_markdown_table
from utils import safe_int as _safe_int


def anomaly_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Delegates to ml_services.anomaly_service — 4-algorithm consensus (IF + LOF + PCA-AE + DBSCAN)
    on individual contracts, same model used by the MLOps tab.
    The previous implementation ran a single Isolation Forest on monthly aggregates,
    which diverged from the MLOps results.
    """
    from ml_services.anomaly_service import detect_anomalies

    branch = _normalize_branch(context.get("branch"))
    # min_score=2 requires at least 2 of 4 algorithms to flag a contract (reduces false positives)
    min_score = _safe_int(context.get("min_anomaly_score"), 2)

    try:
        result = detect_anomalies(
            departement=branch or None,
            contamination=0.05,
            min_score=min_score,
        )
    except Exception as exc:
        return {
            "tool": "anomaly_tool",
            "summary": f"Erreur detection anomalies: {exc}",
            "payload": {"anomalies": [], "engine": "4algo_consensus"},
        }

    if "error" in result:
        return {
            "tool": "anomaly_tool",
            "summary": result["error"],
            "payload": {"anomalies": [], "engine": "4algo_consensus"},
        }

    anomalies = result.get("anomalies", [])
    nb_anomalies = result.get("nb_anomalies", 0)
    nb_contracts = result.get("nb_contracts_analysed", 0)
    score_4 = result.get("score_4", 0)
    score_3 = result.get("score_3", 0)

    if nb_anomalies > 0:
        summary = (
            f"4-algo consensus: {nb_anomalies} contrats anomaux sur {nb_contracts} analyses "
            f"(score=4 critique: {score_4}, score=3 eleve: {score_3})."
        )
    else:
        summary = (
            f"Aucune anomalie contractuelle detectee (seuil min_score={min_score}) "
            f"sur {nb_contracts} contrats analyses."
        )

    table_rows = [
        {
            "id_police":   a.get("id_police", "—"),
            "branche":     a.get("branche", "—"),
            "score":       a.get("anomaly_score", 0),
            "loss_ratio":  a.get("loss_ratio", 0),
            "taux_impaye": a.get("taux_impaye", 0),
            "client":      a.get("client_nom", "—"),
        }
        for a in anomalies[:20]
    ]
    cols = ["id_police", "branche", "score", "loss_ratio", "taux_impaye", "client"]

    score_dist = [
        {"score": "Score 4 (4/4 algo)", "nb_contrats": result.get("score_4", 0)},
        {"score": "Score 3 (3/4 algo)", "nb_contrats": result.get("score_3", 0)},
        {"score": "Score 2 (2/4 algo)", "nb_contrats": result.get("score_2", 0)},
        {"score": "Score 1 (1/4 algo)", "nb_contrats": result.get("score_1", 0)},
    ]

    return {
        "tool": "anomaly_tool",
        "summary": summary,
        "payload": {
            "branch":                branch or "ALL",
            "engine":                "4algo_consensus",
            "nb_contracts_analysed": nb_contracts,
            "nb_anomalies":          nb_anomalies,
            "score_4":               score_4,
            "score_3":               score_3,
            "anomalies":             anomalies[:50],
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Distribution scores anomalie (consensus 4 algorithmes)",
                x_key="score",
                y_key="nb_contrats",
                items=score_dist,
            )
        ],
        "tables": (
            [
                {
                    "title": f"Top contrats anomaux (score >= {min_score})",
                    "columns": cols,
                    "rows": table_rows,
                    "markdown": _to_markdown_table(cols, table_rows),
                }
            ]
            if table_rows
            else []
        ),
    }
