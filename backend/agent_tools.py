from __future__ import annotations

import json
import importlib
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sqlalchemy import text

from db import engine as db_engine
from ml_pipeline import FEATURE_COLUMNS, get_impaye_operations_readiness, load_model_metadata, load_training_dataset


VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}
RAG_DOCUMENTS_PATH = Path(__file__).resolve().parent / "rag_documents.json"
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

DENODO_KPI_API_URL = os.getenv("DENODO_KPI_API_URL", "").strip()
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
QDRANT_COLLECTIONS = [
    item.strip()
    for item in os.getenv("QDRANT_COLLECTIONS", "business_rules,kpi_history,alert_history").split(",")
    if item.strip()
]
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _normalize_branch(branch: str | None) -> str | None:
    if not branch or branch.strip().upper() == "ALL":
        return None
    normalized = branch.strip().upper()
    if normalized not in VALID_BRANCHES:
        return None
    return normalized


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_period_context(context: dict[str, Any]) -> tuple[int, int]:
    now_year = datetime.now(timezone.utc).year
    year_from = _safe_int(context.get("year_from"), now_year - 1)
    year_to = _safe_int(context.get("year_to"), now_year)
    if year_from > year_to:
        year_from, year_to = year_to, year_from
    return year_from, year_to


def _query_dataframe(sql_query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    return pd.read_sql(text(sql_query), db_engine, params=params or {})


def _tokenize(text_value: str) -> set[str]:
    return {token.lower() for token in WORD_PATTERN.findall(text_value) if len(token) > 2}


def _normalize_text(text_value: str) -> str:
    ascii_normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", text_value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", ascii_normalized.strip().lower())


def _load_rag_documents() -> list[dict[str, Any]]:
    if not RAG_DOCUMENTS_PATH.exists():
        return []
    with RAG_DOCUMENTS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [doc for doc in payload if isinstance(doc, dict)]


def _to_markdown_table(columns: list[str], rows: list[dict[str, Any]], max_rows: int = 8) -> str:
    if not columns or not rows:
        return ""

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:max_rows]:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _get_ollama_embedding(text_value: str) -> list[float] | None:
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text_value},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if isinstance(embedding, list) and embedding:
            return [float(x) for x in embedding]
    except Exception:
        return None
    return None


def _qdrant_semantic_search(question: str, top_k: int) -> list[dict[str, Any]]:
    embedding = _get_ollama_embedding(question)
    if not embedding:
        return []

    hits: list[dict[str, Any]] = []
    for collection in QDRANT_COLLECTIONS:
        try:
            response = requests.post(
                f"{QDRANT_URL}/collections/{collection}/points/search",
                json={
                    "vector": embedding,
                    "limit": top_k,
                    "with_payload": True,
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("result", []):
                point_payload = item.get("payload") or {}
                content = (
                    point_payload.get("content")
                    or point_payload.get("text")
                    or point_payload.get("summary")
                    or ""
                )
                if not content:
                    continue
                hits.append(
                    {
                        "source": "qdrant",
                        "collection": collection,
                        "title": point_payload.get("title", point_payload.get("id", collection)),
                        "content": str(content),
                        "score": _safe_float(item.get("score"), 0.0),
                    }
                )
        except Exception:
            continue

    hits.sort(key=lambda value: _safe_float(value.get("score"), 0.0), reverse=True)
    return hits[:top_k]


def _lexical_rag_search(question: str, top_k: int) -> list[dict[str, Any]]:
    documents = _load_rag_documents()
    question_tokens = _tokenize(question)
    if not question_tokens:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        content = f"{document.get('title', '')} {document.get('content', '')}"
        doc_tokens = _tokenize(content)
        overlap = len(question_tokens.intersection(doc_tokens))
        if overlap <= 0:
            continue
        score = overlap / max(len(question_tokens), 1)
        scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "source": "local_rag",
            "collection": "rag_documents",
            "title": doc.get("title", "Document"),
            "content": doc.get("content", ""),
            "score": float(score),
        }
        for score, doc in scored[:top_k]
    ]


def _extract_client_name(question: str, context: dict[str, Any]) -> str | None:
    explicit = context.get("client_name")
    if explicit:
        candidate = str(explicit).strip()
        if candidate:
            return candidate

    quoted = re.search(r"['\"]([^'\"]{3,})['\"]", question)
    if quoted:
        return quoted.group(1).strip()

    named = re.search(r"client\s+([A-Za-z\-\s]{3,})", question, flags=re.IGNORECASE)
    if named:
        return named.group(1).strip()

    return None


def _build_chart_payload(chart_type: str, title: str, x_key: str, y_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": chart_type,
        "title": title,
        "x_key": x_key,
        "y_key": y_key,
        "items": items,
    }


def _fetch_kpi_context_postgres(context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    params = {
        "branch": branch,
        "year_from": year_from,
        "year_to": year_to,
    }

    production_sql = """
        SELECT
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(mt_commission), 0) AS total_commission,
            COUNT(*) AS nb_quittances
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
    """

    impaye_sql = """
        SELECT
            COALESCE(SUM(mt_acp), 0) AS total_mt_acp,
            COUNT(*) AS nb_impayes
        FROM dwh_fact_impaye
        WHERE (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
    """

    churn_sql = """
        WITH portfolio AS (
            SELECT
                COUNT(DISTINCT id_police) AS total_polices
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN :year_from AND :year_to
              AND annee_echeance BETWEEN 1900 AND 2100
              AND mois_echeance BETWEEN 1 AND 12
        ),
        resiliation AS (
            SELECT
                COUNT(DISTINCT id_police) AS polices_resiliees
            FROM dwh_fact_annulation
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_annulation BETWEEN :year_from AND :year_to
              AND annee_annulation BETWEEN 1900 AND 2100
              AND mois_annulation BETWEEN 1 AND 12
        )
        SELECT
            COALESCE(p.total_polices, 0) AS total_polices,
            COALESCE(r.polices_resiliees, 0) AS polices_resiliees,
            ROUND(
                100.0 * COALESCE(r.polices_resiliees, 0) / NULLIF(COALESCE(p.total_polices, 0), 0),
                2
            ) AS taux_resiliation
        FROM portfolio p
        CROSS JOIN resiliation r
    """

    top_branch_sql = """
        SELECT
            branche,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
        GROUP BY branche
        ORDER BY total_pnet DESC
        LIMIT 5
    """

    top_resiliation_branch_sql = """
        WITH rep AS (
            SELECT branche, COUNT(DISTINCT id_police) as polices_resiliees
            FROM dwh_fact_annulation
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_annulation BETWEEN :year_from AND :year_to
            GROUP BY branche
        )
        SELECT branche, polices_resiliees
        FROM rep
        ORDER BY polices_resiliees DESC
        LIMIT 5
    """

    production = _query_dataframe(production_sql, params).iloc[0]
    impaye = _query_dataframe(impaye_sql, params).iloc[0]
    churn = _query_dataframe(churn_sql, params).iloc[0]
    top_branches = _query_dataframe(top_branch_sql, params).to_dict(orient="records")
    top_resiliation_branches = _query_dataframe(top_resiliation_branch_sql, params).to_dict(orient="records")

    total_pnet = _safe_float(production["total_pnet"])
    total_commission = _safe_float(production["total_commission"])
    total_mt_acp = _safe_float(impaye["total_mt_acp"])

    return {
        "source": "postgres",
        "branch": branch or "ALL",
        "year_from": year_from,
        "year_to": year_to,
        "total_pnet": total_pnet,
        "total_commission": total_commission,
        "nb_quittances": _safe_int(production["nb_quittances"]),
        "total_mt_acp": total_mt_acp,
        "nb_impayes": _safe_int(impaye["nb_impayes"]),
        "sp_ratio_proxy_pct": (100.0 * total_mt_acp / total_pnet) if total_pnet > 0 else 0.0,
        "taux_resiliation_pct": _safe_float(churn["taux_resiliation"]),
        "top_branches": top_branches,
        "top_resiliation_branches": top_resiliation_branches,
    }


def _fetch_kpi_context_denodo(context: dict[str, Any]) -> dict[str, Any] | None:
    if not DENODO_KPI_API_URL:
        return None

    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)

    params = {
        "branch": branch,
        "year_from": year_from,
        "year_to": year_to,
    }

    try:
        response = requests.get(DENODO_KPI_API_URL, params=params, timeout=8)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            data = payload if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None

        total_pnet = _safe_float(data.get("total_pnet"))
        total_mt_acp = _safe_float(data.get("total_mt_acp"))
        denodo_payload = {
            "source": "denodo",
            "branch": branch or "ALL",
            "year_from": year_from,
            "year_to": year_to,
            "total_pnet": total_pnet,
            "total_commission": _safe_float(data.get("total_commission")),
            "nb_quittances": _safe_int(data.get("nb_quittances")),
            "total_mt_acp": total_mt_acp,
            "nb_impayes": _safe_int(data.get("nb_impayes")),
            "sp_ratio_proxy_pct": _safe_float(data.get("sp_ratio_proxy_pct"), (100.0 * total_mt_acp / total_pnet) if total_pnet > 0 else 0.0),
            "taux_resiliation_pct": _safe_float(data.get("taux_resiliation_pct")),
            "top_branches": data.get("top_branches", []),
            "top_resiliation_branches": data.get("top_resiliation_branches", []),
        }
        return denodo_payload
    except Exception:
        return None


def kpi_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    payload = _fetch_kpi_context_denodo(context)
    if payload is None:
        payload = _fetch_kpi_context_postgres(context)
        payload["source"] = "postgres_fallback"

    summary = (
        f"Source {payload['source']}: prime nette {payload['total_pnet']:,.0f} TND, "
        f"commission {payload['total_commission']:,.0f} TND, "
        f"resiliation {payload['taux_resiliation_pct']:.2f}%, "
        f"S/P proxy {payload['sp_ratio_proxy_pct']:.2f}%."
    )

    if "resili" in question.lower():
        chart = _build_chart_payload(
            chart_type="bar",
            title="Top branches par volume de resiliation",
            x_key="branche",
            y_key="polices_resiliees",
            items=list(payload.get("top_resiliation_branches", [])),
        )
    else:
        chart = _build_chart_payload(
            chart_type="bar",
            title="Top branches par prime nette",
            x_key="branche",
            y_key="total_pnet",
            items=list(payload.get("top_branches", [])),
        )

    table_rows = [
        {
            "source": payload["source"],
            "branch": payload["branch"],
            "periode": f"{payload['year_from']}-{payload['year_to']}",
            "total_pnet": round(payload["total_pnet"], 3),
            "total_commission": round(payload["total_commission"], 3),
            "sp_ratio_proxy_pct": round(payload["sp_ratio_proxy_pct"], 3),
            "taux_resiliation_pct": round(payload["taux_resiliation_pct"], 3),
        }
    ]

    return {
        "tool": "kpi_tool",
        "summary": summary,
        "payload": payload,
        "charts": [chart],
        "tables": [
            {
                "title": "Synthese KPI",
                "columns": list(table_rows[0].keys()),
                "rows": table_rows,
                "markdown": _to_markdown_table(list(table_rows[0].keys()), table_rows),
            }
        ],
    }


def rag_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    top_k = max(1, min(_safe_int(context.get("top_k"), 4), 12))
    qdrant_hits = _qdrant_semantic_search(question, top_k)
    lexical_hits = _lexical_rag_search(question, top_k)

    merged_documents = qdrant_hits if qdrant_hits else lexical_hits

    # Inject live business context as grounded RAG snippet.
    try:
        kpi_snapshot = _fetch_kpi_context_postgres(context)
        live_snippet = {
            "source": "live_business_context",
            "collection": "kpi_snapshot",
            "title": "Contexte KPI courant",
            "content": (
                f"Prime nette {kpi_snapshot['total_pnet']:,.0f} TND, "
                f"S/P proxy {kpi_snapshot['sp_ratio_proxy_pct']:.2f}%, "
                f"resiliation {kpi_snapshot['taux_resiliation_pct']:.2f}%"
            ),
            "score": 1.0,
        }
        merged_documents = [live_snippet] + merged_documents
    except Exception:
        pass

    summary = (
        f"RAG: {len(merged_documents)} snippets retournes "
        f"(qdrant={len(qdrant_hits)}, fallback_local={len(lexical_hits)})."
    )

    table_rows = [
        {
            "source": doc.get("source"),
            "collection": doc.get("collection"),
            "title": doc.get("title"),
            "score": round(_safe_float(doc.get("score")), 4),
        }
        for doc in merged_documents[:10]
    ]

    return {
        "tool": "rag_tool",
        "summary": summary,
        "payload": {
            "documents": merged_documents[:top_k],
            "qdrant_hits": len(qdrant_hits),
            "local_hits": len(lexical_hits),
            "top_k": top_k,
        },
        "tables": [
            {
                "title": "RAG retrieval",
                "columns": ["source", "collection", "title", "score"],
                "rows": table_rows,
                "markdown": _to_markdown_table(["source", "collection", "title", "score"], table_rows),
            }
        ],
    }


def alerte_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    threshold_impaye_rate = _safe_float(os.getenv("ALERTE_IMPAYE_RATE_PCT"), 2.0)
    threshold_drop_pct = _safe_float(os.getenv("ALERTE_PRODUCTION_DROP_PCT"), 15.0)

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
              AND make_date(annee_echeance, mois_echeance, 1) >= date_trunc('month', current_date) - interval '6 months'
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
              AND make_date(annee_echeance, mois_echeance, 1) >= date_trunc('month', current_date) - interval '6 months'
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

    monthly_df = _query_dataframe(sql_query, {"branch": branch})
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


def _format_metric_value(value: float, unit: str) -> str:
    normalized_unit = unit.strip().upper()
    if normalized_unit == "TND":
        return f"{value:,.0f} TND"
    if normalized_unit == "%":
        return f"{value:.2f}%"
    if normalized_unit == "COUNT":
        return f"{int(round(value)):,.0f}"
    return f"{value:,.2f}"


def _infer_forecast_report_mode(question: str) -> str:
    lowered = _normalize_text(question)

    graph_only_markers = [
        "graphique uniquement",
        "graphe uniquement",
        "uniquement un graphique",
        "uniquement un graphe",
        "juste un graphe",
        "seulement un graphe",
        "seulement un graphique",
        "only graph",
        "graph only",
        "sans table",
        "without table",
    ]
    table_only_markers = [
        "table uniquement",
        "tableau uniquement",
        "juste la table",
        "only table",
        "table only",
        "sans graphique",
        "without graph",
        "without chart",
    ]
    graph_markers = ["graph", "graphe", "graphique", "chart", "plot", "courbe", "diagramme"]
    table_markers = ["table", "tableau", "tabulaire", "lignes", "rows"]

    if _contains_any(lowered, graph_only_markers):
        return "graph_only"
    if _contains_any(lowered, table_only_markers):
        return "table_only"

    graph_requested = _contains_any(lowered, graph_markers)
    table_requested = _contains_any(lowered, table_markers)

    if graph_requested and not table_requested:
        return "graph_pref"
    if table_requested and not graph_requested:
        return "table_pref"
    return "report"


def _detect_forecast_target(normalized_question: str) -> dict[str, Any]:
    count_signal = any(token in normalized_question for token in ["nombre", "nb", "count", "volume", "combien"])
    amount_signal = any(token in normalized_question for token in ["montant", "somme", "total", "valeur", "tnd", "mt_"])

    if any(token in normalized_question for token in ["ratio", "combine", "combin", "s/p", "sp proxy"]):
        return {
            "metric": "sp_ratio_proxy_pct",
            "label": "Ratio S/P Proxy",
            "unit": "%",
            "source_kind": "ratio",
            "value_expr": "",
            "proxy_note": "Proxy analytique du ratio S/P.",
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
        return {
            "metric": "nb_sinistres_proxy",
            "label": "Nombre de sinistres (proxy impayes)",
            "unit": "count",
            "source_kind": "impaye",
            "value_expr": "COUNT(*)",
            "proxy_note": "Aucune table sinistres dediee: proxy calcule via dwh_fact_impaye.",
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

    risk_metrics = {"nb_impayes", "total_impaye", "nb_sinistres_proxy", "nb_annulations", "total_annulation"}
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

    if target["source_kind"] == "impaye":
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
    elif target["source_kind"] == "ratio":
        # S/P Proxy joined calculation
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
            monthly_impaye AS (
                SELECT 
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COALESCE(SUM(mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye
                WHERE (:branch IS NULL OR branche = :branch)
                  AND annee_echeance BETWEEN :year_from AND :year_to
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
            )
            SELECT 
                COALESCE(e.period, i.period) AS period,
                CASE WHEN COALESCE(e.total_pnet, 0) > 0 
                     THEN (COALESCE(i.total_impaye, 0) / e.total_pnet) * 100.0 
                     ELSE 0.0 
                END AS metric_value
            FROM monthly_emission e
            FULL OUTER JOIN monthly_impaye i ON e.period = i.period
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


def anomaly_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)

    sql_query = """
        SELECT
            make_date(annee_echeance, mois_echeance, 1) AS period,
            COALESCE(SUM(mt_acp), 0) AS total_mt_acp,
            COUNT(*) AS nb_impayes
        FROM dwh_fact_impaye
        WHERE (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
          AND mois_echeance BETWEEN 1 AND 12
        GROUP BY make_date(annee_echeance, mois_echeance, 1)
        ORDER BY period
    """

    df = _query_dataframe(sql_query, {"branch": branch, "year_from": year_from, "year_to": year_to})
    if len(df) < 6:
        return {
            "tool": "anomaly_tool",
            "summary": "Donnees insuffisantes pour la detection d anomalies (minimum 6 mois).",
            "payload": {"anomalies": [], "engine": "isolation_forest"},
        }

    numeric = df[["total_mt_acp", "nb_impayes"]].astype(float)
    contamination = min(0.20, max(0.08, 2.0 / len(df)))
    detector = IsolationForest(contamination=contamination, random_state=42)
    labels = detector.fit_predict(numeric)
    scores = detector.decision_function(numeric)

    df = df.assign(anomaly_label=labels, anomaly_score=scores)
    anomalies_df = df[df["anomaly_label"] == -1]
    anomalies = [
        {
            "period": pd.Timestamp(row["period"]).strftime("%Y-%m"),
            "total_mt_acp": _safe_float(row["total_mt_acp"]),
            "nb_impayes": _safe_int(row["nb_impayes"]),
            "anomaly_score": _safe_float(row["anomaly_score"]),
        }
        for _, row in anomalies_df.iterrows()
    ]

    summary = (
        f"Isolation Forest: {len(anomalies)} anomalies detectees."
        if anomalies
        else "Isolation Forest: aucune anomalie significative detectee."
    )

    chart_items = [
        {
            "period": pd.Timestamp(row["period"]).strftime("%Y-%m"),
            "total_mt_acp": _safe_float(row["total_mt_acp"]),
            "is_anomaly": 1 if _safe_int(row["anomaly_label"]) == -1 else 0,
        }
        for _, row in df.iterrows()
    ]

    return {
        "tool": "anomaly_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "engine": "isolation_forest",
            "contamination": contamination,
            "anomalies": anomalies,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Anomalies impaye (IForest)",
                x_key="period",
                y_key="total_mt_acp",
                items=chart_items,
            )
        ],
    }


def drift_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))

    sql_query = """
        SELECT
            make_date(annee_echeance, mois_echeance, 1) AS period,
            AVG(mt_pnet) AS avg_pnet,
            AVG(mt_commission) AS avg_commission,
            AVG(bonus_malus) AS avg_bonus_malus,
            AVG(CASE WHEN mt_pnet > 0 THEN mt_commission / mt_pnet ELSE NULL END) AS avg_commission_rate
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND (:branch IS NULL OR branche = :branch)
                    AND annee_echeance BETWEEN 1900 AND 2100
                    AND mois_echeance BETWEEN 1 AND 12
          AND make_date(annee_echeance, mois_echeance, 1) >= date_trunc('month', current_date) - interval '12 months'
        GROUP BY make_date(annee_echeance, mois_echeance, 1)
        ORDER BY period
    """

    df = _query_dataframe(sql_query, {"branch": branch})
    if len(df) < 8:
        return {
            "tool": "drift_tool",
            "summary": "Donnees insuffisantes pour un drift robuste (minimum 8 points mensuels).",
            "payload": {"metrics": [], "engine": "statistical_fallback"},
        }

    previous = df.iloc[: len(df) // 2].copy()
    current = df.iloc[len(df) // 2 :].copy()
    metrics: list[dict[str, Any]] = []

    highest_status = "low"
    for feature in ["avg_pnet", "avg_commission", "avg_bonus_malus", "avg_commission_rate"]:
        prev_values = previous[feature].astype(float).dropna().to_numpy()
        curr_values = current[feature].astype(float).dropna().to_numpy()

        if len(prev_values) == 0 or len(curr_values) == 0:
            continue

        ks_stat, pvalue = ks_2samp(prev_values, curr_values)
        prev_mean = float(np.mean(prev_values))
        curr_mean = float(np.mean(curr_values))
        delta_pct = (100.0 * (curr_mean - prev_mean) / prev_mean) if prev_mean != 0 else 0.0

        if pvalue < 0.01 and abs(delta_pct) >= 20:
            status = "high"
        elif pvalue < 0.05 and abs(delta_pct) >= 10:
            status = "medium"
        else:

            status = "low"

        if status == "high":
            highest_status = "high"
        elif status == "medium" and highest_status != "high":
            highest_status = "medium"

        metrics.append(
            {
                "feature": feature,
                "previous_mean": prev_mean,
                "current_mean": curr_mean,
                "delta_pct": delta_pct,
                "ks_stat": float(ks_stat),
                "pvalue": float(pvalue),
                "status": status,
            }
        )

    try:
        evidently_module = importlib.import_module("evidently")

        engine = "evidently_available"
        _ = getattr(evidently_module, "__version__", "unknown")
    except Exception:
        engine = "statistical_fallback"

    summary = f"Drift {highest_status} detecte ({engine})."

    return {
        "tool": "drift_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "engine": engine,
            "status": highest_status,
            "metrics": metrics,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Drift delta %",
                x_key="feature",
                y_key="delta_pct",
                items=metrics,
            )
        ],
    }


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


def segmentation_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))

    sql_query = """
        WITH client_production AS (
            SELECT
                c.id_client,
                COALESCE(c.nom, 'N/A') AS nom,
                COALESCE(c.prenom, 'N/A') AS prenom,
                COALESCE(c.ville, 'N/A') AS ville,
                COUNT(DISTINCT p.id_police) AS nb_polices,
                COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
                COALESCE(AVG(p.bonus_malus), 1.0) AS avg_bonus_malus
            FROM dim_client c
            LEFT JOIN dim_police p ON p.id_client = c.id_client
            LEFT JOIN dwh_fact_emission e ON e.id_police = p.id_police AND e.etat_quit IN ('E','P','A')
            WHERE (:branch IS NULL OR p.branche = :branch)
            GROUP BY c.id_client, COALESCE(c.nom, 'N/A'), COALESCE(c.prenom, 'N/A'), COALESCE(c.ville, 'N/A')
        ),
        client_impaye AS (
            SELECT
                p.id_client,
                COUNT(*) AS nb_impayes,
                COALESCE(SUM(i.mt_acp), 0) AS total_impaye
            FROM dwh_fact_impaye i
            JOIN dim_police p ON p.id_police = i.id_police
            WHERE (:branch IS NULL OR i.branche = :branch)
            GROUP BY p.id_client
        )
        SELECT
            cp.id_client,
            cp.nom,
            cp.prenom,
            cp.ville,
            cp.nb_polices,
            cp.total_pnet,
            cp.avg_bonus_malus,
            COALESCE(ci.nb_impayes, 0) AS nb_impayes,
            COALESCE(ci.total_impaye, 0) AS total_impaye
        FROM client_production cp
        LEFT JOIN client_impaye ci ON ci.id_client = cp.id_client
        WHERE cp.nb_polices > 0
    """

    df = _query_dataframe(sql_query, {"branch": branch})
    if len(df) < 10:
        return {
            "tool": "segmentation_tool",
            "summary": "Donnees insuffisantes pour la segmentation clients.",
            "payload": {"segments": [], "engine": "kmeans"},
        }

    df = df.copy()
    df["impaye_ratio"] = np.where(df["nb_polices"] > 0, df["nb_impayes"] / df["nb_polices"], 0.0)

    model_matrix = np.column_stack(
        [
            np.log1p(df["total_pnet"].astype(float).to_numpy()),
            df["impaye_ratio"].astype(float).to_numpy(),
            df["avg_bonus_malus"].astype(float).to_numpy(),
        ]
    )

    n_clusters = min(4, max(3, len(df) // 80))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(model_matrix)
    df["cluster"] = labels

    segment_profiles = (
        df.groupby("cluster", as_index=False)
        .agg(
            nb_clients=("id_client", "count"),
            avg_total_pnet=("total_pnet", "mean"),
            avg_impaye_ratio=("impaye_ratio", "mean"),
            avg_bonus_malus=("avg_bonus_malus", "mean"),
        )
        .sort_values("avg_total_pnet", ascending=False)
        .to_dict(orient="records")
    )

    summary = "Segmentation clients KMeans realisee avec profils de risque et valeur."

    return {
        "tool": "segmentation_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "engine": "kmeans",
            "segments": segment_profiles,
            "sample_clients": (
                df.sort_values(["cluster", "total_pnet"], ascending=[True, False])
                .head(20)[["id_client", "nom", "prenom", "ville", "cluster", "total_pnet", "impaye_ratio"]]
                .to_dict(orient="records")
            ),
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Segments clients",
                x_key="cluster",
                y_key="nb_clients",
                items=segment_profiles,
            )
        ],
    }


def client_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    target_name = _extract_client_name(question, context)

    top_clients_sql = """
        SELECT
            c.id_client,
            COALESCE(c.nom, 'N/A') AS nom,
            COALESCE(c.prenom, 'N/A') AS prenom,
            COALESCE(c.ville, 'N/A') AS ville,
            COUNT(*) AS nb_impayes,
            COALESCE(SUM(i.mt_acp), 0) AS total_impaye,
            COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
        FROM dwh_fact_impaye i
        JOIN dim_police p ON p.id_police = i.id_police
        JOIN dim_client c ON c.id_client = p.id_client
        LEFT JOIN dwh_fact_emission e ON e.num_quittance = i.num_quittance
        WHERE (:branch IS NULL OR i.branche = :branch)
        GROUP BY c.id_client, COALESCE(c.nom, 'N/A'), COALESCE(c.prenom, 'N/A'), COALESCE(c.ville, 'N/A')
        ORDER BY total_impaye DESC
        LIMIT 15
    """

    top_clients = _query_dataframe(top_clients_sql, {"branch": branch}).to_dict(orient="records")

    profile_rows: list[dict[str, Any]] = []
    homonym_rows: list[dict[str, Any]] = []

    if target_name:
        profile_sql = """
            SELECT
                c.id_client,
                COALESCE(c.nom, 'N/A') AS nom,
                COALESCE(c.prenom, 'N/A') AS prenom,
                COALESCE(c.ville, 'N/A') AS ville,
                COUNT(DISTINCT p.id_police) AS nb_polices,
                COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
                COALESCE(SUM(i.mt_acp), 0) AS total_impaye,
                COUNT(i.id) AS nb_impayes
            FROM dim_client c
            LEFT JOIN dim_police p ON p.id_client = c.id_client
            LEFT JOIN dwh_fact_emission e ON e.id_police = p.id_police AND e.etat_quit IN ('E','P','A')
            LEFT JOIN dwh_fact_impaye i ON i.id_police = p.id_police
            WHERE CONCAT(COALESCE(c.nom, ''), ' ', COALESCE(c.prenom, '')) ILIKE :pattern
            GROUP BY c.id_client, COALESCE(c.nom, 'N/A'), COALESCE(c.prenom, 'N/A'), COALESCE(c.ville, 'N/A')
            ORDER BY total_impaye DESC
            LIMIT 20
        """
        profile_rows = _query_dataframe(profile_sql, {"pattern": f"%{target_name}%"}).to_dict(orient="records")

        homonym_sql = """
            SELECT
                COALESCE(nom, 'N/A') AS nom,
                COALESCE(prenom, 'N/A') AS prenom,
                COUNT(*) AS homonym_count
            FROM dim_client
            WHERE CONCAT(COALESCE(nom, ''), ' ', COALESCE(prenom, '')) ILIKE :pattern
            GROUP BY COALESCE(nom, 'N/A'), COALESCE(prenom, 'N/A')
            HAVING COUNT(*) > 1
            ORDER BY homonym_count DESC
            LIMIT 10
        """
        homonym_rows = _query_dataframe(homonym_sql, {"pattern": f"%{target_name}%"}).to_dict(orient="records")

    summary = (
        f"Client analytics: {len(top_clients)} clients top impaye retournes."
        if not target_name
        else f"Client analytics sur '{target_name}': {len(profile_rows)} profils et {len(homonym_rows)} homonymes."
    )

    return {
        "tool": "client_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "target_name": target_name,
            "top_claim_clients": top_clients,
            "named_client_profile": profile_rows,
            "homonym_candidates": homonym_rows,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Top clients par impaye",
                x_key="id_client",
                y_key="total_impaye",
                items=top_clients,
            )
        ],
    }


def _contains_any(text_value: str, keywords: list[str]) -> bool:
    return any(keyword in text_value for keyword in keywords)


def _contains_any_term(text_value: str, terms: list[str]) -> bool:
    for term in terms:
        normalized_term = term.strip().lower()
        if not normalized_term:
            continue

        if " " in normalized_term:
            if normalized_term in text_value:
                return True
            continue

        if re.search(rf"\b{re.escape(normalized_term)}\b", text_value):
            return True

    return False


def _extract_top_n(normalized_question: str, default_value: int = 10) -> int:
    patterns = [r"\btop\s*(\d{1,2})\b", r"\bpremiers?\s*(\d{1,2})\b", r"\bfirst\s*(\d{1,2})\b"]
    for pattern in patterns:
        match = re.search(pattern, normalized_question)
        if match:
            return min(50, max(1, _safe_int(match.group(1), default_value)))
    return default_value


def _detect_sql_metric(normalized_question: str) -> str:
    if _contains_any(normalized_question, ["impaye", "impayes", "sinistre", "sinistres"]):
        return "impaye"
    if _contains_any(normalized_question, ["resiliation", "annulation", "resilie", "resiliee"]):
        return "resiliation"

    # Business default: "zones/branches a risque" maps to impaye risk proxy.
    has_risk_term = _contains_any_term(normalized_question, ["risque", "risques", "a risque", "exposition", "expose"])
    has_spatial_dimension = _contains_any(
        normalized_question,
        ["zone", "zones", "gouvernorat", "gouvernorats", "region", "ville", "branche", "branches"],
    )
    has_ranking_term = _contains_any_term(normalized_question, ["top", "classement", "ranking"])
    if has_risk_term and (has_spatial_dimension or has_ranking_term):
        return "impaye"

    if _contains_any(normalized_question, ["prime", "pnet", "production", "commission"]):
        return "prime"
    if "client" in normalized_question:
        return "client"
    return "prime"


def _detect_sql_aggregation(normalized_question: str, metric: str) -> str:
    if _contains_any_term(normalized_question, ["ratio", "taux"]) and metric in {"impaye", "resiliation"}:
        return "ratio"
    if _contains_any_term(normalized_question, ["moyenne", "average", "avg", "moyen"]):
        return "avg"
    if _contains_any_term(normalized_question, ["distinct", "unique", "uniques", "differents"]):
        return "count_distinct"
    if _contains_any_term(normalized_question, ["nombre", "nb", "count", "combien"]):
        return "count"
    if _contains_any_term(normalized_question, ["somme", "sum", "montant", "total", "cumule", "global"]):
        return "sum"
    if metric == "client":
        return "count"
    if metric == "resiliation":
        return "count"
    return "sum"


def _detect_sql_dimensions(normalized_question: str) -> list[str]:
    dimensions: list[str] = []

    if _contains_any(normalized_question, ["branche", "branches"]):
        dimensions.append("branche")

    if _contains_any(normalized_question, ["gouvernorat", "gouvernorats", "localite", "region", "zone", "ville"]):
        dimensions.append("gouvernorat")

    has_client_dimension = _contains_any(
        normalized_question,
        ["par client", "top client", "top clients", "classement client", "classement des clients"],
    ) or re.search(r"\btop\s*(\d{1,2}\s*)?clients?\b", normalized_question) is not None

    if has_client_dimension:
        dimensions.append("client")

    return dimensions


def _analyze_sql_request(normalized_question: str) -> dict[str, Any]:
    metric = _detect_sql_metric(normalized_question)
    aggregation = _detect_sql_aggregation(normalized_question, metric)
    dimensions = _detect_sql_dimensions(normalized_question)
    is_timeseries = _contains_any(
        normalized_question,
        ["evolution", "trend", "tendance", "mensuel", "mensuelle", "mois", "historique"],
    )
    is_ranking = _contains_any(
        normalized_question,
        ["top", "classement", "ranking", "premier", "premiers", "plus eleve", "plus grand"],
    )
    top_n = _extract_top_n(normalized_question, default_value=10)

    return {
        "metric": metric,
        "aggregation": aggregation,
        "dimensions": dimensions,
        "is_timeseries": is_timeseries,
        "is_ranking": is_ranking,
        "top_n": top_n,
    }


def _metric_descriptor(metric: str, aggregation: str, column_prefix: str = "") -> dict[str, str]:
    prefix = f"{column_prefix}." if column_prefix else ""

    if metric == "impaye":
        if aggregation == "count":
            return {
                "expression": "COUNT(*)",
                "alias": "nb_impayes",
                "label": "Nombre d impayes",
                "unit": "count",
            }
        if aggregation == "avg":
            return {
                "expression": f"COALESCE(AVG({prefix}mt_acp), 0)",
                "alias": "avg_impaye",
                "label": "Moyenne impaye",
                "unit": "TND",
            }
        if aggregation == "count_distinct":
            return {
                "expression": f"COUNT(DISTINCT {prefix}id_police)",
                "alias": "nb_polices_impactees",
                "label": "Polices impactees",
                "unit": "count",
            }
        return {
            "expression": f"COALESCE(SUM({prefix}mt_acp), 0)",
            "alias": "total_impaye",
            "label": "Montant impaye total",
            "unit": "TND",
        }

    if metric == "prime":
        if aggregation == "count":
            return {
                "expression": "COUNT(*)",
                "alias": "nb_quittances",
                "label": "Nombre de quittances",
                "unit": "count",
            }
        if aggregation == "avg":
            return {
                "expression": f"COALESCE(AVG({prefix}mt_pnet), 0)",
                "alias": "avg_pnet",
                "label": "Prime nette moyenne",
                "unit": "TND",
            }
        if aggregation == "count_distinct":
            return {
                "expression": f"COUNT(DISTINCT {prefix}id_police)",
                "alias": "nb_polices",
                "label": "Polices distinctes",
                "unit": "count",
            }
        return {
            "expression": f"COALESCE(SUM({prefix}mt_pnet), 0)",
            "alias": "total_pnet",
            "label": "Prime nette totale",
            "unit": "TND",
        }

    return {
        "expression": "COUNT(*)",
        "alias": "metric_value",
        "label": "Valeur",
        "unit": "count",
    }


def _build_semantic_sql_query_spec(semantic: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | None:
    metric = str(semantic.get("metric", "prime"))
    aggregation = str(semantic.get("aggregation", "sum"))
    dimensions = semantic.get("dimensions") if isinstance(semantic.get("dimensions"), list) else []
    is_timeseries = bool(semantic.get("is_timeseries", False))
    is_ranking = bool(semantic.get("is_ranking", False))
    top_n = _safe_int(semantic.get("top_n"), 10)
    limit_value = top_n if is_ranking else 20

    if metric == "impaye":
        if aggregation == "ratio" and "branche" in dimensions:
            return {
                "sql_id": "branch_impaye_ratio",
                "sql_query": """
                    WITH emission AS (
                        SELECT
                            branche,
                            COALESCE(SUM(mt_pnet), 0) AS total_pnet
                        FROM dwh_fact_emission
                        WHERE etat_quit IN ('E','P','A')
                          AND (:branch IS NULL OR branche = :branch)
                          AND annee_echeance BETWEEN :year_from AND :year_to
                        GROUP BY branche
                    ),
                    impaye AS (
                        SELECT
                            branche,
                            COALESCE(SUM(mt_acp), 0) AS total_impaye
                        FROM dwh_fact_impaye
                        WHERE (:branch IS NULL OR branche = :branch)
                          AND annee_echeance BETWEEN :year_from AND :year_to
                        GROUP BY branche
                    )
                    SELECT
                        e.branche,
                        e.total_pnet,
                        COALESCE(i.total_impaye, 0) AS total_impaye,
                        ROUND(100.0 * COALESCE(i.total_impaye, 0) / NULLIF(e.total_pnet, 0), 3) AS impaye_ratio_pct
                    FROM emission e
                    LEFT JOIN impaye i ON i.branche = e.branche
                    ORDER BY impaye_ratio_pct DESC NULLS LAST, total_impaye DESC
                    LIMIT 10
                """,
                "params": params,
                "chart": {
                    "type": "bar",
                    "title": "Ratio impaye par branche",
                    "x_key": "branche",
                    "y_key": "impaye_ratio_pct",
                },
                "result_kind": "breakdown",
            }

        if is_timeseries:
            descriptor = _metric_descriptor("impaye", "sum" if aggregation == "ratio" else aggregation)
            alias = descriptor["alias"]
            return {
                "sql_id": f"monthly_impaye_{alias}",
                "sql_query": f"""
                    SELECT
                        make_date(annee_echeance, mois_echeance, 1) AS period,
                        {descriptor['expression']} AS {alias}
                    FROM dwh_fact_impaye
                    WHERE (:branch IS NULL OR branche = :branch)
                      AND annee_echeance BETWEEN :year_from AND :year_to
                      AND annee_echeance BETWEEN 1900 AND 2100
                      AND mois_echeance BETWEEN 1 AND 12
                    GROUP BY make_date(annee_echeance, mois_echeance, 1)
                    ORDER BY period
                """,
                "params": params,
                "chart": {
                    "type": "line",
                    "title": f"Evolution mensuelle {descriptor['label'].lower()}",
                    "x_key": "period",
                    "y_key": alias,
                },
                "result_kind": "timeseries",
            }

        if "gouvernorat" in dimensions:
            descriptor = _metric_descriptor("impaye", aggregation, column_prefix="i")
            alias = descriptor["alias"]
            return {
                "sql_id": f"impaye_by_gouvernorat_{alias}",
                "sql_query": f"""
                    SELECT
                        COALESCE(a.localite_agent, 'N/A') AS gouvernorat,
                        {descriptor['expression']} AS {alias}
                    FROM dwh_fact_impaye i
                    LEFT JOIN dim_agent a ON a.id_agent = i.id_agent
                    WHERE (:branch IS NULL OR i.branche = :branch)
                      AND i.annee_echeance BETWEEN :year_from AND :year_to
                    GROUP BY COALESCE(a.localite_agent, 'N/A')
                    ORDER BY {alias} DESC
                    LIMIT {limit_value}
                """,
                "params": params,
                "chart": {
                    "type": "bar",
                    "title": f"{descriptor['label']} par gouvernorat",
                    "x_key": "gouvernorat",
                    "y_key": alias,
                },
                "result_kind": "breakdown",
            }

        if "client" in dimensions:
            descriptor = _metric_descriptor("impaye", aggregation, column_prefix="i")
            alias = descriptor["alias"]
            sql_id_prefix = "top_clients_impaye" if is_ranking else "impaye_by_client"
            return {
                "sql_id": f"{sql_id_prefix}_{alias}",
                "sql_query": f"""
                    SELECT
                        c.id_client,
                        COALESCE(c.nom, 'N/A') AS nom,
                        COALESCE(c.prenom, 'N/A') AS prenom,
                        {descriptor['expression']} AS {alias}
                    FROM dwh_fact_impaye i
                    JOIN dim_police p ON p.id_police = i.id_police
                    JOIN dim_client c ON c.id_client = p.id_client
                    WHERE (:branch IS NULL OR i.branche = :branch)
                      AND i.annee_echeance BETWEEN :year_from AND :year_to
                    GROUP BY c.id_client, COALESCE(c.nom, 'N/A'), COALESCE(c.prenom, 'N/A')
                    ORDER BY {alias} DESC
                    LIMIT {limit_value}
                """,
                "params": params,
                "chart": {
                    "type": "bar",
                    "title": f"{descriptor['label']} par client",
                    "x_key": "id_client",
                    "y_key": alias,
                },
                "result_kind": "breakdown",
            }

        if "branche" in dimensions:
            descriptor = _metric_descriptor("impaye", aggregation)
            alias = descriptor["alias"]
            return {
                "sql_id": f"impaye_by_branche_{alias}",
                "sql_query": f"""
                    SELECT
                        branche,
                        {descriptor['expression']} AS {alias}
                    FROM dwh_fact_impaye
                    WHERE (:branch IS NULL OR branche = :branch)
                      AND annee_echeance BETWEEN :year_from AND :year_to
                    GROUP BY branche
                    ORDER BY {alias} DESC
                    LIMIT {limit_value}
                """,
                "params": params,
                "chart": {
                    "type": "bar",
                    "title": f"{descriptor['label']} par branche",
                    "x_key": "branche",
                    "y_key": alias,
                },
                "result_kind": "breakdown",
            }

        return {
            "sql_id": "total_impayes_overview",
            "sql_query": """
                SELECT
                    COUNT(*) AS nb_impayes,
                    COALESCE(SUM(mt_acp), 0) AS total_impaye,
                    COALESCE(AVG(mt_acp), 0) AS avg_impaye,
                    COUNT(DISTINCT id_police) AS nb_polices_impactees
                FROM dwh_fact_impaye
                WHERE (:branch IS NULL OR branche = :branch)
                  AND annee_echeance BETWEEN :year_from AND :year_to
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_impayes", "label": "Nombre d impayes", "unit": "count"},
                {"key": "total_impaye", "label": "Montant cumule impaye", "unit": "TND"},
                {"key": "avg_impaye", "label": "Moyenne impaye", "unit": "TND"},
                {"key": "nb_polices_impactees", "label": "Polices impactees", "unit": "count"},
            ],
        }

    if metric == "resiliation":
        if "branche" in dimensions:
            return {
                "sql_id": "branch_resiliation_rate",
                "sql_query": """
                    WITH emission AS (
                        SELECT
                            branche,
                            COUNT(DISTINCT id_police) AS nb_polices
                        FROM dwh_fact_emission
                        WHERE etat_quit IN ('E','P','A')
                          AND (:branch IS NULL OR branche = :branch)
                          AND annee_echeance BETWEEN :year_from AND :year_to
                        GROUP BY branche
                    ),
                    annulation AS (
                        SELECT
                            branche,
                            COUNT(DISTINCT id_police) AS nb_resiliees
                        FROM dwh_fact_annulation
                        WHERE (:branch IS NULL OR branche = :branch)
                          AND annee_annulation BETWEEN :year_from AND :year_to
                        GROUP BY branche
                    )
                    SELECT
                        e.branche,
                        COALESCE(e.nb_polices, 0) AS nb_polices,
                        COALESCE(a.nb_resiliees, 0) AS nb_resiliees,
                        ROUND(100.0 * COALESCE(a.nb_resiliees, 0) / NULLIF(COALESCE(e.nb_polices, 0), 0), 3) AS taux_resiliation_pct
                    FROM emission e
                    LEFT JOIN annulation a ON a.branche = e.branche
                    ORDER BY taux_resiliation_pct DESC NULLS LAST, nb_resiliees DESC
                    LIMIT 10
                """,
                "params": params,
                "chart": {
                    "type": "bar",
                    "title": "Taux de resiliation par branche",
                    "x_key": "branche",
                    "y_key": "taux_resiliation_pct",
                },
                "result_kind": "breakdown",
            }

        return {
            "sql_id": "overall_resiliation_rate",
            "sql_query": """
                WITH portfolio AS (
                    SELECT
                        COUNT(DISTINCT id_police) AS total_polices
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:branch IS NULL OR branche = :branch)
                      AND annee_echeance BETWEEN :year_from AND :year_to
                ),
                annulation AS (
                    SELECT
                        COUNT(DISTINCT id_police) AS nb_resiliees
                    FROM dwh_fact_annulation
                    WHERE (:branch IS NULL OR branche = :branch)
                      AND annee_annulation BETWEEN :year_from AND :year_to
                )
                SELECT
                    COALESCE(p.total_polices, 0) AS total_polices,
                    COALESCE(a.nb_resiliees, 0) AS nb_resiliees,
                    ROUND(100.0 * COALESCE(a.nb_resiliees, 0) / NULLIF(COALESCE(p.total_polices, 0), 0), 3) AS taux_resiliation_pct
                FROM portfolio p
                CROSS JOIN annulation a
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "taux_resiliation_pct", "label": "Taux de resiliation", "unit": "%"},
                {"key": "nb_resiliees", "label": "Polices resiliees", "unit": "count"},
                {"key": "total_polices", "label": "Polices totales", "unit": "count"},
            ],
        }

    if metric == "client":
        return {
            "sql_id": "total_clients",
            "sql_query": "SELECT COUNT(*) AS total_clients FROM dim_client",
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_clients", "label": "Total clients", "unit": "count"},
            ],
        }

    if metric == "prime":
        descriptor = _metric_descriptor("prime", aggregation)
        alias = descriptor["alias"]

        if is_timeseries:
            return {
                "sql_id": f"monthly_prime_{alias}",
                "sql_query": f"""
                    SELECT
                        make_date(annee_echeance, mois_echeance, 1) AS period,
                        {descriptor['expression']} AS {alias}
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:branch IS NULL OR branche = :branch)
                      AND annee_echeance BETWEEN :year_from AND :year_to
                      AND annee_echeance BETWEEN 1900 AND 2100
                      AND mois_echeance BETWEEN 1 AND 12
                    GROUP BY make_date(annee_echeance, mois_echeance, 1)
                    ORDER BY period
                """,
                "params": params,
                "chart": {
                    "type": "line",
                    "title": f"Evolution mensuelle {descriptor['label'].lower()}",
                    "x_key": "period",
                    "y_key": alias,
                },
                "result_kind": "timeseries",
            }

        if "branche" in dimensions or is_ranking:
            top_limit = top_n if is_ranking else 10
            return {
                "sql_id": f"prime_by_branche_{alias}",
                "sql_query": f"""
                    SELECT
                        branche,
                        {descriptor['expression']} AS {alias}
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:branch IS NULL OR branche = :branch)
                      AND annee_echeance BETWEEN :year_from AND :year_to
                    GROUP BY branche
                    ORDER BY {alias} DESC
                    LIMIT {top_limit}
                """,
                "params": params,
                "chart": {
                    "type": "bar",
                    "title": f"{descriptor['label']} par branche",
                    "x_key": "branche",
                    "y_key": alias,
                },
                "result_kind": "breakdown",
            }

        return {
            "sql_id": "prime_overview",
            "sql_query": """
                SELECT
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COALESCE(AVG(mt_pnet), 0) AS avg_pnet,
                    COUNT(*) AS nb_quittances,
                    COUNT(DISTINCT id_police) AS nb_polices
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND annee_echeance BETWEEN :year_from AND :year_to
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_pnet", "label": "Prime nette totale", "unit": "TND"},
                {"key": "avg_pnet", "label": "Prime nette moyenne", "unit": "TND"},
                {"key": "nb_quittances", "label": "Nombre de quittances", "unit": "count"},
                {"key": "nb_polices", "label": "Polices distinctes", "unit": "count"},
            ],
        }

    return None


def _infer_sql_report_mode(question: str) -> str:
    lowered = _normalize_text(question)

    graph_only_markers = [
        "graphique uniquement",
        "graphe uniquement",
        "uniquement un graphique",
        "uniquement un graphe",
        "juste un graphe",
        "seulement un graphe",
        "seulement un graphique",
        "only graph",
        "graph only",
        "sans table",
        "without table",
        "data viz only",
    ]
    table_only_markers = [
        "table uniquement",
        "tableau uniquement",
        "juste la table",
        "only table",
        "table only",
        "sans graphique",
        "without graph",
        "without chart",
    ]
    graph_markers = ["graph", "graphe", "graphique", "chart", "plot", "visual", "courbe", "diagramme"]
    table_markers = ["table", "tableau", "tabulaire", "lignes", "rows"]

    if _contains_any(lowered, graph_only_markers):
        return "graph_only"
    if _contains_any(lowered, table_only_markers):
        return "table_only"

    graph_requested = _contains_any(lowered, graph_markers)
    table_requested = _contains_any(lowered, table_markers)

    if graph_requested and not table_requested:
        return "graph_pref"
    if table_requested and not graph_requested:
        return "table_pref"
    return "report"


def _build_sql_query_spec(question: str, context: dict[str, Any]) -> dict[str, Any]:
    lowered = _normalize_text(question)
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    params = {
        "branch": branch,
        "year_from": year_from,
        "year_to": year_to,
    }

    semantic = _analyze_sql_request(lowered)
    semantic_spec = _build_semantic_sql_query_spec(semantic, params)
    if semantic_spec:
        semantic_spec["semantic"] = semantic
        return semantic_spec

    return {
        "sql_id": "top_branches_prime",
        "sql_query": """
            SELECT
                branche,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN :year_from AND :year_to
            GROUP BY branche
            ORDER BY total_pnet DESC
            LIMIT 10
        """,
        "params": params,
        "chart": {
            "type": "bar",
            "title": "Top branches par prime nette",
            "x_key": "branche",
            "y_key": "total_pnet",
        },
        "result_kind": "breakdown",
    }


def _prepare_sql_rows(df: pd.DataFrame, sql_id: str) -> list[dict[str, Any]]:
    working_df = df.copy()

    if "period" in working_df.columns:
        working_df["period"] = pd.to_datetime(working_df["period"], errors="coerce").dt.strftime("%Y-%m")

    rows = working_df.to_dict(orient="records")

    if sql_id == "total_clients" and rows:
        total_clients = _safe_int(rows[0].get("total_clients"), 0)
        rows = [{"label": "clients", "total_clients": total_clients}]

    if sql_id == "total_impayes_overview" and rows:
        nb_impayes = _safe_int(rows[0].get("nb_impayes"), 0)
        total_impaye = _safe_float(rows[0].get("total_impaye"), 0.0)
        avg_impaye = _safe_float(rows[0].get("avg_impaye"), 0.0)
        nb_polices_impactees = _safe_int(rows[0].get("nb_polices_impactees"), 0)
        rows = [
            {
                "label": "impayes",
                "nb_impayes": nb_impayes,
                "total_impaye": total_impaye,
                "avg_impaye": avg_impaye,
                "nb_polices_impactees": nb_polices_impactees,
            }
        ]

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                normalized_row[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                normalized_row[key] = value.strftime("%Y-%m-%d")
            else:

                normalized_row[key] = value
        normalized_rows.append(normalized_row)

    return normalized_rows


def _build_sql_kpis(rows: list[dict[str, Any]], query_spec: dict[str, Any]) -> list[dict[str, Any]]:
    result_kind = str(query_spec.get("result_kind", "tabular")).lower()
    if result_kind != "scalar" or not rows:
        return []

    first_row = rows[0]
    kpi_fields = query_spec.get("kpi_fields")
    kpis: list[dict[str, Any]] = []

    if isinstance(kpi_fields, list):
        for field in kpi_fields:
            if not isinstance(field, dict):
                continue
            key = str(field.get("key", "")).strip()
            if not key or key not in first_row:
                continue

            value = first_row.get(key)
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (int, float)):
                continue

            kpis.append(
                {
                    "key": key,
                    "label": str(field.get("label", key.replace("_", " ").title())),
                    "value": value,
                    "unit": str(field.get("unit", "")),
                }
            )

    if kpis:
        return kpis[:4]

    for key, value in first_row.items():
        if key == "label":
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            kpis.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "value": value,
                    "unit": "",
                }
            )

    return kpis[:4]


def _first_numeric_item(row: dict[str, Any], excluded_keys: set[str] | None = None) -> tuple[str | None, float]:
    excluded = excluded_keys or set()
    for key, value in row.items():
        if key in excluded:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            return key, _safe_float(value)
    return None, 0.0


def _first_dimension_value(row: dict[str, Any]) -> str:
    preferred_keys = ["branche", "gouvernorat", "id_client", "nom", "prenom", "label", "period"]
    for key in preferred_keys:
        if key in row and row.get(key) is not None:
            raw = str(row.get(key)).strip()
            if raw:
                return raw

    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "N/A"


def _format_scope_label(branch: str | None) -> str:
    return "toutes les branches" if not branch else f"la branche {branch}"


def _format_period_label(year_from: int, year_to: int) -> str:
    if year_from == year_to:
        return f"{year_from}"
    return f"{year_from}-{year_to}"


def _build_sql_context_line(sql_id: str, branch: str | None, year_from: int, year_to: int) -> str:
    scope_label = _format_scope_label(branch)
    period_label = _format_period_label(year_from, year_to)

    if sql_id == "overall_resiliation_rate":
        return f"Vue globale de la resiliation sur {scope_label} (periode {period_label})."
    if sql_id == "branch_resiliation_rate":
        return f"Comparaison des taux de resiliation par branche (periode {period_label})."
    if sql_id == "total_impayes_overview":
        return f"Vue globale des impayes sur {scope_label} (periode {period_label})."
    if sql_id == "prime_overview":
        return f"Vue globale de la production sur {scope_label} (periode {period_label})."
    if sql_id.startswith("monthly_prime_") or sql_id.startswith("monthly_impaye_"):
        return f"Evolution mensuelle sur {scope_label} (periode {period_label})."
    if sql_id.startswith("impaye_by_branche_"):
        return f"Comparaison des impayes par branche (periode {period_label})."
    if sql_id.startswith("impaye_by_gouvernorat_"):
        return f"Comparaison des impayes par gouvernorat (periode {period_label})."
    if sql_id.startswith("top_clients_impaye_") or sql_id.startswith("impaye_by_client_"):
        return f"Classement des clients par impaye (periode {period_label})."
    if sql_id.startswith("prime_by_branche_"):
        return f"Comparaison de la production par branche (periode {period_label})."

    return f"Analyse metier sur {scope_label} (periode {period_label})."


def _build_sql_report_details(
    *,
    sql_id: str,
    rows: list[dict[str, Any]],
    branch: str | None,
    year_from: int,
    year_to: int,
) -> dict[str, Any]:
    context_line = _build_sql_context_line(sql_id, branch, year_from, year_to)

    if not rows:
        return {
            "context": context_line,
            "analysis": "Aucune ligne retournee par la requete sur ce perimetre.",
            "decision": "Impossible de conclure sans donnees; elargir le scope temporel ou metier.",
            "actions": [
                "Elargir la periode d analyse (annees supplementaires).",
                "Verifier les filtres de branche et la qualite du chargement des tables sources.",
            ],
        }

    if sql_id == "top_branches_prime":
        total_prime = sum(_safe_float(item.get("total_pnet"), 0.0) for item in rows)
        leader = rows[0]
        leader_branch = str(leader.get("branche", "N/A"))
        leader_prime = _safe_float(leader.get("total_pnet"), 0.0)
        leader_share = (100.0 * leader_prime / total_prime) if total_prime > 0 else 0.0
        concentration = "elevee" if leader_share >= 65.0 else "moderee"
        return {
            "context": context_line,
            "analysis": (
                f"Branche leader {leader_branch} avec {leader_prime:,.0f} TND, soit {leader_share:.2f}% du total analyse."
            ),
            "decision": f"Concentration portefeuille {concentration}; pilotage commercial a ajuster par branche.",
            "actions": [
                "Renforcer la production sur les branches sous-ponderees si l objectif est la diversification.",
                "Fixer un seuil de concentration cible par branche et le suivre mensuellement.",
            ],
        }

    if sql_id == "monthly_prime_trend" or sql_id.startswith("monthly_prime_") or sql_id.startswith("monthly_impaye_"):
        first_item = rows[0]
        last_item = rows[-1]
        metric_key = "total_pnet" if "total_pnet" in first_item else None
        if metric_key is None:
            metric_key, _ = _first_numeric_item(first_item, excluded_keys={"annee_echeance", "mois_echeance"})

        first_value = _safe_float(first_item.get(metric_key or "", 0.0), 0.0)
        last_value = _safe_float(last_item.get(metric_key or "", 0.0), 0.0)
        growth_pct = (100.0 * (last_value - first_value) / first_value) if first_value > 0 else 0.0
        trend = "haussiere" if growth_pct >= 3.0 else "baissiere" if growth_pct <= -3.0 else "stable"
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": (
                f"Evolution {metric_label} de {first_item.get('period')} a {last_item.get('period')}: {growth_pct:.2f}% ({trend})."
            ),
            "decision": f"Tendance {trend}; ajuster capacite commerciale et objectifs de production.",
            "actions": [
                "Comparer la trajectoire mensuelle aux objectifs budgetaires et replanifier si ecart > 5%.",
                "Identifier les mois de rupture et lancer une analyse causale par branche.",
            ],
        }

    if sql_id == "branch_impaye_ratio":
        worst = rows[0]
        worst_branch = str(worst.get("branche", "N/A"))
        worst_ratio = _safe_float(worst.get("impaye_ratio_pct"), 0.0)
        return {
            "context": context_line,
            "analysis": f"Branche la plus exposee aux impayes: {worst_branch} avec ratio {worst_ratio:.2f}%.",
            "decision": "Prioriser le recouvrement sur les branches a ratio impaye eleve.",
            "actions": [
                "Definir des plans de recouvrement differencies par branche selon le ratio impaye.",
                "Reviser les regles de souscription sur la branche la plus risquee.",
            ],
        }

    if sql_id == "total_impayes_overview":
        snapshot = rows[0]
        nb_impayes = _safe_int(snapshot.get("nb_impayes"), 0)
        total_impaye = _safe_float(snapshot.get("total_impaye"), 0.0)
        avg_impaye = _safe_float(snapshot.get("avg_impaye"), 0.0)
        nb_polices_impactees = _safe_int(snapshot.get("nb_polices_impactees"), 0)
        return {
            "context": context_line,
            "analysis": (
                f"Stock global des impayes: {nb_impayes:,} impayes, montant cumule {total_impaye:,.0f} TND, "
                f"moyenne {avg_impaye:,.0f} TND, "
                f"touchant {nb_polices_impactees:,} polices."
            ),
            "decision": "Le volume impaye global justifie un pilotage recouvrement par priorite de montant et anciennete.",
            "actions": [
                "Segmenter le stock impaye par anciennete et lancer des vagues de recouvrement ciblees.",
                "Suivre mensuellement la reduction du nombre d impayes et du montant cumule.",
            ],
        }

    if sql_id == "prime_overview":
        snapshot = rows[0]
        total_pnet = _safe_float(snapshot.get("total_pnet"), 0.0)
        avg_pnet = _safe_float(snapshot.get("avg_pnet"), 0.0)
        nb_quittances = _safe_int(snapshot.get("nb_quittances"), 0)
        return {
            "context": context_line,
            "analysis": (
                f"Production globale: total {total_pnet:,.0f} TND, moyenne {avg_pnet:,.0f} TND sur "
                f"{nb_quittances:,} quittances."
            ),
            "decision": "Piloter la production avec un suivi conjoint du volume et de la valeur moyenne par quittance.",
            "actions": [
                "Comparer la valeur moyenne aux objectifs budgetaires par branche.",
                "Suivre mensuellement le couple volume-valeur pour prevenir la derive commerciale.",
            ],
        }

    if sql_id == "overall_resiliation_rate":
        snapshot = rows[0]
        total_polices = _safe_int(snapshot.get("total_polices"), 0)
        nb_resiliees = _safe_int(snapshot.get("nb_resiliees"), 0)
        taux_resiliation = _safe_float(snapshot.get("taux_resiliation_pct"), 0.0)
        return {
            "context": context_line,
            "analysis": (
                f"Resiliation globale: {nb_resiliees:,} polices resiliees sur {total_polices:,}, "
                f"soit {taux_resiliation:.2f}%."
            ),
            "decision": "Le taux de resiliation global doit etre compare a la cible retention et segmente par branche.",
            "actions": [
                "Lancer des actions retention sur les segments les plus contributes a la resiliation.",
                "Suivre l evolution mensuelle du taux pour mesurer l effet des plans d action.",
            ],
        }

    if sql_id == "branch_resiliation_rate":
        worst = rows[0]
        worst_branch = str(worst.get("branche", "N/A"))
        worst_rate = _safe_float(worst.get("taux_resiliation_pct"), 0.0)
        return {
            "context": context_line,
            "analysis": f"Taux de resiliation le plus eleve: {worst_branch} a {worst_rate:.2f}%.",
            "decision": "Risque retention cible par branche; intervention proactive requise.",
            "actions": [
                "Lancer un plan de retention sur les branches en tete de resiliation.",
                "Analyser les motifs d annulation pour corriger les causes recurrentes.",
            ],
        }

    if sql_id == "top_governorates_by_impaye" or sql_id.startswith("impaye_by_gouvernorat_"):
        first = rows[0]
        gov = _first_dimension_value(first)
        metric_key, metric_value = _first_numeric_item(first)
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": f"Gouvernorat prioritaire {gov} avec {metric_label} {metric_value:,.0f}.",
            "decision": "Concentration geographique du risque impaye; prioriser les equipes terrain.",
            "actions": [
                "Affecter les actions recouvrement en priorite sur les zones geographiques les plus exposees.",
                "Suivre hebdomadairement la baisse du stock impaye sur le gouvernorat leader.",
            ],
        }

    if sql_id == "top_clients_impaye" or sql_id.startswith("top_clients_impaye_") or sql_id.startswith("impaye_by_client_"):
        first = rows[0]
        client_label = f"{first.get('nom', 'N/A')} {first.get('prenom', '')}".strip()
        metric_key, metric_value = _first_numeric_item(first)
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": f"Client le plus expose: {client_label} avec {metric_label} {metric_value:,.0f}.",
            "decision": "Pilotage recouvrement cible sur les clients les plus materialites.",
            "actions": [
                "Mettre en priorite les dossiers top impayes dans le plan de recouvrement.",
                "Segmenter ces clients par anciennete et risque de defaut pour adapter la strategie.",
            ],
        }

    if sql_id.startswith("impaye_by_branche_") or sql_id.startswith("prime_by_branche_"):
        leader = rows[0]
        dimension_value = _first_dimension_value(leader)
        metric_key, metric_value = _first_numeric_item(leader)
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": f"Leader sur la repartition: {dimension_value} avec {metric_label} {metric_value:,.0f}.",
            "decision": "Concentrer les actions de pilotage sur les dimensions en tete et surveiller la concentration.",
            "actions": [
                "Suivre mensuellement le top des dimensions pour detecter les changements de structure.",
                "Definir des objectifs de reequilibrage si la concentration depasse le seuil cible.",
            ],
        }

    if sql_id == "total_clients":
        total_clients = _safe_int(rows[0].get("total_clients"), 0)
        return {
            "context": context_line,
            "analysis": f"Volume portefeuille clients: {total_clients:,} clients.",
            "decision": "Utiliser cette base comme referentiel de penetration des offres et retention.",
            "actions": [
                "Suivre le taux d activation et de retention sur ce referentiel clients.",
                "Croiser avec impayes/resiliation pour prioriser les segments a valeur.",
            ],
        }

    return {
        "context": context_line,
        "analysis": f"{len(rows)} lignes exploitables retournees.",
        "decision": "Resultat SQL disponible pour pilotage metier.",
        "actions": ["Exploiter ce resultat dans un cycle de suivi mensuel."]
    }


def sql_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    report_mode = _infer_sql_report_mode(question)
    query_spec = _build_sql_query_spec(question, context)

    sql_id = str(query_spec["sql_id"])
    sql_query = str(query_spec["sql_query"])
    params = dict(query_spec.get("params", {}))

    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    df = _query_dataframe(sql_query, params)
    rows = _prepare_sql_rows(df, sql_id)

    report_details = _build_sql_report_details(
        sql_id=sql_id,
        rows=rows,
        branch=branch,
        year_from=year_from,
        year_to=year_to,
    )

    result_kind = str(query_spec.get("result_kind", "tabular")).lower()
    kpis = _build_sql_kpis(rows, query_spec)
    is_scalar_result = result_kind == "scalar" and len(rows) == 1

    chart_spec = query_spec.get("chart", {}) if isinstance(query_spec.get("chart"), dict) else {}
    include_chart = report_mode in {"report", "graph_only", "graph_pref"} and not is_scalar_result
    include_table = report_mode in {"report", "table_only", "table_pref", "graph_pref"}

    # Scalar outputs are displayed as KPI cards in UI by default.
    if is_scalar_result and report_mode in {"report", "graph_only", "graph_pref"}:
        include_table = False

    charts: list[dict[str, Any]] = []
    if include_chart and rows and chart_spec:
        chart_items = rows[:24]
        charts.append(
            _build_chart_payload(
                chart_type=str(chart_spec.get("type", "bar")),
                title=str(chart_spec.get("title", f"SQL chart {sql_id}")),
                x_key=str(chart_spec.get("x_key", list(chart_items[0].keys())[0])),
                y_key=str(chart_spec.get("y_key", list(chart_items[0].keys())[1] if len(chart_items[0].keys()) > 1 else list(chart_items[0].keys())[0])),
                items=chart_items,
            )
        )

    tables: list[dict[str, Any]] = []
    if include_table:
        table_columns = list(rows[0].keys()) if rows else []
        tables.append(
            {
                "title": f"SQL result {sql_id}",
                "columns": table_columns,
                "rows": rows,
                "markdown": _to_markdown_table(table_columns, rows),
            }
        )

    period_label = _format_period_label(year_from, year_to)
    scope_label = _format_scope_label(branch)
    summary = f"{report_details['analysis']} ({scope_label}, periode {period_label})."

    return {
        "tool": "sql_tool",
        "summary": summary,
        "payload": {
            "sql_id": sql_id,
            "rows": rows,
            "branch": branch or "ALL",
            "year_from": year_from,
            "year_to": year_to,
            "report_mode": report_mode,
            "result_kind": result_kind,
            "kpis": kpis,
            "semantic": query_spec.get("semantic", {}),
            "context": report_details.get("context", ""),
            "analysis": report_details.get("analysis", ""),
            "decision": report_details.get("decision", ""),
            "actions": report_details.get("actions", []),
        },
        "charts": charts,
        "tables": tables,
    }


def data_query_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    lowered = question.lower()
    should_use_sql = (
        "sql" in lowered
        or "requete" in lowered
        or "gouvernorat" in lowered
        or ("total" in lowered and "client" in lowered)
    )
    if should_use_sql:
        return sql_tool(question=question, context=context)
    return kpi_tool(question=question, context=context)


# Backward-compatible aliases
def tool_kpi(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return kpi_tool(question=question, context=context)


def tool_rag(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return rag_tool(question=question, context=context)


def tool_forecast(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return forecast_tool(question=question, context=context)


def tool_anomaly(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return anomaly_tool(question=question, context=context)


def tool_drift(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return drift_tool(question=question, context=context)


def tool_explain(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return explain_tool(question=question, context=context)


def tool_segment(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return segmentation_tool(question=question, context=context)


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


TOOL_REGISTRY = {
    "data_query_tool": data_query_tool,
    "kpi_tool": kpi_tool,
    "rag_tool": rag_tool,
    "alerte_tool": alerte_tool,
    "forecast_tool": forecast_tool,
    "anomaly_tool": anomaly_tool,
    "drift_tool": drift_tool,
    "explain_tool": explain_tool,
    "segmentation_tool": segmentation_tool,
    "client_tool": client_tool,
    "sql_tool": sql_tool,
    # Compatibility aliases used by previous prompts/tests
    "kpi": kpi_tool,
    "rag": rag_tool,
    "forecast": forecast_tool,
    "anomaly": anomaly_tool,
    "drift": drift_tool,
    "explain": explain_tool,
    "segment": segmentation_tool,
    "ml_predict_tool": ml_predict_tool,
    "predict": ml_predict_tool,
}


def run_tool(tool_name: str, question: str, context: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool '{tool_name}'")
    return TOOL_REGISTRY[tool_name](question=question, context=context)