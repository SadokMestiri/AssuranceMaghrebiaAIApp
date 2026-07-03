from __future__ import annotations

import json
import importlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from db import query_dataframe as _query_dataframe
from ml_pipeline import FEATURE_COLUMNS, get_impaye_operations_readiness, load_model_metadata, load_training_dataset
from utils import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    normalize_text as _normalize_text,
    format_metric_value as _format_metric_value,
    format_branch_label as _format_scope_label,
    infer_report_mode as _infer_report_mode,
)
from config import (
    QDRANT_URL, QDRANT_COLLECTIONS,
    OLLAMA_HOST, OLLAMA_EMBED_MODEL,
    DATA_YEAR_FROM, DATA_YEAR_TO,
    ALERTE_IMPAYE_RATE_PCT, ALERTE_PRODUCTION_DROP_PCT,
    NEXTJS_API_URL,
)

VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}
RAG_DOCUMENTS_PATH = Path(__file__).resolve().parent / "rag_documents.json"
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def _normalize_branch(branch: str | None) -> str | None:
    if not branch or branch.strip().upper() == "ALL":
        return None
    normalized = branch.strip().upper()
    if normalized not in VALID_BRANCHES:
        return None
    return normalized


def _resolve_period_context(context: dict[str, Any]) -> tuple[int, int]:
    # The frontend sends year_from/year_to explicitly; when absent fall back
    # to the full dataset range defined in config.
    year_from = _safe_int(context.get("year_from"), DATA_YEAR_FROM)
    year_to   = _safe_int(context.get("year_to"),   DATA_YEAR_TO)
    if year_from == 0:
        year_from = DATA_YEAR_FROM
    if year_to == 0:
        year_to = DATA_YEAR_TO
    if year_from > year_to:
        year_from, year_to = year_to, year_from
    return year_from, year_to



def _tokenize(text_value: str) -> set[str]:
    return {token.lower() for token in WORD_PATTERN.findall(text_value) if len(token) > 2}


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

    # NOTE: intentionally queries dim_police WITHOUT year filter to match the dashboard
    # (kpi_router.py churn_sql). The taux_resiliation is a portfolio-state KPI, not a
    # flow KPI — it reflects the current situation of all policies regardless of period.
    churn_sql = """
        SELECT
            COUNT(*) AS total_polices,
            SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_actives,
            SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
            ROUND(
                100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                2
            ) AS taux_resiliation
        FROM dim_police
        WHERE (:branch IS NULL OR branche = :branch)
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
        SELECT
            branche,
            COUNT(*) AS total_polices,
            SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
            ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS taux_resiliation_pct
        FROM dim_police
        WHERE branche IN ('AUTO', 'IRDS', 'SANTE')
          AND (:branch IS NULL OR branche = :branch)
        GROUP BY branche
        ORDER BY taux_resiliation_pct DESC
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

    # ── Optional dimension KPIs ────────────────────────────────────────────
    vehicule_kpis = {}
    try:
        v_sql = """
            SELECT COUNT(*) AS total_vehicules,
                   COUNT(DISTINCT TRIM(UPPER(v.marque))) AS nb_marques
            FROM dim_vehicule v JOIN dim_police p ON p.id_police = v.id_police
            WHERE (:branch IS NULL OR p.branche = :branch)
        """
        row = _query_dataframe(v_sql, params).iloc[0]
        vehicule_kpis = {
            "vehicule_total": _safe_int(row["total_vehicules"]),
            "vehicule_nb_marques": _safe_int(row["nb_marques"]),
        }
    except Exception:
        pass

    sinistre_kpis = {}
    try:
        s_sql = """
            SELECT COUNT(*) AS total_sinistres,
                   COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                   COALESCE(SUM(mt_evaluation), 0) AS total_provisions
            FROM dwh_fact_sinistre
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_survenance BETWEEN :year_from AND :year_to
        """
        row = _query_dataframe(s_sql, params).iloc[0]
        sinistre_kpis = {
            "nb_sinistres": _safe_int(row["total_sinistres"]),
            "total_mt_paye_sinistres": _safe_float(row["total_mt_paye"]),
            "total_provisions": _safe_float(row["total_provisions"]),
        }
    except Exception:
        pass

    client_kpis = {}
    try:
        c_sql = """
            SELECT COUNT(*) AS total_clients,
                   COUNT(DISTINCT ville) FILTER (WHERE ville IS NOT NULL AND TRIM(ville) != '') AS nb_villes
            FROM dim_client
        """
        row = _query_dataframe(c_sql, {}).iloc[0]
        client_kpis = {
            "client_total": _safe_int(row["total_clients"]),
            "client_nb_villes": _safe_int(row["nb_villes"]),
        }
    except Exception:
        pass

    police_kpis = {}
    try:
        pol_sql = """
            SELECT COUNT(*) AS total_polices,
                   SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_en_vigueur,
                   SUM(CASE WHEN type_police = 'flotte' THEN 1 ELSE 0 END) AS polices_flotte
            FROM dim_police WHERE (:branch IS NULL OR branche = :branch)
        """
        row = _query_dataframe(pol_sql, params).iloc[0]
        police_kpis = {
            "police_total": _safe_int(row["total_polices"]),
            "police_en_vigueur": _safe_int(row["polices_en_vigueur"]),
            "police_flotte": _safe_int(row["polices_flotte"]),
        }
    except Exception:
        pass

    agent_kpis = {}
    try:
        a_sql = """
            SELECT COUNT(*) AS total_agents,
                   SUM(CASE WHEN etat_agent = 'A' THEN 1 ELSE 0 END) AS agents_actifs
            FROM dim_agent
        """
        row = _query_dataframe(a_sql, {}).iloc[0]
        agent_kpis = {
            "agent_total": _safe_int(row["total_agents"]),
            "agent_actifs": _safe_int(row["agents_actifs"]),
        }
    except Exception:
        pass

    produit_kpis = {}
    try:
        prod_sql = """
            SELECT COUNT(DISTINCT code_produit) AS nb_produits,
                   COUNT(DISTINCT famille_risque) AS nb_familles
            FROM dim_produit WHERE (:branch IS NULL OR branche = :branch)
        """
        row = _query_dataframe(prod_sql, params).iloc[0]
        produit_kpis = {
            "produit_total": _safe_int(row["nb_produits"]),
            "produit_nb_familles": _safe_int(row["nb_familles"]),
        }
    except Exception:
        pass

    # ── Ratio Combiné RÉEL = (sinistres_payés + commission) / prime_nette ──
    # Formule identique à kpi_router.py (dashboard)
    total_mt_paye_sinistres = _safe_float(sinistre_kpis.get("total_mt_paye_sinistres", 0.0))
    ratio_combine_pct = round(
        100.0 * (total_mt_paye_sinistres + total_commission) / total_pnet, 2
    ) if total_pnet > 0 else 0.0
    # S/P pur (sinistres payés seulement / prime nette)
    sp_ratio_reel_pct = round(
        (100.0 * total_mt_paye_sinistres / total_pnet), 2
    ) if total_pnet > 0 else 0.0
    # Proxy impayé (gardé pour rétrocompat frontend)
    sp_ratio_proxy_pct = round(
        (100.0 * total_mt_acp / total_pnet), 2
    ) if total_pnet > 0 else 0.0

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
        # Ratio Combiné = (sinistres payés + commission) / prime nette
        "ratio_combine_pct": ratio_combine_pct,
        # S/P pur sinistres
        "sp_ratio_reel_pct": sp_ratio_reel_pct,
        # Proxy impayé (rétrocompat)
        "sp_ratio_proxy_pct": sp_ratio_proxy_pct,
        "taux_resiliation_pct": _safe_float(churn["taux_resiliation"]),
        "total_polices_churn": _safe_int(churn["total_polices"]),
        "polices_resiliees": _safe_int(churn["polices_resiliees"]),
        "top_branches": top_branches,
        "top_resiliation_branches": top_resiliation_branches,
        **vehicule_kpis,
        **sinistre_kpis,
        **client_kpis,
        **police_kpis,
        **agent_kpis,
        **produit_kpis,
    }


def _detect_kpi_focus(question: str) -> str:
    """Detect which single KPI the user is asking about. Returns focus key."""
    q = _normalize_text(question)
    if any(t in q for t in ["ratio combine", "ratio combin", "combined ratio", "combined"]):
        return "ratio_combine"
    if any(t in q for t in ["resiliation", "churn", "resilie"]):
        return "resiliation"
    if any(t in q for t in ["impaye", "impayes", "recouvrement"]):
        return "impaye"
    if any(t in q for t in ["sinistre", "sinistres", "s/p", "sp ratio", "sp reel"]):
        return "sinistre"
    if any(t in q for t in ["commission"]):
        return "commission"
    if any(t in q for t in ["prime", "pnet", "production", "quittance"]):
        return "prime"
    return "overview"  # full overview — no specific focus


def kpi_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    payload = _fetch_kpi_context_postgres(context)
    payload["source"] = "postgres"

    focus = _detect_kpi_focus(question)
    ratio_combine = _safe_float(payload.get("ratio_combine_pct"), 0.0)
    sp_reel = _safe_float(payload.get("sp_ratio_reel_pct"), 0.0)
    taux_resiliation = _safe_float(payload.get("taux_resiliation_pct"), 0.0)
    total_pnet = _safe_float(payload.get("total_pnet"), 0.0)
    total_commission = _safe_float(payload.get("total_commission"), 0.0)

    # ── Build focused KPIs based on what was asked ─────────────────────────
    if focus == "ratio_combine":
        focused_kpis = [
            {"key": "ratio_combine_pct",      "label": "Ratio Combiné",          "value": ratio_combine,    "unit": "%"},
            {"key": "sp_ratio_reel_pct",       "label": "S/P pur (sinistres/pnet)","value": sp_reel,         "unit": "%"},
            {"key": "total_mt_paye_sinistres", "label": "Sinistres payés",         "value": _safe_float(payload.get("total_mt_paye_sinistres"), 0.0), "unit": "TND"},
            {"key": "total_commission",        "label": "Commission",              "value": total_commission,  "unit": "TND"},
            {"key": "total_pnet",              "label": "Prime nette",             "value": total_pnet,        "unit": "TND"},
        ]
        summary = (
            f"Ratio Combiné {payload['branch']} {payload['year_from']}-{payload['year_to']}: "
            f"{ratio_combine:.2f}% "
            f"(S/P pur {sp_reel:.2f}% + expense {100.0*total_commission/total_pnet:.2f}% si pnet>0)."
        )
        charts = []  # no chart needed for a single ratio
        tables = [{
            "title": "Ratio Combiné",
            "columns": ["kpi", "valeur"],
            "rows": [
                {"kpi": "Ratio Combiné",            "valeur": f"{ratio_combine:.2f}%"},
                {"kpi": "S/P pur",                  "valeur": f"{sp_reel:.2f}%"},
                {"kpi": "Expense ratio",            "valeur": f"{100.0*total_commission/total_pnet:.2f}%" if total_pnet > 0 else "N/A"},
                {"kpi": "Sinistres payés",          "valeur": f"{_safe_float(payload.get('total_mt_paye_sinistres'),0):,.0f} TND"},
                {"kpi": "Commission",               "valeur": f"{total_commission:,.0f} TND"},
                {"kpi": "Prime nette",              "valeur": f"{total_pnet:,.0f} TND"},
            ],
            "markdown": "",
        }]

    elif focus == "resiliation":
        focused_kpis = [
            {"key": "taux_resiliation_pct", "label": "Taux de résiliation", "value": taux_resiliation, "unit": "%"},
            {"key": "polices_resiliees",    "label": "Polices résiliées",   "value": _safe_float(payload.get("polices_resiliees"), 0.0), "unit": "count"},
            {"key": "total_polices_churn",  "label": "Polices totales",     "value": _safe_float(payload.get("total_polices_churn"), 0.0), "unit": "count"},
        ]
        summary = f"Taux résiliation {payload['branch']} {payload['year_from']}-{payload['year_to']}: {taux_resiliation:.2f}%."
        charts = [_build_chart_payload("bar", "Résiliation par branche", "branche", "polices_resiliees",
                                        list(payload.get("top_resiliation_branches", [])))]
        tables = []

    elif focus == "sinistre":
        nb_sin = _safe_float(payload.get("nb_sinistres"), 0.0)
        mt_paye = _safe_float(payload.get("total_mt_paye_sinistres"), 0.0)
        focused_kpis = [
            {"key": "sp_ratio_reel_pct",       "label": "S/P réel",          "value": sp_reel,   "unit": "%"},
            {"key": "nb_sinistres",             "label": "Nb sinistres",       "value": nb_sin,    "unit": "count"},
            {"key": "total_mt_paye_sinistres",  "label": "Montant payé",       "value": mt_paye,   "unit": "TND"},
        ]
        summary = f"Sinistres {payload['branch']} {payload['year_from']}-{payload['year_to']}: {int(nb_sin):,} sinistres, S/P réel {sp_reel:.2f}%."
        charts = []
        tables = []

    elif focus == "impaye":
        nb_imp = _safe_float(payload.get("nb_impayes"), 0.0)
        mt_acp = _safe_float(payload.get("total_mt_acp"), 0.0)
        sp_proxy = _safe_float(payload.get("sp_ratio_proxy_pct"), 0.0)
        focused_kpis = [
            {"key": "nb_impayes",      "label": "Nb impayes",    "value": nb_imp,  "unit": "count"},
            {"key": "total_mt_acp",    "label": "Montant ACP",   "value": mt_acp,  "unit": "TND"},
            {"key": "taux_impaye_pnet", "label": "Taux impayé/prime nette", "value": sp_proxy, "unit": "%"},
        ]
        summary = f"Impayes {payload['branch']} {payload['year_from']}-{payload['year_to']}: {int(nb_imp):,} impayes, ACP {mt_acp:,.0f} TND."
        charts = []
        tables = []

    elif focus == "prime":
        nb_quitt = _safe_float(payload.get("nb_quittances"), 0.0)
        focused_kpis = [
            {"key": "total_pnet",    "label": "Prime nette",      "value": total_pnet,    "unit": "TND"},
            {"key": "total_commission","label": "Commission",     "value": total_commission,"unit": "TND"},
            {"key": "nb_quittances", "label": "Nb quittances",    "value": nb_quitt,       "unit": "count"},
        ]
        summary = f"Production {payload['branch']} {payload['year_from']}-{payload['year_to']}: prime nette {total_pnet:,.0f} TND, {int(nb_quitt):,} quittances."
        charts = [_build_chart_payload("bar", "Top branches par prime nette", "branche", "total_pnet",
                                        list(payload.get("top_branches", [])))]
        tables = []

    else:
        # Full overview — show everything
        focused_kpis = [
            {"key": "total_pnet",           "label": "Prime nette",     "value": total_pnet,      "unit": "TND"},
            {"key": "ratio_combine_pct",    "label": "Ratio Combiné",   "value": ratio_combine,   "unit": "%"},
            {"key": "taux_resiliation_pct", "label": "Taux résiliation","value": taux_resiliation,"unit": "%"},
            {"key": "nb_sinistres",         "label": "Sinistres",       "value": _safe_float(payload.get("nb_sinistres"),0.0), "unit": "count"},
            {"key": "nb_impayes",           "label": "Impayes",         "value": _safe_float(payload.get("nb_impayes"),0.0),  "unit": "count"},
        ]
        parts = [
            f"prime nette {total_pnet:,.0f} TND",
            f"ratio combine {ratio_combine:.2f}%",
            f"resiliation {taux_resiliation:.2f}%",
        ]
        summary = f"Vue globale KPI {payload['branch']} {payload['year_from']}-{payload['year_to']}: {', '.join(parts)}."
        charts = [_build_chart_payload("bar", "Top branches par prime nette", "branche", "total_pnet",
                                        list(payload.get("top_branches", [])))]
        overview_row = {
            "branch":               payload["branch"],
            "periode":              f"{payload['year_from']}-{payload['year_to']}",
            "prime_nette":          round(total_pnet, 0),
            "ratio_combine_pct":    ratio_combine,
            "taux_resiliation_pct": round(taux_resiliation, 2),
            "nb_quittances":        payload.get("nb_quittances", 0),
            "nb_sinistres":         payload.get("nb_sinistres", 0),
            "nb_impayes":           payload.get("nb_impayes", 0),
        }
        tables = [{
            "title": "Synthese KPI",
            "columns": list(overview_row.keys()),
            "rows": [overview_row],
            "markdown": _to_markdown_table(list(overview_row.keys()), [overview_row]),
        }]

    # Attach focused_kpis to payload so _compose_precise_metric_answer can read them
    payload["kpis"] = focused_kpis
    payload["focus"] = focus

    return {
        "tool": "kpi_tool",
        "summary": summary,
        "payload": payload,
        "charts": charts,
        "tables": tables,
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
    threshold_impaye_rate = ALERTE_IMPAYE_RATE_PCT
    threshold_drop_pct    = ALERTE_PRODUCTION_DROP_PCT

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
                COUNT(i.num_quittance) AS nb_impayes
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
    """
    Détecte la métrique/dimension demandée dans la question.
    """
    
    # ── Sinistre (dédicated table) ──
    sinistre_terms = ["sinistre", "sinistres", "montant paye sinistre", "provision", "provisions"]
    if any(term in normalized_question for term in sinistre_terms):
        if "nature" in normalized_question:
            return "sinistre_nature"
        if "etat" in normalized_question or "ouvert" in normalized_question or "clos" in normalized_question:
            return "sinistre_etat"
        if "responsabilite" in normalized_question:
            return "sinistre_responsabilite"
        if "branche" in normalized_question:
            return "sinistre_by_branche"
        if "evolution" in normalized_question or "mensuel" in normalized_question or "tendance" in normalized_question:
            return "sinistre_monthly_trend"
        return "sinistre_overview"
    if "sinistre" in normalized_question:
        if "materiel" in normalized_question or "matériel" in normalized_question:
            return "sinistre_materiel"
        if "gouvernorat" in normalized_question:
            return "sinistre_by_gouvernorat"

    
    # ── Impayé ──
    if any(term in normalized_question for term in ["impaye", "impayes", "acp", "recouvrement"]):
        if "branche" in normalized_question:
            return "impaye_rate_by_branch"
        if "gouvernorat" in normalized_question or "region" in normalized_question:
            return "impaye_by_gouvernorat"
        if "evolution" in normalized_question or "mensuel" in normalized_question:
            return "impaye_monthly_trend"
        if "client" in normalized_question or "top" in normalized_question:
            return "impaye_top_clients"
        return "impaye_overview"

    # ── Résiliation / Churn ──
    if any(term in normalized_question for term in ["resiliation", "annulation", "resilie", "resiliee", "churn"]):
        if "branche" in normalized_question:
            return "resiliation_by_branch"
        if "sexe" in normalized_question:
            return "resiliation_by_sexe"
        if "agent" in normalized_question:
            return "resiliation_by_agent"
        return "resiliation_overview"
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIENT DIMENSIONS - Conditions SPÉCIFIQUES d'abord
    # ═══════════════════════════════════════════════════════════════════════
    
    # D'abord les questions très spécifiques qui pourraient être mal classées
    if "age moyen" in normalized_question or "âge moyen" in normalized_question:
        return "client_age_moyen"
    
    if "tranche" in normalized_question and "age" in normalized_question:
        return "client_age_tranches"
    
    if "sexe" in normalized_question:
        return "client_sexe"
    
    if "personne physique" in normalized_question or "personnes physiques" in normalized_question:
        return "client_type_personne"

    if "personne morale" in normalized_question or "personnes morales" in normalized_question:
        return "client_type_personne"
    
    if "natp" in normalized_question or "nationalite" in normalized_question:
        return "client_natp"
    
    # Ensuite les questions client génériques
    client_terms = ["client", "clients", "assure", "assures"]
    if any(term in normalized_question for term in client_terms):
        if "ville" in normalized_question and ("couvert" in normalized_question or "distinct" in normalized_question or "top" in normalized_question):
            return "client_top_villes"
        if "nombre total" in normalized_question or "total clients" in normalized_question:
            return "client_total"
        return "client_overview"
    
    # ── Agent dimensions ──
    agent_terms = ["agent", "agents", "reseau", "distributeur"]
    if any(term in normalized_question for term in agent_terms):
        if "etat" in normalized_question or "actif" in normalized_question or "inactif" in normalized_question:
            return "agent_etat"
        if "type agent" in normalized_question or "type d'agent" in normalized_question:
            return "agent_type"
        if "groupe" in normalized_question:
            return "agent_groupe"
        if "localite" in normalized_question or "ville" in normalized_question:
            return "agent_localite"
        if "top" in normalized_question and ("prime" in normalized_question or "pnet" in normalized_question):
            return "agent_top_prime"
        if "top" in normalized_question and ("police" in normalized_question or "nombre" in normalized_question):
            return "agent_top_polices"
        if "prime moyenne" in normalized_question or "prime nette moyenne" in normalized_question:
            return "agent_prime_moyenne"
        if "agent" in normalized_question and "nombre de polices" in normalized_question and "top" in normalized_question:
            return "agent_top_polices"
        return "agent_overview"
    
    
    # ── Produit dimensions ──
    if any(term in normalized_question for term in [
        "produit", "produits",
        "famille de risque", "familles de risque", "famille risque",
        "top produit", "top produits",
        "prime nette par famille",
        "nombre de produits", "produits distincts"
    ]):
        if "famille" in normalized_question:
            return "produit_familles"
        if "top" in normalized_question and "prime" in normalized_question:
            return "produit_top_prime"
        if "top" in normalized_question and "quittance" in normalized_question:
            return "produit_top_quittances"
        if "branche" in normalized_question:
            return "produit_by_branche"
        if "total" in normalized_question and "prime" in normalized_question:
            return "produit_total_pnet"
        if "top" in normalized_question:
            return "produit_top_prime"
        return "produit_count"
    
    # ── Véhicule dimensions ──
    vehicule_terms = ["vehicule", "vehicules", "voiture", "voitures", "parc auto"]
    if any(term in normalized_question for term in vehicule_terms):
        if "marque" in normalized_question:
            return "vehicule_top_marques"
        if "genre" in normalized_question or "type" in normalized_question:
            return "vehicule_genre"
        if "puissance" in normalized_question and "moyenne" in normalized_question:
            return "vehicule_puissance_moyenne"
        if "puissance" in normalized_question and "tranche" in normalized_question:
            return "vehicule_puissance_tranches"
        if "age" in normalized_question or "anciennete" in normalized_question:
            return "vehicule_age_moyen"
        if "total" in normalized_question or "nombre" in normalized_question:
            return "vehicule_total"
        return "vehicule_overview"
    if "vehicule" in normalized_question or "voiture" in normalized_question:
        if "particuliere" in normalized_question or "vp" in normalized_question:
            return "vehicule_vp"
        if "utilitaire" in normalized_question or "vu" in normalized_question:
            return "vehicule_vu"
        if "marque" in normalized_question and "distinct" in normalized_question:
            return "vehicule_marques_distinctes"
        if "puissance" in normalized_question and "tranche" in normalized_question:
            return "vehicule_puissance_tranches"
        if "bonus malus" in normalized_question or "bm" in normalized_question:
            return "police_bm_moyen"
    
    # ── Police dimensions ──
    police_terms = ["police", "polices", "portefeuille"]
    if any(term in normalized_question for term in police_terms):
        if "situation" in normalized_question:
            return "police_situation"
        if "periodicite" in normalized_question or "reglement" in normalized_question:
            return "police_periodicite"
        if "type police" in normalized_question or "individuel" in normalized_question or "flotte" in normalized_question:
            return "police_type"
        if "duree" in normalized_question:
            return "police_duree"
        if "bonus malus" in normalized_question and "moyen" in normalized_question:
            return "police_bm_moyen"
        if "bonus malus" in normalized_question and "distribution" in normalized_question:
            return "police_bm_distribution"
        if "total" in normalized_question or "nombre" in normalized_question:
            return "police_total"
        return "police_overview"

    # QUITTANCES EMISES
    if "quittance" in normalized_question and ("nombre" in normalized_question or "nb" in normalized_question or "émises" in normalized_question or "emises" in normalized_question):
        return "quittances_emises"
    
    # ── KPI transactionnels ──
    if any(term in normalized_question for term in ["prime", "pnet", "production", "commission"]):
        if "evolution" in normalized_question or "mensuel" in normalized_question or "tendance" in normalized_question:
            return "prime_monthly_trend"
        if "branche" in normalized_question:
            return "prime_by_branch"
        if "annee" in normalized_question or "annuel" in normalized_question:
            return "prime_yearly"
        return "prime_overview"

    if any(term in normalized_question for term in ["ratio combine", "ratio combin", "combined ratio"]):
        return "ratio_combine"

    # ── Géographique ──
    if "gouvernorat" in normalized_question or "region" in normalized_question:
        if "sinistre" in normalized_question:
            return "sinistre_by_gouvernorat"
        if "impaye" in normalized_question:
            return "impaye_by_gouvernorat"
        return "top_zones_risque"
    if ("ville" in normalized_question and "couvert" in normalized_question) or "nombre de villes" in normalized_question:
        return "client_top_villes"
    if ("zone" in normalized_question and "risque" in normalized_question) or "top zones" in normalized_question:
        return "top_zones_risque"
    
    
    # ── Fallback par défaut ──
    return "prime_overview"


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
        "normalized_question": normalized_question,
    }


def _build_semantic_sql_query_spec(semantic: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | None:
    """
    Construit la requête SQL en fonction de la métrique/dimension détectée.
    """
    metric = str(semantic.get("metric", "prime_overview"))
    is_timeseries = bool(semantic.get("is_timeseries", False))
    is_ranking = bool(semantic.get("is_ranking", False))
    top_n = _safe_int(semantic.get("top_n"), 10)
    limit_value = top_n if is_ranking else 50
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIENT DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "client_total":
        return {
            "sql_id": "client_total",
            "sql_query": "SELECT COUNT(*) AS total_clients FROM dim_client",
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "total_clients", "label": "Total clients", "unit": "count"}],
        }
    
    if metric == "client_sexe":
        return {
            "sql_id": "client_by_sexe",
            "sql_query": """
                SELECT 
                    c.sexe,
                    COUNT(DISTINCT c.id_client) AS nb_clients,
                    ROUND(100.0 * COUNT(DISTINCT c.id_client) / SUM(COUNT(DISTINCT c.id_client)) OVER (), 1) AS pct
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                WHERE c.sexe IN ('F', 'M')
                AND p.situation = 'V'
                GROUP BY c.sexe
                ORDER BY c.sexe
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "sexe", "label": "Sexe", "unit": ""},
                {"key": "nb_clients", "label": "Nombre de clients (portefeuille actif)", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Répartition par sexe (portefeuille actif)", "x_key": "sexe", "y_key": "nb_clients"},
        }
    
    if metric == "client_age_moyen":
        return {
            "sql_id": "client_age_moyen",
            "sql_query": """
                SELECT 
                    ROUND(AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance))), 1) AS age_moyen
                FROM dim_client
                WHERE date_naissance IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "age_moyen", "label": "Âge moyen des clients", "unit": "ans"}],
        }
    
    if metric == "client_age_tranches":
        return {
            "sql_id": "client_age_tranches",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 25 THEN '<25'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 35 THEN '25-34'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 45 THEN '35-44'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 55 THEN '45-54'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 65 THEN '55-64'
                        ELSE '65+'
                    END AS tranche_age,
                    COUNT(*) AS nb_clients
                FROM dim_client
                WHERE date_naissance IS NOT NULL
                GROUP BY tranche_age
                ORDER BY MIN(EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)))
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "tranche_age", "label": "Tranche d'âge", "unit": ""},
                {"key": "nb_clients", "label": "Nombre de clients", "unit": "count"},
            ],
            "chart": {"type": "bar", "title": "Tranches d'âge des clients", "x_key": "tranche_age", "y_key": "nb_clients"},
        }
    
    if metric == "client_natp":
        return {
            "sql_id": "client_by_natp",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN natp = 'C' THEN 'Citoyen'
                        WHEN natp = 'P' THEN 'Permanent'
                        WHEN natp = 'R' THEN 'Résident'
                        WHEN natp = 'S' THEN 'Sans papiers'
                        ELSE natp
                    END AS nationalite,
                    COUNT(*) AS nb_clients,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_client
                WHERE natp IS NOT NULL
                GROUP BY natp
                ORDER BY nb_clients DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "nationalite", "label": "Nationalité", "unit": ""},
                {"key": "nb_clients", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Nationalité des clients", "x_key": "nationalite", "y_key": "nb_clients"},
        }
    
    if metric == "client_overview":
        return {
            "sql_id": "client_overview",
            "sql_query": """
                SELECT 
                    COUNT(DISTINCT c.id_client) AS total_clients,
                    COUNT(DISTINCT c.ville) AS nb_villes,
                    COUNT(DISTINCT CASE WHEN c.type_personne = 'P' THEN c.id_client END) AS nb_personnes_physiques,
                    COUNT(DISTINCT CASE WHEN c.type_personne = 'M' THEN c.id_client END) AS nb_personnes_morales,
                    COUNT(DISTINCT CASE WHEN c.sexe = 'F' THEN c.id_client END) AS nb_femmes,
                    COUNT(DISTINCT CASE WHEN c.sexe = 'M' THEN c.id_client END) AS nb_hommes,
                    ROUND(AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.date_naissance))), 1) AS age_moyen
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                WHERE p.situation = 'V'
                AND c.date_naissance IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_clients", "label": "Total clients (portefeuille actif)", "unit": "count"},
                {"key": "nb_villes", "label": "Villes couvertes", "unit": "count"},
                {"key": "nb_personnes_physiques", "label": "Personnes physiques", "unit": "count"},
                {"key": "nb_personnes_morales", "label": "Personnes morales", "unit": "count"},
                {"key": "nb_femmes", "label": "Femmes", "unit": "count"},
                {"key": "nb_hommes", "label": "Hommes", "unit": "count"},
                {"key": "age_moyen", "label": "Âge moyen", "unit": "ans"},
            ],
        }
    if metric == "client_type_personne":
        return {
            "sql_id": "client_by_type",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN c.type_personne = 'P' THEN 'Personne Physique'
                        WHEN c.type_personne = 'M' THEN 'Personne Morale'
                        ELSE c.type_personne
                    END AS type_client,
                    COUNT(DISTINCT c.id_client) AS nb_clients,
                    ROUND(100.0 * COUNT(DISTINCT c.id_client) / SUM(COUNT(DISTINCT c.id_client)) OVER (), 1) AS pct
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                WHERE c.type_personne IN ('P', 'M')
                AND p.situation = 'V'
                AND (:branch IS NULL OR p.branche = :branch)
                GROUP BY c.type_personne
                ORDER BY c.type_personne
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "type_client", "label": "Type", "unit": ""},
                {"key": "nb_clients", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Type de personne (portefeuille actif)", "x_key": "type_client", "y_key": "nb_clients"},
        }
    if metric == "client_top_villes":
        limit_value = top_n if is_ranking else 10
        return {
            "sql_id": "client_top_villes",
            "sql_query": f"""
                SELECT 
                    COALESCE(c.ville, 'N/A') AS ville,
                    COUNT(DISTINCT c.id_client) AS nb_clients
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                WHERE c.ville IS NOT NULL 
                AND TRIM(c.ville) != ''
                AND p.situation = 'V'
                AND (:branch IS NULL OR p.branche = :branch)
                GROUP BY c.ville
                ORDER BY nb_clients DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "ville", "label": "Ville", "unit": ""},
                {"key": "nb_clients", "label": "Nombre de clients (portefeuille actif)", "unit": "count"},
            ],
            "chart": {"type": "bar", "title": "Top villes par concentration clients", "x_key": "ville", "y_key": "nb_clients"},
        }

    
    # ═══════════════════════════════════════════════════════════════════════
    # AGENT DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "agent_overview":
        return {
            "sql_id": "agent_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_agents,
                    SUM(CASE WHEN etat_agent = 'A' THEN 1 ELSE 0 END) AS agents_actifs,
                    SUM(CASE WHEN etat_agent IN ('R', 'I') THEN 1 ELSE 0 END) AS agents_inactifs,
                    COUNT(DISTINCT groupe_agent) AS nb_groupes,
                    COUNT(DISTINCT localite_agent) AS nb_localites
                FROM dim_agent
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_agents", "label": "Total agents", "unit": "count"},
                {"key": "agents_actifs", "label": "Agents actifs", "unit": "count"},
                {"key": "agents_inactifs", "label": "Agents inactifs", "unit": "count"},
                {"key": "nb_groupes", "label": "Groupes de distribution", "unit": "count"},
                {"key": "nb_localites", "label": "Localités couvertes", "unit": "count"},
            ],
        }
    
    if metric == "agent_etat":
        return {
            "sql_id": "agent_by_etat",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN etat_agent = 'A' THEN 'Actif'
                        WHEN etat_agent = 'R' THEN 'Résilié'
                        WHEN etat_agent = 'I' THEN 'Inactif'
                        WHEN etat_agent = 'S' THEN 'Suspendu'
                        ELSE etat_agent
                    END AS statut,
                    COUNT(*) AS nb_agents,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_agent
                GROUP BY etat_agent
                ORDER BY nb_agents DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "statut", "label": "Statut", "unit": ""},
                {"key": "nb_agents", "label": "Nombre d'agents", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "État des agents", "x_key": "statut", "y_key": "nb_agents"},
        }
    
    if metric == "agent_type":
        return {
            "sql_id": "agent_by_type",
            "sql_query": """
                SELECT 
                    type_agent,
                    COUNT(*) AS nb_agents,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_agent
                WHERE type_agent IS NOT NULL
                GROUP BY type_agent
                ORDER BY nb_agents DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "type_agent", "label": "Type d'agent", "unit": ""},
                {"key": "nb_agents", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Type d'agent", "x_key": "type_agent", "y_key": "nb_agents"},
        }
    
    if metric == "agent_groupe":
        return {
            "sql_id": "agent_by_groupe",
            "sql_query": """
                SELECT 
                    groupe_agent,
                    COUNT(*) AS nb_agents,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_agent
                WHERE groupe_agent IS NOT NULL
                GROUP BY groupe_agent
                ORDER BY nb_agents DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "groupe_agent", "label": "Groupe", "unit": ""},
                {"key": "nb_agents", "label": "Nombre d'agents", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Agents par groupe", "x_key": "groupe_agent", "y_key": "nb_agents"},
        }
    
    if metric == "agent_localite":
        return {
            "sql_id": "agent_by_localite",
            "sql_query": f"""
                SELECT 
                    COALESCE(localite_agent, 'N/A') AS localite,
                    COUNT(*) AS nb_agents
                FROM dim_agent
                WHERE localite_agent IS NOT NULL
                GROUP BY localite_agent
                ORDER BY nb_agents DESC
                LIMIT {limit_value}
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "localite", "label": "Localité", "unit": ""},
                {"key": "nb_agents", "label": "Nombre d'agents", "unit": "count"},
            ],
            "chart": {"type": "bar", "title": "Top localités agents", "x_key": "localite", "y_key": "nb_agents"},
        }
    
    if metric == "agent_top_prime":
        return {
            "sql_id": "agent_top_prime",
            "sql_query": f"""
                SELECT 
                    COALESCE(a.nom_agent, 'N/A') AS agent,
                    COALESCE(a.groupe_agent, 'N/A') AS groupe,
                    COUNT(DISTINCT e.id_police) AS nb_polices,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_agent a
                JOIN dwh_fact_emission e ON e.id_agent = a.id_agent
                WHERE e.etat_quit IN ('E', 'P', 'A')
                  AND (:branch IS NULL OR e.branche = :branch)
                  AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY a.nom_agent, a.groupe_agent, a.id_agent
                ORDER BY total_pnet DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "groupe", "label": "Groupe", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top agents par prime nette", "x_key": "agent", "y_key": "total_pnet"},
        }
    
    if metric == "agent_prime_moyenne":
        return {
            "sql_id": "agent_prime_moyenne",
            "sql_query": """
                SELECT 
                    COALESCE(SUM(e.mt_pnet), 0) / NULLIF(SUM(CASE WHEN a.etat_agent = 'A' THEN 1 ELSE 0 END), 0) AS prime_moyenne_agent_actif
                FROM dim_agent a
                LEFT JOIN dwh_fact_emission e ON e.id_agent = a.id_agent
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR e.branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [{"key": "prime_moyenne_agent_actif", "label": "Prime nette moyenne par agent actif", "unit": "TND"}],
        }
    if metric == "agent_top_polices":
        limit_value = top_n if is_ranking else 10
        return {
            "sql_id": "agent_top_polices",
            "sql_query": f"""
                SELECT 
                    COALESCE(a.nom_agent, 'N/A') AS agent,
                    COALESCE(a.groupe_agent, 'N/A') AS groupe,
                    COUNT(DISTINCT e.id_police) AS nb_polices,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_agent a
                JOIN dwh_fact_emission e ON e.id_agent = a.id_agent
                WHERE e.etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR e.branche = :branch)
                AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY a.nom_agent, a.groupe_agent, a.id_agent
                ORDER BY nb_polices DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "groupe", "label": "Groupe", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top agents par nombre de polices", "x_key": "agent", "y_key": "nb_polices"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRODUIT DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "produit_count":
        return {
            "sql_id": "produit_count",
            "sql_query": """
                SELECT 
                    COUNT(DISTINCT code_produit) AS nb_produits,
                    COUNT(DISTINCT famille_risque) AS nb_familles,
                    COUNT(DISTINCT branche) AS nb_branches
                FROM dim_produit
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_produits", "label": "Produits distincts", "unit": "count"},
                {"key": "nb_familles", "label": "Familles de risque", "unit": "count"},
                {"key": "nb_branches", "label": "Branches couvertes", "unit": "count"},
            ],
        }
    
    if metric == "produit_familles":
        return {
            "sql_id": "produit_by_famille",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.famille_risque, 'N/A') AS famille,
                    COUNT(DISTINCT p.code_produit) AS nb_produits,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.famille_risque
                ORDER BY total_pnet DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "famille", "label": "Famille de risque", "unit": ""},
                {"key": "nb_produits", "label": "Nombre de produits", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Prime nette par famille de risque", "x_key": "famille", "y_key": "total_pnet"},
        }
    
    if metric == "produit_top_prime":
        return {
            "sql_id": "produit_top_prime",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.lib_produit, CAST(p.code_produit AS VARCHAR)) AS produit,
                    p.branche,
                    COUNT(DISTINCT pol.id_police) AS nb_polices,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.lib_produit, p.code_produit, p.branche
                ORDER BY total_pnet DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "produit", "label": "Produit", "unit": ""},
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top produits par prime nette", "x_key": "produit", "y_key": "total_pnet"},
        }
    
    if metric == "produit_by_branche":
        return {
            "sql_id": "produit_by_branche",
            "sql_query": """
                SELECT 
                    branche,
                    COUNT(DISTINCT code_produit) AS nb_produits,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY branche
                ORDER BY total_pnet DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_produits", "label": "Nb produits", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "pie", "title": "Prime nette par branche", "x_key": "branche", "y_key": "total_pnet"},
        }
    
    if metric == "produit_top_quittances":
        return {
            "sql_id": "produit_top_quittances",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.lib_produit, CAST(p.code_produit AS VARCHAR)) AS produit,
                    p.branche,
                    COUNT(e.num_quittance) AS nb_quittances,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.lib_produit, p.code_produit, p.branche
                ORDER BY nb_quittances DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "produit", "label": "Produit", "unit": ""},
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_quittances", "label": "Quittances", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top produits par volume quittances", "x_key": "produit", "y_key": "nb_quittances"},
        }
    if metric == "produit_total_pnet":
        return {
            "sql_id": "produit_total_pnet",
            "sql_query": """
                SELECT 
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
                    COUNT(DISTINCT e.num_quittance) AS nb_quittances
                FROM dwh_fact_emission e
                JOIN dim_police p ON p.id_police = e.id_police
                WHERE e.etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR p.branche = :branch)
                AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_pnet", "label": "Prime nette totale des produits", "unit": "TND"},
                {"key": "nb_quittances", "label": "Nombre de quittances", "unit": "count"},
            ],
        }
    if metric == "produit_top_single":
        return {
            "sql_id": "produit_top_single",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.lib_produit, CAST(p.code_produit AS VARCHAR)) AS produit,
                    p.branche,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.lib_produit, p.code_produit, p.branche
                ORDER BY total_pnet DESC
                LIMIT 1
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "produit", "label": "Top produit", "unit": ""},
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # VEHICULE DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "vehicule_total":
        return {
            "sql_id": "vehicule_total",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_vehicules,
                    COUNT(DISTINCT marque) AS nb_marques,
                    ROUND(AVG(puissance), 1) AS puissance_moyenne,
                    ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM date_mec)), 1) AS age_moyen
                FROM dim_vehicule
                WHERE puissance IS NOT NULL AND puissance > 0
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_vehicules", "label": "Total véhicules", "unit": "count"},
                {"key": "nb_marques", "label": "Marques distinctes", "unit": "count"},
                {"key": "puissance_moyenne", "label": "Puissance moyenne", "unit": "CV"},
                {"key": "age_moyen", "label": "Âge moyen", "unit": "ans"},
            ],
        }
    
    if metric == "vehicule_top_marques":
        return {
            "sql_id": "vehicule_top_marques",
            "sql_query": f"""
                SELECT 
                    COALESCE(marque, 'N/A') AS marque,
                    COUNT(*) AS nb_vehicules,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_vehicule
                WHERE marque IS NOT NULL
                GROUP BY marque
                ORDER BY nb_vehicules DESC
                LIMIT {limit_value}
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "marque", "label": "Marque", "unit": ""},
                {"key": "nb_vehicules", "label": "Nombre de véhicules", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Top marques par parc assuré", "x_key": "marque", "y_key": "nb_vehicules"},
        }
    
    if metric == "vehicule_genre":
        return {
            "sql_id": "vehicule_by_genre",
            "sql_query": """
                SELECT 
                    COALESCE(genre_vehicule, 'N/A') AS genre,
                    COUNT(*) AS nb_vehicules,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_vehicule
                WHERE genre_vehicule IS NOT NULL
                GROUP BY genre_vehicule
                ORDER BY nb_vehicules DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "genre", "label": "Genre", "unit": ""},
                {"key": "nb_vehicules", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Genre de véhicule", "x_key": "genre", "y_key": "nb_vehicules"},
        }
    
    if metric == "vehicule_puissance_moyenne":
        return {
            "sql_id": "vehicule_puissance_moyenne",
            "sql_query": """
                SELECT 
                    ROUND(AVG(puissance), 1) AS puissance_moyenne
                FROM dim_vehicule
                WHERE puissance IS NOT NULL AND puissance > 0
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "puissance_moyenne", "label": "Puissance moyenne", "unit": "CV"}],
        }
    
    if metric == "vehicule_age_moyen":
        return {
            "sql_id": "vehicule_age_moyen",
            "sql_query": """
                SELECT 
                    ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM date_mec)), 1) AS age_moyen
                FROM dim_vehicule
                WHERE date_mec IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "age_moyen", "label": "Âge moyen des véhicules", "unit": "ans"}],
        }
    
    if metric == "vehicule_overview":
        return {
            "sql_id": "vehicule_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_vehicules,
                    SUM(CASE WHEN genre_vehicule = 'VP' THEN 1 ELSE 0 END) AS nb_vp,
                    SUM(CASE WHEN genre_vehicule IN ('VU', 'PL', 'TC', 'AR') THEN 1 ELSE 0 END) AS nb_utilitaires,
                    COUNT(DISTINCT marque) AS nb_marques,
                    ROUND(AVG(puissance), 1) AS puissance_moyenne,
                    ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM date_mec)), 1) AS age_moyen
                FROM dim_vehicule
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_vehicules", "label": "Total véhicules", "unit": "count"},
                {"key": "nb_vp", "label": "Voitures particulières", "unit": "count"},
                {"key": "nb_utilitaires", "label": "Véhicules utilitaires", "unit": "count"},
                {"key": "nb_marques", "label": "Marques distinctes", "unit": "count"},
                {"key": "puissance_moyenne", "label": "Puissance moyenne", "unit": "CV"},
                {"key": "age_moyen", "label": "Âge moyen", "unit": "ans"},
            ],
        }
    if metric == "vehicule_vp":
        return {
            "sql_id": "vehicule_vp",
            "sql_query": """
                SELECT COUNT(*) AS nb_vp
                FROM dim_vehicule
                WHERE genre_vehicule = 'VP'
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_vp", "label": "Voitures particulières", "unit": "count"}],
        }
    if metric == "vehicule_vu":
        return {
            "sql_id": "vehicule_vu",
            "sql_query": """
                SELECT COUNT(*) AS nb_vu
                FROM dim_vehicule
                WHERE genre_vehicule IN ('VU', 'PL', 'TC', 'AR')
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_vu", "label": "Véhicules utilitaires", "unit": "count"}],
        }
    if metric == "vehicule_marques_distinctes":
        return {
            "sql_id": "vehicule_marques_distinctes",
            "sql_query": """
                SELECT COUNT(DISTINCT marque) AS nb_marques
                FROM dim_vehicule
                WHERE marque IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_marques", "label": "Marques distinctes", "unit": "count"}],
        }
    if metric == "vehicule_puissance_tranches":
        return {
            "sql_id": "vehicule_puissance_tranches",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN puissance <= 4 THEN '≤4 CV'
                        WHEN puissance <= 7 THEN '5-7 CV'
                        WHEN puissance <= 10 THEN '8-10 CV'
                        WHEN puissance <= 14 THEN '11-14 CV'
                        ELSE '15+ CV'
                    END AS tranche_puissance,
                    COUNT(*) AS nb_vehicules,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_vehicule
                WHERE puissance IS NOT NULL
                GROUP BY tranche_puissance
                ORDER BY MIN(puissance)
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "tranche_puissance", "label": "Tranche de puissance", "unit": ""},
                {"key": "nb_vehicules", "label": "Nombre de véhicules", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Répartition par tranche de puissance", "x_key": "tranche_puissance", "y_key": "nb_vehicules"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # POLICE DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "police_total":
        return {
            "sql_id": "police_total",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_en_vigueur,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    SUM(CASE WHEN type_police = 'flotte' THEN 1 ELSE 0 END) AS polices_flotte,
                    ROUND(AVG(bonus_malus), 2) AS bm_moyen
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_en_vigueur", "label": "En vigueur", "unit": "count"},
                {"key": "polices_resiliees", "label": "Résiliées", "unit": "count"},
                {"key": "polices_flotte", "label": "Polices flotte", "unit": "count"},
                {"key": "bm_moyen", "label": "Bonus-Malus moyen", "unit": ""},
            ],
        }
    
    if metric == "police_situation":
        return {
            "sql_id": "police_by_situation",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN situation = 'V' THEN 'En vigueur'
                        WHEN situation = 'R' THEN 'Résiliée'
                        WHEN situation = 'T' THEN 'Terminée'
                        WHEN situation = 'S' THEN 'Suspendue'
                        WHEN situation = 'A' THEN 'Annulée'
                        ELSE situation
                    END AS statut,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                GROUP BY situation
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "statut", "label": "Situation", "unit": ""},
                {"key": "nb_polices", "label": "Nombre de polices", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Situation du portefeuille", "x_key": "statut", "y_key": "nb_polices"},
        }
    
    if metric == "police_periodicite":
        return {
            "sql_id": "police_by_periodicite",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN periodicite = 'A' THEN 'Annuelle'
                        WHEN periodicite = 'S' THEN 'Semestrielle'
                        WHEN periodicite = 'T' THEN 'Trimestrielle'
                        WHEN periodicite = 'C' THEN 'Comptant'
                        ELSE periodicite
                    END AS periodicite_label,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                AND periodicite IS NOT NULL
                GROUP BY periodicite
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "periodicite_label", "label": "Périodicité", "unit": ""},
                {"key": "nb_polices", "label": "Nombre de polices", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Périodicité de règlement", "x_key": "periodicite_label", "y_key": "nb_polices"},
        }
    
    if metric == "police_type":
        return {
            "sql_id": "police_by_type",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN type_police = 'individuel' THEN 'Individuelle'
                        WHEN type_police = 'flotte' THEN 'Flotte'
                        ELSE type_police
                    END AS type_label,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                  AND type_police IS NOT NULL
                GROUP BY type_police
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "type_label", "label": "Type", "unit": ""},
                {"key": "nb_polices", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Type de police", "x_key": "type_label", "y_key": "nb_polices"},
        }
    
    if metric == "police_duree":
        return {
            "sql_id": "police_by_duree",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN duree = 'R' THEN 'Renouvelable'
                        WHEN duree = 'F' THEN 'Ferme'
                        WHEN duree = 'STR' THEN 'STR'
                        ELSE duree
                    END AS duree_label,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                  AND duree IS NOT NULL
                GROUP BY duree
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "duree_label", "label": "Durée", "unit": ""},
                {"key": "nb_polices", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Durée de police", "x_key": "duree_label", "y_key": "nb_polices"},
        }
    
    if metric == "police_bm_moyen":
        return {
            "sql_id": "police_bm_moyen",
            "sql_query": """
                SELECT 
                    ROUND(AVG(bonus_malus), 2) AS bm_moyen
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                AND bonus_malus IS NOT NULL
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [{"key": "bm_moyen", "label": "Bonus-Malus moyen", "unit": ""}],
        }
    
    if metric == "police_bm_distribution":
        return {
            "sql_id": "police_bm_distribution",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN bonus_malus <= 1 THEN '≤1 (Bonus)'
                        WHEN bonus_malus <= 1.5 THEN '1.01-1.5'
                        WHEN bonus_malus <= 2 THEN '1.51-2'
                        ELSE '>2 (Malus)'
                    END AS tranche_bm,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                AND bonus_malus IS NOT NULL
                GROUP BY tranche_bm
                ORDER BY MIN(bonus_malus)
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "tranche_bm", "label": "Bonus-Malus", "unit": ""},
                {"key": "nb_polices", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Distribution Bonus-Malus", "x_key": "tranche_bm", "y_key": "nb_polices"},
        }
    
    if metric == "police_overview":
        return {
            "sql_id": "police_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_en_vigueur,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    SUM(CASE WHEN type_police = 'flotte' THEN 1 ELSE 0 END) AS polices_flotte,
                    ROUND(AVG(bonus_malus), 2) AS bm_moyen,
                    ROUND(100.0 * SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) / COUNT(*), 1) AS taux_vigueur
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_en_vigueur", "label": "En vigueur", "unit": "count"},
                {"key": "polices_resiliees", "label": "Résiliées", "unit": "count"},
                {"key": "polices_flotte", "label": "Polices flotte", "unit": "count"},
                {"key": "bm_moyen", "label": "BM moyen", "unit": ""},
                {"key": "taux_vigueur", "label": "Taux d'en vigueur", "unit": "%"},
            ],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SINISTRE DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "sinistre_overview":
        return {
            "sql_id": "sinistre_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_sinistres,
                    SUM(CASE WHEN etat_sinistre = 'Ouvert' THEN 1 ELSE 0 END) AS sinistres_ouverts,
                    SUM(CASE WHEN etat_sinistre = 'Clos' THEN 1 ELSE 0 END) AS sinistres_clos,
                    COALESCE(SUM(mt_evaluation), 0) AS total_mt_evaluation,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                    SUM(CASE WHEN nature_sinistre = 'Matériel' THEN 1 ELSE 0 END) AS sinistres_materiels
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_sinistres", "label": "Total sinistres", "unit": "count"},
                {"key": "sinistres_ouverts", "label": "Sinistres ouverts", "unit": "count"},
                {"key": "sinistres_clos", "label": "Sinistres clos", "unit": "count"},
                {"key": "total_mt_evaluation", "label": "Montant évalué", "unit": "TND"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "sinistres_materiels", "label": "Sinistres matériels", "unit": "count"},
            ],
        }
    
    if metric == "sinistre_nature":
        return {
            "sql_id": "sinistre_by_nature",
            "sql_query": f"""
                SELECT 
                    COALESCE(nature_sinistre, 'N/A') AS nature,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                GROUP BY nature_sinistre
                ORDER BY nb_sinistres DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "nature", "label": "Nature", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Sinistres par nature", "x_key": "nature", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_etat":
        return {
            "sql_id": "sinistre_by_etat",
            "sql_query": """
                SELECT 
                    etat_sinistre,
                    COUNT(*) AS nb_sinistres,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                GROUP BY etat_sinistre
                ORDER BY nb_sinistres DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "etat_sinistre", "label": "État", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "État des sinistres", "x_key": "etat_sinistre", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_responsabilite":
        return {
            "sql_id": "sinistre_by_responsabilite",
            "sql_query": """
                SELECT 
                    responsabilite,
                    CASE 
                        WHEN responsabilite = 0 THEN '0% (Tiers responsable)'
                        WHEN responsabilite = 50 THEN '50% (Partage)'
                        WHEN responsabilite = 100 THEN '100% (Assuré)'
                        ELSE CAST(responsabilite AS VARCHAR) || '%'
                    END AS responsabilite_label,
                    COUNT(*) AS nb_sinistres,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                  AND responsabilite IS NOT NULL
                GROUP BY responsabilite
                ORDER BY responsabilite
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "responsabilite_label", "label": "Responsabilité", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Responsabilité engagée", "x_key": "responsabilite_label", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_by_branche":
        return {
            "sql_id": "sinistre_by_branche",
            "sql_query": """
                SELECT 
                    branche,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(mt_evaluation), 0) AS total_mt_evaluation,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                GROUP BY branche
                ORDER BY nb_sinistres DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "total_mt_evaluation", "label": "Montant évalué", "unit": "TND"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Sinistres par branche", "x_key": "branche", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_monthly_trend":
        return {
            "sql_id": "sinistre_monthly_trend",
            "sql_query": """
                SELECT 
                    make_date(annee_survenance, mois_survenance, 1) AS period,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(mt_evaluation), 0) AS total_mt_evaluation,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                  AND annee_survenance BETWEEN 1900 AND 2100
                  AND mois_survenance BETWEEN 1 AND 12
                GROUP BY make_date(annee_survenance, mois_survenance, 1)
                ORDER BY period
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "line", "title": "Évolution mensuelle sinistres", "x_key": "period", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_materiel":
        return {
            "sql_id": "sinistre_materiel",
            "sql_query": """
                SELECT COUNT(*) AS nb_sinistres_materiels
                FROM dwh_fact_sinistre
                WHERE nature_sinistre = 'Matériel'
                AND (:branch IS NULL OR branche = :branch)
                AND (:year_from IS NULL OR annee_survenance >= :year_from)
                AND (:year_to IS NULL OR annee_survenance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_sinistres_materiels", "label": "Sinistres matériels", "unit": "count"}],
        }
    if metric == "sinistre_by_gouvernorat":
        limit_value = top_n if is_ranking else 20
        return {
            "sql_id": "sinistre_by_gouvernorat",
            "sql_query": f"""
                SELECT 
                    TRIM(UPPER(c.ville)) AS gouvernorat,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(s.mt_paye), 0) AS total_mt_paye,
                    COALESCE(SUM(s.mt_evaluation), 0) AS total_mt_evaluation
                FROM dwh_fact_sinistre s
                JOIN dim_client c ON c.id_client = s.id_client
                WHERE c.ville IS NOT NULL
                AND TRIM(c.ville) != ''
                AND (:branch IS NULL OR s.branche = :branch)
                AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
                AND (:year_to IS NULL OR s.annee_survenance <= :year_to)
                GROUP BY TRIM(UPPER(c.ville))
                ORDER BY nb_sinistres DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "gouvernorat", "label": "Gouvernorat", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre de sinistres", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Sinistres par gouvernorat", "x_key": "gouvernorat", "y_key": "nb_sinistres"},
        }
    
    if metric == "top_zones_risque":
        limit_value = min(10, top_n) if is_ranking else 10
        return {
            "sql_id": "top_zones_risque",
            "sql_query": f"""
                WITH emission AS (
                    SELECT
                        TRIM(UPPER(c.ville)) AS gouvernorat,
                        COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                    FROM dwh_fact_emission e
                    JOIN dim_police p ON p.id_police = e.id_police        
                    JOIN dim_client c ON c.id_client = p.id_client        
                    WHERE e.etat_quit IN ('E','P','A')
                    AND e.mt_pnet >= 0
                    AND c.ville IS NOT NULL AND TRIM(c.ville) != ''
                    AND (:branch IS NULL OR e.branche = :branch)
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                    GROUP BY TRIM(UPPER(c.ville))
                ),
                sinistres AS (
                    SELECT
                        TRIM(UPPER(c.ville)) AS gouvernorat,
                        COUNT(*) AS nb_sinistres,
                        COALESCE(SUM(s.mt_paye), 0) AS total_mt_paye
                    FROM dwh_fact_sinistre s
                    JOIN dim_client c ON c.id_client = s.id_client
                    WHERE c.ville IS NOT NULL AND TRIM(c.ville) != ''
                    AND (:branch IS NULL OR s.branche = :branch)
                    AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
                    AND (:year_to IS NULL OR s.annee_survenance <= :year_to)
                    GROUP BY TRIM(UPPER(c.ville))
                ),
                merged AS (
                    SELECT
                        COALESCE(e.gouvernorat, s.gouvernorat) AS gouvernorat,
                        COALESCE(e.total_pnet, 0) AS total_pnet,
                        COALESCE(s.nb_sinistres, 0) AS nb_sinistres,
                        COALESCE(s.total_mt_paye, 0) AS total_mt_paye,
                        ROUND(100.0 * COALESCE(s.total_mt_paye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0), 2) AS taux_sinistres_sur_pnet_pct
                    FROM emission e
                    FULL OUTER JOIN sinistres s ON s.gouvernorat = e.gouvernorat
                )
                SELECT
                    gouvernorat,
                    total_pnet,
                    nb_sinistres,
                    total_mt_paye,
                    COALESCE(taux_sinistres_sur_pnet_pct, 0) AS taux_sinistres_sur_pnet_pct,
                    ROUND(
                        0.70 * COALESCE(taux_sinistres_sur_pnet_pct, 0)
                        + 30.0 * COALESCE(nb_sinistres, 0) / NULLIF(MAX(nb_sinistres) OVER (), 0),
                        2
                    ) AS score_risque
                FROM merged
                WHERE COALESCE(total_pnet, 0) > 0 OR COALESCE(nb_sinistres, 0) > 0
                ORDER BY score_risque DESC NULLS LAST
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "gouvernorat", "label": "Gouvernorat", "unit": ""},
                {"key": "nb_sinistres", "label": "Nb sinistres", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "taux_sinistres_sur_pnet_pct", "label": "Taux S/P", "unit": "%"},
                {"key": "score_risque", "label": "Score risque", "unit": ""},
            ],
            "chart": {"type": "bar", "title": "Top zones à risque", "x_key": "gouvernorat", "y_key": "score_risque"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # KPI TRANSACTIONNELS (existants, légèrement enrichis)
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "prime_overview":
        return {
            "sql_id": "prime_overview",
            "sql_query": """
                SELECT 
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COALESCE(AVG(mt_pnet), 0) AS avg_pnet,
                    COUNT(*) AS nb_quittances,
                    COUNT(DISTINCT id_police) AS nb_polices,
                    COALESCE(SUM(mt_commission), 0) AS total_commission
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_pnet", "label": "Prime nette totale", "unit": "TND"},
                {"key": "avg_pnet", "label": "Prime nette moyenne", "unit": "TND"},
                {"key": "nb_quittances", "label": "Nombre de quittances", "unit": "count"},
                {"key": "nb_polices", "label": "Polices distinctes", "unit": "count"},
                {"key": "total_commission", "label": "Commission totale", "unit": "TND"},
            ],
        }
    
    if metric == "ratio_combine":
        return {
            "sql_id": "ratio_combine",
            "sql_query": """
                WITH emission AS (
                    SELECT COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                           COALESCE(SUM(mt_commission), 0) AS total_commission
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:branch IS NULL OR branche = :branch)
                      AND (:year_from IS NULL OR annee_echeance >= :year_from)
                      AND (:year_to IS NULL OR annee_echeance <= :year_to)
                ),
                sinistre AS (
                    SELECT COALESCE(SUM(mt_paye), 0) AS total_mt_paye
                    FROM dwh_fact_sinistre
                    WHERE (:branch IS NULL OR branche = :branch)
                      AND (:year_from IS NULL OR annee_survenance >= :year_from)
                      AND (:year_to IS NULL OR annee_survenance <= :year_to)
                )
                SELECT 
                    e.total_pnet,
                    e.total_commission,
                    s.total_mt_paye,
                    ROUND(100.0 * (s.total_mt_paye + e.total_commission) / NULLIF(e.total_pnet, 0), 2) AS ratio_combine_pct,
                    ROUND(100.0 * s.total_mt_paye / NULLIF(e.total_pnet, 0), 2) AS sp_ratio_pct,
                    ROUND(100.0 * e.total_commission / NULLIF(e.total_pnet, 0), 2) AS expense_ratio_pct
                FROM emission e, sinistre s
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "ratio_combine_pct", "label": "Ratio Combiné", "unit": "%"},
                {"key": "sp_ratio_pct", "label": "S/P pur", "unit": "%"},
                {"key": "expense_ratio_pct", "label": "Expense ratio", "unit": "%"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
                {"key": "total_mt_paye", "label": "Sinistres payés", "unit": "TND"},
                {"key": "total_commission", "label": "Commission", "unit": "TND"},
            ],
        }
    
    if metric == "prime_monthly_trend":
        return {
            "sql_id": "prime_monthly_trend",
            "sql_query": """
                SELECT 
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COUNT(*) AS nb_quittances
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
                ORDER BY period
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "line", "title": "Évolution mensuelle prime nette", "x_key": "period", "y_key": "total_pnet"},
        }
    
    if metric == "prime_by_branch":
        return {
            "sql_id": "prime_by_branch",
            "sql_query": """
                SELECT 
                    branche,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COUNT(*) AS nb_quittances,
                    ROUND(100.0 * COALESCE(SUM(mt_pnet), 0) / SUM(COALESCE(SUM(mt_pnet), 0)) OVER (), 2) AS pct
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                GROUP BY branche
                ORDER BY total_pnet DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
                {"key": "nb_quittances", "label": "Quittances", "unit": "count"},
                {"key": "pct", "label": "Part", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Part de production par branche", "x_key": "branche", "y_key": "total_pnet"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # RÉSILIATION
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "resiliation_overview":
        return {
            "sql_id": "resiliation_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_actives,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / COUNT(*), 2) AS taux_resiliation_pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_actives", "label": "Polices actives", "unit": "count"},
                {"key": "polices_resiliees", "label": "Polices résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux de résiliation", "unit": "%"},
            ],
        }
    
    if metric == "resiliation_by_branch":
        return {
            "sql_id": "resiliation_by_branch",
            "sql_query": """
                SELECT 
                    branche,
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS nb_resiliees,
                    ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / COUNT(*), 2) AS taux_resiliation_pct
                FROM dim_police
                WHERE branche IN ('AUTO', 'IRDS', 'SANTE')
                  AND (:branch IS NULL OR branche = :branch)
                GROUP BY branche
                ORDER BY taux_resiliation_pct DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "nb_resiliees", "label": "Polices résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux de résiliation", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Résiliation par branche", "x_key": "branche", "y_key": "taux_resiliation_pct"},
        }
    
    if metric == "resiliation_by_sexe":
        return {
            "sql_id": "resiliation_by_sexe",
            "sql_query": """
                SELECT 
                    COALESCE(c.sexe, 'N/A') AS sexe,
                    COUNT(DISTINCT p.id_police) AS total_polices,
                    SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    ROUND(100.0 * SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) / COUNT(DISTINCT p.id_police), 2) AS taux_resiliation_pct
                FROM dim_police p
                LEFT JOIN dim_client c ON c.id_client = p.id_client
                WHERE (:branch IS NULL OR p.branche = :branch)
                  AND c.sexe IN ('F', 'M')
                GROUP BY c.sexe
                ORDER BY c.sexe
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "sexe", "label": "Sexe", "unit": ""},
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_resiliees", "label": "Polices résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Résiliation par sexe", "x_key": "sexe", "y_key": "polices_resiliees"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # IMPAYÉ
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "impaye_overview":
        return {
            "sql_id": "impaye_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS nb_impayes,
                    COALESCE(SUM(mt_acp), 0) AS total_impaye,
                    COALESCE(AVG(mt_acp), 0) AS avg_impaye,
                    COUNT(DISTINCT id_police) AS nb_polices_impactees
                FROM dwh_fact_impaye
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_impayes", "label": "Nombre d'impayés", "unit": "count"},
                {"key": "total_impaye", "label": "Montant total impayé", "unit": "TND"},
                {"key": "avg_impaye", "label": "Montant moyen", "unit": "TND"},
                {"key": "nb_polices_impactees", "label": "Polices impactées", "unit": "count"},
            ],
        }
    
    if metric == "impaye_rate_by_branch":
        return {
            "sql_id": "impaye_rate_by_branch",
            "sql_query": """
                WITH emission AS (
                    SELECT branche, COALESCE(SUM(mt_pnet), 0) AS total_pnet
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:year_from IS NULL OR annee_echeance >= :year_from)
                      AND (:year_to IS NULL OR annee_echeance <= :year_to)
                    GROUP BY branche
                ),
                impayes AS (
                    SELECT branche, COALESCE(SUM(mt_acp), 0) AS total_impaye
                    FROM dwh_fact_impaye
                    WHERE (:year_from IS NULL OR annee_echeance >= :year_from)
                      AND (:year_to IS NULL OR annee_echeance <= :year_to)
                    GROUP BY branche
                )
                SELECT 
                    COALESCE(e.branche, i.branche) AS branche,
                    COALESCE(e.total_pnet, 0) AS total_pnet,
                    COALESCE(i.total_impaye, 0) AS total_impaye,
                    ROUND(100.0 * COALESCE(i.total_impaye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0), 2) AS taux_impaye_pct
                FROM emission e
                FULL OUTER JOIN impayes i ON i.branche = e.branche
                ORDER BY taux_impaye_pct DESC NULLS LAST
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
                {"key": "total_impaye", "label": "Impayé", "unit": "TND"},
                {"key": "taux_impaye_pct", "label": "Taux impayé/prime", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Taux impayé par branche", "x_key": "branche", "y_key": "taux_impaye_pct"},
        }
    # ═══════════════════════════════════════════════════════════════════════
    # QUITTANCES EMISES
    # ═══════════════════════════════════════════════════════════════════════
    if metric == "quittances_emises":
        return {
            "sql_id": "quittances_emises",
            "sql_query": """
                SELECT 
                    COUNT(DISTINCT num_quittance) AS nb_quittances,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR branche = :branch)
                AND (:year_from IS NULL OR annee_echeance >= :year_from)
                AND (:year_to IS NULL OR annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_quittances", "label": "Nombre de quittances émises", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette totale", "unit": "TND"},
            ],
        }
    # ── Handlers for metrics detected above that had no spec ──────────────────

    if metric == "impaye_by_gouvernorat":
        return {
            "sql_id": "impaye_by_gouvernorat",
            "sql_query": f"""
                SELECT
                    TRIM(UPPER(c.ville)) AS gouvernorat,
                    COUNT(i.num_quittance) AS nb_impayes,
                    COALESCE(SUM(i.mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye i
                JOIN dim_police p ON p.id_police = i.id_police
                JOIN dim_client c ON c.id_client = p.id_client
                WHERE c.ville IS NOT NULL AND TRIM(c.ville) != ''
                  AND (:branch IS NULL OR i.branche = :branch)
                  AND (:year_from IS NULL OR i.annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR i.annee_echeance <= :year_to)
                GROUP BY TRIM(UPPER(c.ville))
                ORDER BY total_impaye DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "gouvernorat", "label": "Gouvernorat", "unit": ""},
                {"key": "nb_impayes", "label": "Nombre d'impayés", "unit": "count"},
                {"key": "total_impaye", "label": "Montant impayé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Impayés par gouvernorat", "x_key": "gouvernorat", "y_key": "total_impaye"},
        }

    if metric == "impaye_monthly_trend":
        return {
            "sql_id": "impaye_monthly_trend",
            "sql_query": """
                SELECT
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COUNT(*) AS nb_impayes,
                    COALESCE(SUM(mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
                ORDER BY period
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "line", "title": "Évolution mensuelle des impayés", "x_key": "period", "y_key": "total_impaye"},
        }

    if metric == "impaye_top_clients":
        return {
            "sql_id": "impaye_top_clients",
            "sql_query": f"""
                SELECT
                    COALESCE(c.prenom || ' ' || c.nom, c.nom, CAST(i.id_client AS VARCHAR)) AS client,
                    COUNT(i.num_quittance) AS nb_impayes,
                    COALESCE(SUM(i.mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye i
                JOIN dim_police p ON p.id_police = i.id_police
                JOIN dim_client c ON c.id_client = p.id_client
                WHERE (:branch IS NULL OR i.branche = :branch)
                  AND (:year_from IS NULL OR i.annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR i.annee_echeance <= :year_to)
                GROUP BY i.id_client, c.nom, c.prenom
                ORDER BY total_impaye DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "client", "label": "Client", "unit": ""},
                {"key": "nb_impayes", "label": "Impayés", "unit": "count"},
                {"key": "total_impaye", "label": "Montant impayé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top clients par montant impayé", "x_key": "client", "y_key": "total_impaye"},
        }

    if metric == "resiliation_by_agent":
        return {
            "sql_id": "resiliation_by_agent",
            "sql_query": f"""
                SELECT
                    COALESCE(a.nom_agent, CAST(p.id_agent AS VARCHAR)) AS agent,
                    COUNT(p.id_police) AS total_polices,
                    SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) AS nb_resiliees,
                    ROUND(
                        100.0 * SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) / COUNT(p.id_police), 2
                    ) AS taux_resiliation_pct
                FROM dim_police p
                LEFT JOIN dim_agent a ON a.id_agent = p.id_agent
                WHERE p.id_agent IS NOT NULL
                  AND (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.id_agent, a.nom_agent
                ORDER BY nb_resiliees DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "nb_resiliees", "label": "Résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux résiliation", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Résiliation par agent", "x_key": "agent", "y_key": "nb_resiliees"},
        }

    if metric == "prime_yearly":
        return {
            "sql_id": "prime_yearly",
            "sql_query": """
                SELECT
                    annee_echeance AS annee,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COUNT(DISTINCT id_police) AS nb_polices,
                    COUNT(*) AS nb_quittances
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E', 'P', 'A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                  AND annee_echeance BETWEEN 1900 AND 2100
                GROUP BY annee_echeance
                ORDER BY annee_echeance
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "bar", "title": "Prime nette par année", "x_key": "annee", "y_key": "total_pnet"},
        }

    return None


def _infer_sql_report_mode(question: str) -> str:
    return _infer_report_mode(question)


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

def dim_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Interroge l'API Next.js /api/dims pour les dimensions (clients, agents, produits, véhicules, polices, sinistres)
    """
    import requests
    
    # URL de l'API Next.js
    nextjs_api = NEXTJS_API_URL
    
    # Construire les paramètres
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    
    params = {}
    if branch:
        params["branch"] = branch
    if year_from:
        params["year_from"] = year_from
    if year_to:
        params["year_to"] = year_to
    
    try:
        response = requests.get(f"{nextjs_api}/api/dims", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {
            "tool": "dim_tool",
            "summary": f"Erreur d'accès aux données dimensions: {str(exc)}",
            "payload": {"error": str(exc)},
            "charts": [],
            "tables": [],
        }
    
    normalized = _normalize_text(question)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIENTS
    # ═══════════════════════════════════════════════════════════════════════
    if "client" in normalized or "assure" in normalized:
        clients_data = data.get("clients", {})
        kpis = clients_data.get("kpis", {})
        
        # Répartition par sexe
        if "sexe" in normalized:
            rows = clients_data.get("sexe", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des clients par sexe: {kpis.get('nb_f', 0)} femmes ({kpis.get('pct_f', 0)}%), {kpis.get('nb_m', 0)} hommes ({kpis.get('pct_m', 0)}%)",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "nb_femmes", "label": "Femmes", "value": kpis.get("nb_f", 0), "unit": "count"},
                    {"key": "nb_hommes", "label": "Hommes", "value": kpis.get("nb_m", 0), "unit": "count"},
                    {"key": "pct_f", "label": "Pourcentage femmes", "value": kpis.get("pct_f", 0), "unit": "%"},
                    {"key": "pct_m", "label": "Pourcentage hommes", "value": kpis.get("pct_m", 0), "unit": "%"},
                ]},
                "charts": [_build_chart_payload("pie", "Répartition par sexe", "label", "count", rows)],
                "tables": [{"title": "Clients par sexe", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Âge moyen
        if "age" in normalized:
            age_tranches = clients_data.get("ageTranches", [])
            return {
                "tool": "dim_tool",
                "summary": f"Âge moyen des clients: {kpis.get('age_moyen', 0)} ans",
                "payload": {"result_kind": "scalar", "rows": [{"age_moyen": kpis.get("age_moyen", 0)}], "kpis": [
                    {"key": "age_moyen", "label": "Âge moyen", "value": kpis.get("age_moyen", 0), "unit": "ans"},
                ]},
                "charts": [_build_chart_payload("bar", "Tranches d'âge", "label", "count", age_tranches)] if age_tranches else [],
                "tables": [],
            }
        
        # Type de personne
        if "type personne" in normalized or "personne physique" in normalized or "personne morale" in normalized:
            rows = clients_data.get("typePersonne", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par type de personne: {kpis.get('pct_moral', 0)}% personnes morales",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "pct_moral", "label": "Personnes morales", "value": kpis.get("pct_moral", 0), "unit": "%"},
                ]},
                "charts": [_build_chart_payload("pie", "Type de personne", "label", "count", rows)],
                "tables": [{"title": "Type de personne", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Nationalité
        if "nationalite" in normalized or "natp" in normalized:
            rows = clients_data.get("natp", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par nationalité",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Nationalité", "label", "count", rows)],
                "tables": [{"title": "Nationalité", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Top villes
        if "ville" in normalized:
            rows = clients_data.get("topVilles", [])
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} villes par concentration clients",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "nb_villes", "label": "Villes couvertes", "value": kpis.get("nb_villes", 0), "unit": "count"},
                ]},
                "charts": [_build_chart_payload("bar", "Top villes", "label", "count", rows)],
                "tables": [{"title": "Top villes", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Vue globale clients
        return {
            "tool": "dim_tool",
            "summary": f"Total clients: {kpis.get('total', 0)}, {kpis.get('pct_f', 0)}% femmes, {kpis.get('age_moyen', 0)} ans moy.",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total clients", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "nb_f", "label": "Femmes", "value": kpis.get("nb_f", 0), "unit": "count"},
                {"key": "nb_m", "label": "Hommes", "value": kpis.get("nb_m", 0), "unit": "count"},
                {"key": "pct_moral", "label": "Personnes morales", "value": kpis.get("pct_moral", 0), "unit": "%"},
                {"key": "nb_villes", "label": "Villes couvertes", "value": kpis.get("nb_villes", 0), "unit": "count"},
                {"key": "age_moyen", "label": "Âge moyen", "value": kpis.get("age_moyen", 0), "unit": "ans"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    if "agent" in normalized:
        agents_data = data.get("agents", {})
        kpis = agents_data.get("kpis", {})
        
        # État des agents
        if "etat" in normalized:
            rows = agents_data.get("etat", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des agents par état: {kpis.get('actifs', 0)} actifs, {kpis.get('inactifs', 0)} inactifs",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "État des agents", "label", "count", rows)],
                "tables": [{"title": "État des agents", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Groupe
        if "groupe" in normalized:
            rows = agents_data.get("groupes", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des agents par groupe",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Agents par groupe", "label", "count", rows)],
                "tables": [{"title": "Agents par groupe", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Type d'agent
        if "type" in normalized:
            rows = agents_data.get("typeAgent", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des agents par type",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Type d'agent", "label", "count", rows)],
                "tables": [{"title": "Type d'agent", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Localités
        if "localite" in normalized or "ville" in normalized:
            rows = agents_data.get("localites", [])[:10]
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} localités agents",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Top localités agents", "label", "count", rows)],
                "tables": [{"title": "Top localités", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Vue globale agents
        return {
            "tool": "dim_tool",
            "summary": f"Total agents: {kpis.get('total', 0)}, actifs: {kpis.get('actifs', 0)}",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total agents", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "actifs", "label": "Agents actifs", "value": kpis.get("actifs", 0), "unit": "count"},
                {"key": "inactifs", "label": "Agents inactifs", "value": kpis.get("inactifs", 0), "unit": "count"},
                {"key": "nb_groupes", "label": "Groupes", "value": kpis.get("nb_groupes", 0), "unit": "count"},
                {"key": "nb_localites", "label": "Localités", "value": kpis.get("nb_localites", 0), "unit": "count"},
                {"key": "avg_pnet", "label": "Prime nette moyenne", "value": kpis.get("avg_pnet", 0), "unit": "TND"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # FAMILLES DE RISQUE (standalone, no "produit" required)
    # ═══════════════════════════════════════════════════════════════════════

    if "famille" in normalized or "familles" in normalized:
        produits_data = data.get("produits", {})
        if "familles" in normalized and "nombre" in normalized:
            # Question: "nombre de familles de risque"
            kpis = produits_data.get("kpis", {})
            return {
                "tool": "dim_tool",
                "summary": f"Nombre de familles de risque: {kpis.get('nb_familles', 0)}",
                "payload": {
                    "result_kind": "scalar",
                    "rows": [{"nb_familles": kpis.get("nb_familles", 0)}],
                    "kpis": [{"key": "nb_familles", "label": "Familles de risque", "value": kpis.get("nb_familles", 0), "unit": "count"}]
                },
                "charts": [],
                "tables": [],
            }
        else:
            rows = produits_data.get("byFamille", [])
            return {
                "tool": "dim_tool",
                "summary": f"Prime nette par famille de risque",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Familles de risque", "label", "pnet", rows)],
                "tables": [{"title": "Familles de risque", "columns": ["label", "pnet"], "rows": rows}],
            }

    # ═══════════════════════════════════════════════════════════════════════
    # PRODUITS
    # ═══════════════════════════════════════════════════════════════════════
    if "produit" in normalized:
        produits_data = data.get("produits", {})
        kpis = produits_data.get("kpis", {})
        
        # Nombre de produits distincts
        if "nombre de produits" in normalized or "distinct" in normalized:
            return {
                "tool": "dim_tool",
                "summary": f"Produits distincts: {kpis.get('nb_produits', 0)}, Familles: {kpis.get('nb_familles', 0)}, Branches: {kpis.get('nb_branches', 0)}",
                "payload": {
                    "result_kind": "scalar",
                    "rows": [kpis],
                    "kpis": [
                        {"key": "nb_produits", "label": "Produits distincts", "value": kpis.get("nb_produits", 0), "unit": "count"},
                        {"key": "nb_familles", "label": "Familles de risque", "value": kpis.get("nb_familles", 0), "unit": "count"},
                        {"key": "nb_branches", "label": "Branches couvertes", "value": kpis.get("nb_branches", 0), "unit": "count"},
                        {"key": "total_pnet", "label": "Prime nette totale", "value": kpis.get("total_pnet", 0), "unit": "TND"},
                    ],
                },
                "charts": [],
                "tables": [],
            }
        
        # Familles de risque
        if "famille" in normalized:
            rows = produits_data.get("byFamille", [])
            return {
                "tool": "dim_tool",
                "summary": f"Prime nette par famille de risque",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Familles de risque", "label", "pnet", rows)],
                "tables": [{"title": "Familles de risque", "columns": ["label", "pnet"], "rows": rows}],
            }
        
        # Top produits
        if "top" in normalized:
            rows = produits_data.get("topProduits", [])
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} produits par prime nette",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Top produits", "label", "pnet", rows)],
                "tables": [{"title": "Top produits", "columns": ["label", "pnet"], "rows": rows}],
            }
    
    
    # ═══════════════════════════════════════════════════════════════════════
    # VÉHICULES
    # ═══════════════════════════════════════════════════════════════════════
    if "vehicule" in normalized or "voiture" in normalized:
        vehicules_data = data.get("vehicules", {})
        kpis = vehicules_data.get("kpis", {})
        
        # Top marques
        if "marque" in normalized:
            rows = vehicules_data.get("topMarques", [])
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} marques par parc assuré",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Top marques", "label", "count", rows)],
                "tables": [{"title": "Top marques", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Genre
        if "genre" in normalized:
            rows = vehicules_data.get("byGenre", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par genre de véhicule",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Genre de véhicule", "label", "count", rows)],
                "tables": [{"title": "Genre de véhicule", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Puissance
        if "puissance" in normalized:
            rows = vehicules_data.get("byPuissance", [])
            return {
                "tool": "dim_tool",
                "summary": f"Puissance moyenne: {kpis.get('avg_puissance', 0)} CV",
                "payload": {"result_kind": "scalar", "rows": [{"puissance_moyenne": kpis.get("avg_puissance", 0)}], "kpis": [
                    {"key": "avg_puissance", "label": "Puissance moyenne", "value": kpis.get("avg_puissance", 0), "unit": "CV"},
                ]},
                "charts": [_build_chart_payload("bar", "Puissance fiscale", "label", "count", rows)] if rows else [],
                "tables": [],
            }
        
        # Vue globale véhicules
        return {
            "tool": "dim_tool",
            "summary": f"Total véhicules: {kpis.get('total', 0)}, Puissance moyenne: {kpis.get('avg_puissance', 0)} CV",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total véhicules", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "nb_vp", "label": "Voitures particulières", "value": kpis.get("nb_vp", 0), "unit": "count"},
                {"key": "nb_vu", "label": "Véhicules utilitaires", "value": kpis.get("nb_vu", 0), "unit": "count"},
                {"key": "nb_marques", "label": "Marques distinctes", "value": kpis.get("nb_marques", 0), "unit": "count"},
                {"key": "avg_puissance", "label": "Puissance moyenne", "value": kpis.get("avg_puissance", 0), "unit": "CV"},
                {"key": "avg_age", "label": "Âge moyen", "value": kpis.get("avg_age", 0), "unit": "ans"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # POLICES
    # ═══════════════════════════════════════════════════════════════════════
    if "police" in normalized:
        polices_data = data.get("polices", {})
        kpis = polices_data.get("kpis", {})
        
        # Situation
        if "situation" in normalized:
            rows = polices_data.get("bySituation", [])
            return {
                "tool": "dim_tool",
                "summary": f"Situation du portefeuille: {kpis.get('en_vigueur', 0)} polices en vigueur",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Situation du portefeuille", "label", "count", rows)],
                "tables": [{"title": "Situation", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Périodicité
        if "periodicite" in normalized:
            rows = polices_data.get("byPeriodicite", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par périodicité",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Périodicité", "label", "count", rows)],
                "tables": [{"title": "Périodicité", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Type de police
        if "type" in normalized:
            rows = polices_data.get("byType", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par type de police: {kpis.get('pct_indiv', 0)}% individuelles, {kpis.get('pct_flotte', 0)}% flotte",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Type de police", "label", "count", rows)],
                "tables": [{"title": "Type de police", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Bonus-Malus
        if "bonus malus" in normalized:
            rows = polices_data.get("bonusMalus", [])
            return {
                "tool": "dim_tool",
                "summary": f"Distribution Bonus-Malus, moyenne: {kpis.get('avg_bm', 0)}",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "avg_bm", "label": "Bonus-Malus moyen", "value": kpis.get("avg_bm", 0), "unit": ""},
                ]},
                "charts": [_build_chart_payload("bar", "Distribution Bonus-Malus", "label", "count", rows)],
                "tables": [{"title": "Bonus-Malus", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Vue globale polices
        return {
            "tool": "dim_tool",
            "summary": f"Total polices: {kpis.get('total', 0)}, En vigueur: {kpis.get('en_vigueur', 0)}",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total polices", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "en_vigueur", "label": "En vigueur", "value": kpis.get("en_vigueur", 0), "unit": "count"},
                {"key": "resiliees", "label": "Résiliées", "value": kpis.get("resiliees", 0), "unit": "count"},
                {"key": "individuelles", "label": "Individuelles", "value": kpis.get("individuelles", 0), "unit": "count"},
                {"key": "flottes", "label": "Flotte", "value": kpis.get("flottes", 0), "unit": "count"},
                {"key": "avg_bm", "label": "BM moyen", "value": kpis.get("avg_bm", 0), "unit": ""},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SINISTRES
    # ═══════════════════════════════════════════════════════════════════════
    if "sinistre" in normalized:
        sinistres_data = data.get("sinistres", {})
        kpis = sinistres_data.get("kpis", {})
        
        # Par nature
        if "nature" in normalized:
            rows = sinistres_data.get("byNature", [])
            return {
                "tool": "dim_tool",
                "summary": f"Sinistres par nature",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Sinistres par nature", "label", "count", rows)],
                "tables": [{"title": "Sinistres par nature", "columns": ["label", "count"], "rows": rows}],
            }
        
        # État
        if "etat" in normalized:
            rows = sinistres_data.get("byEtat", [])
            return {
                "tool": "dim_tool",
                "summary": f"État des sinistres: {kpis.get('ouverts', 0)} ouverts, {kpis.get('clos', 0)} clos",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "État des sinistres", "label", "count", rows)],
                "tables": [{"title": "État des sinistres", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Responsabilité
        if "responsabilite" in normalized:
            rows = sinistres_data.get("byResponsabilite", [])
            return {
                "tool": "dim_tool",
                "summary": f"Responsabilité engagée",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Responsabilité engagée", "label", "count", rows)],
                "tables": [{"title": "Responsabilité", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Par branche
        if "branche" in normalized:
            rows = sinistres_data.get("byBranche", [])
            return {
                "tool": "dim_tool",
                "summary": f"Sinistres par branche",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Sinistres par branche", "label", "count", rows)],
                "tables": [{"title": "Sinistres par branche", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Évolution mensuelle
        if "evolution" in normalized or "mensuel" in normalized:
            rows = sinistres_data.get("monthly", [])
            return {
                "tool": "dim_tool",
                "summary": f"Évolution mensuelle des sinistres",
                "payload": {"result_kind": "timeseries", "rows": rows},
                "charts": [_build_chart_payload("line", "Évolution mensuelle sinistres", "label", "count", rows)],
                "tables": [],
            }
        
        # Vue globale sinistres
        return {
            "tool": "dim_tool",
            "summary": f"Total sinistres: {kpis.get('total', 0)}, Ouverts: {kpis.get('ouverts', 0)}, Montant payé: {kpis.get('total_paye', 0):,.0f} TND",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total sinistres", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "ouverts", "label": "Ouverts", "value": kpis.get("ouverts", 0), "unit": "count"},
                {"key": "clos", "label": "Clos", "value": kpis.get("clos", 0), "unit": "count"},
                {"key": "total_eval", "label": "Montant évalué", "value": kpis.get("total_eval", 0), "unit": "TND"},
                {"key": "total_paye", "label": "Montant payé", "value": kpis.get("total_paye", 0), "unit": "TND"},
                {"key": "nb_materiel", "label": "Sinistres matériels", "value": kpis.get("nb_materiel", 0), "unit": "count"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # Fallback
    return {
        "tool": "dim_tool",
        "summary": "Je peux vous fournir des informations sur les clients, agents, produits, véhicules, polices et sinistres. Que souhaitez-vous exactement ?",
        "payload": {"result_kind": "text", "message": "Spécifiez la dimension (clients, agents, produits, véhicules, polices, sinistres) et l'information souhaitée."},
        "charts": [],
        "tables": [],
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
    "dim_tool": dim_tool,
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