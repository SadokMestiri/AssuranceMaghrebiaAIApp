from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_DNS, uuid4, uuid5

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from db import engine as db_engine


BASE_DIR = Path(__file__).resolve().parent
RAG_DOCUMENTS_PATH = BASE_DIR / "rag_documents.json"

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _query_dataframe(sql_query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    return pd.read_sql(text(sql_query), db_engine, params=params or {})


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return float(value)
        return None

    if isinstance(value, (int, np.integer, str, bool)):
        return value

    return str(value)


def _load_business_rules_documents() -> list[dict[str, Any]]:
    if not RAG_DOCUMENTS_PATH.exists():
        return []
    with RAG_DOCUMENTS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    docs = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        docs.append(
            {
                "id": f"business_rules_{idx}",
                "title": str(item.get("title", f"Regle {idx + 1}")),
                "content": str(item.get("content", "")).strip(),
                "source": "rag_documents.json",
            }
        )
    return docs


def _load_kpi_history_documents() -> list[dict[str, Any]]:
    sql_query = """
        WITH emission AS (
            SELECT
                make_date(annee_echeance, mois_echeance, 1) AS period,
                branche,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                COALESCE(SUM(mt_commission), 0) AS total_commission
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
                            AND annee_echeance BETWEEN 1900 AND 2100
                            AND mois_echeance BETWEEN 1 AND 12
              AND make_date(annee_echeance, mois_echeance, 1) >= date_trunc('month', current_date) - interval '18 months'
            GROUP BY make_date(annee_echeance, mois_echeance, 1), branche
        ),
        impaye AS (
            SELECT
                make_date(annee_echeance, mois_echeance, 1) AS period,
                branche,
                COALESCE(SUM(mt_acp), 0) AS total_impaye
            FROM dwh_fact_impaye
                        WHERE annee_echeance BETWEEN 1900 AND 2100
                            AND mois_echeance BETWEEN 1 AND 12
                            AND make_date(annee_echeance, mois_echeance, 1) >= date_trunc('month', current_date) - interval '18 months'
            GROUP BY make_date(annee_echeance, mois_echeance, 1), branche
        )
        SELECT
            COALESCE(e.period, i.period) AS period,
            COALESCE(e.branche, i.branche, 'ALL') AS branche,
            COALESCE(e.total_pnet, 0) AS total_pnet,
            COALESCE(e.total_commission, 0) AS total_commission,
            COALESCE(i.total_impaye, 0) AS total_impaye,
            ROUND(100.0 * COALESCE(i.total_impaye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0), 3) AS impaye_rate_pct
        FROM emission e
        FULL OUTER JOIN impaye i ON i.period = e.period AND i.branche = e.branche
        ORDER BY period DESC, branche
        LIMIT 600
    """

    df = _query_dataframe(sql_query)
    docs = []
    for _, row in df.iterrows():
        period = str(row["period"])[:10]
        branche = str(row["branche"])
        total_pnet = float(row["total_pnet"] or 0.0)
        total_commission = float(row["total_commission"] or 0.0)
        total_impaye = float(row["total_impaye"] or 0.0)
        impaye_rate = float(row["impaye_rate_pct"] or 0.0)

        docs.append(
            {
                "id": f"kpi_{period}_{branche}",
                "title": f"KPI {branche} {period}",
                "content": (
                    f"Periode {period}. Branche {branche}. Prime nette {total_pnet:.2f} TND. "
                    f"Commission {total_commission:.2f} TND. Impaye {total_impaye:.2f} TND. "
                    f"Taux impaye {impaye_rate:.3f}%"
                ),
                "source": "postgres_kpi",
                "metadata": {
                    "period": period,
                    "branche": branche,
                    "total_pnet": total_pnet,
                    "total_commission": total_commission,
                    "total_impaye": total_impaye,
                    "impaye_rate_pct": impaye_rate,
                },
            }
        )
    return docs


def _load_alert_history_documents() -> list[dict[str, Any]]:
    sql_query = """
        WITH monthly AS (
            SELECT
                make_date(i.annee_echeance, i.mois_echeance, 1) AS period,
                i.branche,
                COALESCE(SUM(i.mt_acp), 0) AS total_impaye,
                COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
            FROM dwh_fact_impaye i
            LEFT JOIN dwh_fact_emission e
              ON e.num_quittance = i.num_quittance
             AND e.etat_quit IN ('E','P','A')
                         AND e.annee_echeance BETWEEN 1900 AND 2100
                         AND e.mois_echeance BETWEEN 1 AND 12
                        WHERE i.annee_echeance BETWEEN 1900 AND 2100
                            AND i.mois_echeance BETWEEN 1 AND 12
                            AND make_date(i.annee_echeance, i.mois_echeance, 1) >= date_trunc('month', current_date) - interval '18 months'
            GROUP BY make_date(i.annee_echeance, i.mois_echeance, 1), i.branche
        )
        SELECT
            period,
            branche,
            total_impaye,
            total_pnet,
            ROUND(100.0 * total_impaye / NULLIF(total_pnet, 0), 3) AS impaye_rate_pct
        FROM monthly
        ORDER BY period DESC, branche
        LIMIT 600
    """

    df = _query_dataframe(sql_query)
    docs = []
    for _, row in df.iterrows():
        period = str(row["period"])[:10]
        branche = str(row["branche"])
        total_impaye = float(row["total_impaye"] or 0.0)
        total_pnet = float(row["total_pnet"] or 0.0)
        rate = float(row["impaye_rate_pct"] or 0.0)

        level = "high" if rate >= 2.0 else "medium" if rate >= 1.0 else "low"
        docs.append(
            {
                "id": f"alert_{period}_{branche}",
                "title": f"Alerte impaye {branche} {period}",
                "content": (
                    f"Alerte {level}. Branche {branche}, periode {period}. "
                    f"Impaye {total_impaye:.2f} TND pour prime {total_pnet:.2f} TND. "
                    f"Taux impaye {rate:.3f}%"
                ),
                "source": "postgres_alert",
                "metadata": {
                    "period": period,
                    "branche": branche,
                    "total_impaye": total_impaye,
                    "total_pnet": total_pnet,
                    "impaye_rate_pct": rate,
                    "alert_level": level,
                },
            }
        )
    return docs


def _hash_fallback_embedding(text_value: str, vector_size: int = 384) -> list[float]:
    digest = hashlib.sha256(text_value.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=vector_size)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return [0.0] * vector_size
    return (vector / norm).astype(float).tolist()


def _embed_text(text_value: str) -> tuple[list[float], str]:
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": text_value,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Invalid embedding payload from Ollama")
        return [float(item) for item in embedding], "ollama"
    except Exception:
        return _hash_fallback_embedding(text_value), "hash_fallback"


def _ensure_collection(collection_name: str, vector_size: int) -> None:
    probe = requests.get(f"{QDRANT_URL}/collections/{collection_name}", timeout=10)
    if probe.status_code == 200:
        return

    response = requests.put(
        f"{QDRANT_URL}/collections/{collection_name}",
        json={
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        },
        timeout=15,
    )
    if response.status_code not in {200, 201}:
        raise RuntimeError(f"Cannot create/update collection {collection_name}: {response.text}")


def _upsert_points(collection_name: str, points: list[dict[str, Any]]) -> None:
    if not points:
        return
    response = requests.put(
        f"{QDRANT_URL}/collections/{collection_name}/points?wait=true",
        json={"points": points},
        timeout=25,
    )
    if response.status_code not in {200, 201}:
        raise RuntimeError(f"Cannot upsert points in {collection_name}: {response.text}")


def _to_qdrant_point_id(raw_id: Any) -> str | int:
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        normalized = raw_id.strip()
        if normalized.isdigit():
            return int(normalized)
        # Deterministic UUID from semantic identifier.
        return str(uuid5(NAMESPACE_DNS, f"maghrebia:{normalized}"))
    return str(uuid4())


def _build_points(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[str]]:
    points: list[dict[str, Any]] = []
    vector_size = 0
    engines_used: set[str] = set()

    for document in documents:
        content = str(document.get("content", "")).strip()
        if not content:
            continue

        embedding, engine_name = _embed_text(content)
        engines_used.add(engine_name)
        vector_size = len(embedding)
        payload = {
            "title": document.get("title", ""),
            "content": content,
            "source": document.get("source", ""),
            "metadata": _sanitize_for_json(document.get("metadata", {})),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "embedding_engine": engine_name,
        }

        points.append(
            {
                "id": _to_qdrant_point_id(document.get("id") or uuid4()),
                "vector": embedding,
                "payload": payload,
            }
        )

    if vector_size == 0:
        raise RuntimeError("No vectors were generated")

    return points, vector_size, sorted(engines_used)


def run_indexing(max_docs_per_collection: int = 400) -> dict[str, Any]:
    collections = {
        "business_rules": _load_business_rules_documents(),
        "kpi_history": _load_kpi_history_documents(),
        "alert_history": _load_alert_history_documents(),
    }

    report: dict[str, Any] = {
        "status": "ok",
        "qdrant_url": QDRANT_URL,
        "ollama_host": OLLAMA_HOST,
        "embed_model": OLLAMA_EMBED_MODEL,
        "collections": {},
    }

    for collection_name, documents in collections.items():
        sliced_documents = documents[:max_docs_per_collection]
        if not sliced_documents:
            report["collections"][collection_name] = {
                "indexed": 0,
                "skipped": True,
                "reason": "no_documents",
            }
            continue

        points, vector_size, engines_used = _build_points(sliced_documents)
        _ensure_collection(collection_name, vector_size)
        _upsert_points(collection_name, points)

        report["collections"][collection_name] = {
            "indexed": len(points),
            "vector_size": vector_size,
            "source_documents": len(sliced_documents),
            "embedding_engines": engines_used,
        }

    return report


if __name__ == "__main__":
    outcome = run_indexing()
    print(json.dumps(outcome, ensure_ascii=True, indent=2))
