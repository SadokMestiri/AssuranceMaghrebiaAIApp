from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from utils import safe_float as _safe_float, safe_int as _safe_int
from config import QDRANT_URL, QDRANT_COLLECTIONS, OLLAMA_HOST, OLLAMA_EMBED_MODEL
from tools._shared import _to_markdown_table, _build_chart_payload

RAG_DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "rag_documents.json"
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text_value: str) -> set[str]:
    return {token.lower() for token in WORD_PATTERN.findall(text_value) if len(token) > 2}


def _load_rag_documents() -> list[dict[str, Any]]:
    if not RAG_DOCUMENTS_PATH.exists():
        return []
    with RAG_DOCUMENTS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [doc for doc in payload if isinstance(doc, dict)]


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

    n_q = max(len(question_tokens), 1)
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        title_tokens = _tokenize(document.get("title", ""))
        content_tokens = _tokenize(document.get("content", ""))
        title_overlap = len(question_tokens.intersection(title_tokens))
        content_overlap = len(question_tokens.intersection(content_tokens))
        if title_overlap + content_overlap <= 0:
            continue
        # Title matches worth 3x content matches so the dedicated document
        # for a concept outranks documents that merely mention it in passing.
        score = (title_overlap * 3 + content_overlap) / (n_q * 3)
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


def rag_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    from tools.kpi import _fetch_kpi_context_postgres
    top_k = max(1, min(_safe_int(context.get("top_k"), 4), 12))
    qdrant_hits = _qdrant_semantic_search(question, top_k)
    lexical_hits = _lexical_rag_search(question, top_k)

    # Merge both: dedup by title, keep best score from either method.
    # Lexical is precise for exact insurance terms; semantic catches paraphrases.
    seen: dict[str, dict[str, Any]] = {}
    for hit in lexical_hits:
        title = hit.get("title", "")
        if title not in seen or hit["score"] > seen[title]["score"]:
            seen[title] = hit
    for hit in qdrant_hits:
        title = hit.get("title", "")
        if title in seen:
            if hit["score"] > seen[title]["score"]:
                seen[title] = hit
        else:
            seen[title] = hit
    merged_documents: list[dict[str, Any]] = sorted(
        seen.values(), key=lambda x: x.get("score", 0.0), reverse=True
    )[:top_k]
    if not merged_documents:
        merged_documents = qdrant_hits or lexical_hits

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
