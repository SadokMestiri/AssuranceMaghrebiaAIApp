from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent_graph import get_agent_capabilities, run_agent_query_sync
from indexer import run_indexing


router = APIRouter(prefix="/agent", tags=["agent"])
YEAR_MIN = 2019
YEAR_MAX = 2026


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    branch: str | None = None
    year_from: int | None = Field(default=None, ge=YEAR_MIN, le=YEAR_MAX)
    year_to: int | None = Field(default=None, ge=YEAR_MIN, le=YEAR_MAX)
    month: str | None = None
    gouvernorat: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)
    horizon_months: int = Field(default=3, ge=1, le=12)
    client_name: str | None = None
    skip_llm: bool = False
    force_llm: bool = False


class AgentIndexRequest(BaseModel):
    max_docs_per_collection: int = Field(default=400, ge=20, le=5000)


class AgentWarmupRequest(BaseModel):
    preindex: bool = False
    max_docs_per_collection: int = Field(default=250, ge=20, le=5000)
    strict: bool = False


def _check_http_endpoint(url: str, timeout_seconds: float = 2.5) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout_seconds)
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def perform_agent_warmup(
    *,
    preindex: bool,
    max_docs_per_collection: int,
    strict: bool,
) -> dict[str, Any]:
    status_payload = get_agent_status()
    dependencies = status_payload.get("dependencies", {})
    ollama_ok = bool((dependencies.get("ollama") or {}).get("ok"))
    qdrant_ok = bool((dependencies.get("qdrant") or {}).get("ok"))

    preindex_report: dict[str, Any] = {
        "requested": preindex,
        "executed": False,
        "reason": "not_requested",
    }

    errors: list[str] = []

    if preindex:
        if not qdrant_ok:
            preindex_report = {
                "requested": True,
                "executed": False,
                "reason": "qdrant_unavailable",
            }
            errors.append("preindex skipped: qdrant unavailable")
        else:
            try:
                index_report = run_indexing(max_docs_per_collection=max_docs_per_collection)
                preindex_report = {
                    "requested": True,
                    "executed": True,
                    "reason": "ok",
                    "indexing": index_report,
                }
            except Exception as exc:
                preindex_report = {
                    "requested": True,
                    "executed": False,
                    "reason": "indexing_error",
                    "error": str(exc),
                }
                errors.append(f"indexing error: {exc}")

    warmup_ok = qdrant_ok and (ollama_ok or not strict)
    status = "ok" if warmup_ok and not errors else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strict": strict,
        "dependencies": dependencies,
        "preindex": preindex_report,
        "errors": errors,
    }


@router.get("/capabilities")
def get_capabilities() -> dict[str, Any]:
    return {
        "status": "ready",
        "agent": get_agent_capabilities(),
    }


@router.get("/status")
def get_agent_status() -> dict[str, Any]:
    ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")

    ollama = _check_http_endpoint(f"{ollama_host}/api/tags")
    qdrant = _check_http_endpoint(f"{qdrant_url}/collections")

    ready = bool(ollama.get("ok")) and bool(qdrant.get("ok"))
    return {
        "status": "ready" if ready else "degraded",
        "dependencies": {
            "ollama": ollama,
            "qdrant": qdrant,
        },
    }


@router.get("/eval/smoke")
def run_agent_smoke_eval() -> dict[str, Any]:
    scenarios = [
        {
            "name": "forecast_prime_auto",
            "question": "Donne une prevision de prime pour la branche AUTO",
            "context": {"branch": "AUTO", "horizon_months": 3},
            "expected_intent": "forecast",
            "expected_tool": "forecast_tool",
        },
        {
            "name": "top_gouvernorat_impaye",
            "question": "Donne le top gouvernorat par impaye",
            "context": {"top_k": 3},
            "expected_intent": "sql",
            "expected_tool": "sql_tool",
        },
        {
            "name": "client_profile",
            "question": "Analyse le client Ahmed",
            "context": {"client_name": "Ahmed"},
            "expected_intent": "client",
            "expected_tool": "client_tool",
        },
    ]

    results = []
    passes = 0
    for scenario in scenarios:
        try:
            outcome = run_agent_query_sync(scenario["question"], context=scenario["context"])
            intent_ok = outcome.get("intent") == scenario["expected_intent"]
            tool_ok = scenario["expected_tool"] in outcome.get("invoked_tools", [])
            scenario_passed = intent_ok and tool_ok
            if scenario_passed:
                passes += 1
            results.append(
                {
                    "name": scenario["name"],
                    "pass": scenario_passed,
                    "intent": outcome.get("intent"),
                    "invoked_tools": outcome.get("invoked_tools", []),
                    "llm_used": outcome.get("llm_used", False),
                    "errors": outcome.get("errors", []),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": scenario["name"],
                    "pass": False,
                    "error": str(exc),
                }
            )

    total = len(scenarios)
    status = "ok" if passes == total else "warning"
    return {
        "status": status,
        "passed": passes,
        "total": total,
        "results": results,
    }


@router.post("/index/rebuild")
def rebuild_agent_index(payload: AgentIndexRequest) -> dict[str, Any]:
    try:
        report = run_indexing(max_docs_per_collection=payload.max_docs_per_collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}")

    return {
        "status": "ok",
        "indexing": report,
    }


@router.post("/warmup")
def warmup_agent(payload: AgentWarmupRequest) -> dict[str, Any]:
    report = perform_agent_warmup(
        preindex=payload.preindex,
        max_docs_per_collection=payload.max_docs_per_collection,
        strict=payload.strict,
    )
    return {
        "status": report.get("status", "degraded"),
        "warmup": report,
    }


@router.post("/query")
def query_agent(payload: AgentQueryRequest) -> dict[str, Any]:
    if payload.year_from and payload.year_to and payload.year_from > payload.year_to:
        raise HTTPException(status_code=400, detail="year_from must be <= year_to")

    context = {
        "branch": payload.branch,
        "year_from": payload.year_from,
        "year_to": payload.year_to,
        "month": payload.month,
        "gouvernorat": payload.gouvernorat,
        "top_k": payload.top_k,
        "horizon_months": payload.horizon_months,
        "client_name": payload.client_name,
        "skip_llm": payload.skip_llm,
        "force_llm": payload.force_llm,
    }

    try:
        result = run_agent_query_sync(question=payload.question, context=context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent query failed: {exc}")

    return {
        "status": "ok",
        "agent": result,
    }


@router.post("/chat")
def chat_agent(payload: AgentQueryRequest) -> dict[str, Any]:
    return query_agent(payload)
