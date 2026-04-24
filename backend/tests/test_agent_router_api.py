from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent_router


def test_get_capabilities_returns_ready() -> None:
    response = agent_router.get_capabilities()

    assert response["status"] == "ready"
    assert "tools" in response["agent"]
    assert len(response["agent"]["tools"]) >= 7


def test_query_agent_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_router,
        "run_agent_query_sync",
        lambda question, context: {
            "status": "ok",
            "question": question,
            "invoked_tools": ["kpi_tool", "rag_tool"],
            "answer": "Synthese agent IA",
            "tool_results": [],
            "errors": [],
            "guardrails": {"domain_ok": True},
        },
    )

    payload = agent_router.AgentQueryRequest(
        question="donne les kpi principaux",
        branch="AUTO",
        year_from=2021,
        year_to=2025,
        top_k=3,
    )

    response = agent_router.query_agent(payload)

    assert response["status"] == "ok"
    assert response["agent"]["status"] == "ok"
    assert response["agent"]["invoked_tools"] == ["kpi_tool", "rag_tool"]


def test_query_agent_validates_year_range() -> None:
    payload = agent_router.AgentQueryRequest(
        question="test",
        year_from=2025,
        year_to=2024,
    )

    with pytest.raises(HTTPException) as exc_info:
        agent_router.query_agent(payload)

    assert exc_info.value.status_code == 400


def test_query_agent_maps_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_error(question: str, context: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("agent crashed")

    monkeypatch.setattr(agent_router, "run_agent_query_sync", _raise_error)

    payload = agent_router.AgentQueryRequest(question="donne les kpi")

    with pytest.raises(HTTPException) as exc_info:
        agent_router.query_agent(payload)

    assert exc_info.value.status_code == 500


def test_chat_agent_alias_uses_query_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_router,
        "run_agent_query_sync",
        lambda question, context: {
            "status": "ok",
            "question": question,
            "invoked_tools": ["kpi_tool"],
            "answer": "Synthese agent IA",
            "tool_results": [],
            "errors": [],
            "guardrails": {"domain_ok": True},
        },
    )

    payload = agent_router.AgentQueryRequest(question="donne les kpi")
    response = agent_router.chat_agent(payload)

    assert response["status"] == "ok"
    assert response["agent"]["status"] == "ok"


def test_get_agent_status_reports_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_router,
        "_check_http_endpoint",
        lambda url, timeout_seconds=2.5: {"ok": False, "error": "down"},
    )

    response = agent_router.get_agent_status()

    assert response["status"] == "degraded"
    assert response["dependencies"]["ollama"]["ok"] is False
    assert response["dependencies"]["qdrant"]["ok"] is False


def test_run_agent_smoke_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_runner(question: str, context: dict[str, object]) -> dict[str, object]:
        if "prevision" in question:
            return {"intent": "forecast", "invoked_tools": ["forecast_tool"], "llm_used": False, "errors": []}
        if "gouvernorat" in question:
            return {"intent": "sql", "invoked_tools": ["sql_tool"], "llm_used": False, "errors": []}
        return {"intent": "client", "invoked_tools": ["client_tool"], "llm_used": False, "errors": []}

    monkeypatch.setattr(agent_router, "run_agent_query_sync", _mock_runner)

    response = agent_router.run_agent_smoke_eval()

    assert response["status"] == "ok"
    assert response["passed"] == response["total"]


def test_rebuild_agent_index_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_router,
        "run_indexing",
        lambda max_docs_per_collection=400: {
            "status": "ok",
            "collections": {"business_rules": {"indexed": 12}},
        },
    )

    payload = agent_router.AgentIndexRequest(max_docs_per_collection=200)
    response = agent_router.rebuild_agent_index(payload)

    assert response["status"] == "ok"
    assert response["indexing"]["status"] == "ok"


def test_warmup_agent_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_router,
        "perform_agent_warmup",
        lambda preindex, max_docs_per_collection, strict: {
            "status": "ok",
            "strict": strict,
            "dependencies": {
                "ollama": {"ok": True},
                "qdrant": {"ok": True},
            },
            "preindex": {
                "requested": preindex,
                "executed": preindex,
                "reason": "ok",
            },
            "errors": [],
        },
    )

    payload = agent_router.AgentWarmupRequest(preindex=True, max_docs_per_collection=120, strict=False)
    response = agent_router.warmup_agent(payload)

    assert response["status"] == "ok"
    assert response["warmup"]["preindex"]["requested"] is True
