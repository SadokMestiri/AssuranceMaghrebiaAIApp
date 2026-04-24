from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent_graph


def test_is_domain_question() -> None:
    assert agent_graph.is_domain_question("donne moi les kpi impaye par branche") is True
    assert agent_graph.is_domain_question("ecris un poeme romantique") is False


def test_detect_requested_tools_multiple() -> None:
    tools = agent_graph.detect_requested_tools("fais une prevision et detecte une anomalie impaye")

    assert "forecast" in tools
    assert "anomaly" in tools


def test_detect_requested_tools_default() -> None:
    tools = agent_graph.detect_requested_tools("bonjour")

    assert tools == ["kpi"]


def test_classify_question_returns_intent_and_confidence() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question("donne une prevision prime nette")

    assert intent == "forecast"
    assert confidence > 0.5
    assert "forecast" in tools
    assert required == ["forecast"]
    assert skip_llm is False


def test_classify_question_prediction_sinistres_routes_to_forecast() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "donne moi les predictions du nombre des sinistres dans la branche auto"
    )

    assert intent == "forecast"
    assert confidence >= 0.62
    assert "forecast" in tools
    assert required == ["forecast"]
    assert skip_llm is False


def test_classify_question_prevois_with_horizon_routes_to_forecast() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "Prevois la prime nette 2026 toutes branches en te basant sur l historique 2019-2025, horizon 12 mois"
    )

    assert intent == "forecast"
    assert confidence >= 0.62
    assert tools == ["forecast"]
    assert required == ["forecast"]
    assert skip_llm is False


def test_classify_question_forecast_table_request_stays_forecast_only() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "donne un tableau de prevision des impayes pour la branche auto sur 4 mois"
    )

    assert intent == "forecast"
    assert confidence >= 0.62
    assert tools == ["forecast"]
    assert required == ["forecast"]
    assert skip_llm is False


def test_classify_question_forecast_graph_request_stays_forecast_only() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "donne un graphique de prevision des impayes pour la branche auto sur 4 mois"
    )

    assert intent == "forecast"
    assert confidence >= 0.62
    assert tools == ["forecast"]
    assert required == ["forecast"]
    assert skip_llm is False


def test_classify_question_overview_routes_to_multi_tool() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "donne moi toutes les informations assurance en vue globale et synthese complete"
    )

    assert intent == "overview"
    assert confidence >= 0.62
    assert "kpi" in tools
    assert "forecast" in tools
    assert "anomaly" in tools
    assert required == ["kpi"]
    assert skip_llm is False


def test_classify_question_graph_request_routes_to_sql() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "fais une visualisation graphique du ratio impaye par branche"
    )

    assert intent == "sql"
    assert confidence >= 0.62
    assert tools == ["sql"]
    assert required == ["sql"]
    assert skip_llm is True


def test_classify_question_sql_prime_chart_stays_sql_only() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "donne moi uniquement un graphique top branches prime nette 2025"
    )

    assert intent == "sql"
    assert confidence >= 0.62
    assert tools == ["sql"]
    assert required == ["sql"]
    assert skip_llm is True


def test_classify_question_ratio_impaye_par_branche_routes_to_sql() -> None:
    intent, confidence, tools, required, skip_llm = agent_graph.classify_question(
        "ratio impaye par branche"
    )

    assert intent == "sql"
    assert confidence >= 0.62
    assert tools == ["sql"]
    assert required == ["sql"]
    assert skip_llm is True


def test_run_agent_query_sync_blocked_for_out_of_scope() -> None:
    result = agent_graph.run_agent_query_sync("ecris une recette de gateau")

    assert result["status"] == "blocked"
    assert result["guardrails"]["domain_allowed"] is False


def test_run_agent_query_sync_uses_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": tool_name,
            "summary": f"result for {tool_name}",
            "payload": {},
        },
    )
    monkeypatch.setattr(agent_graph, "_call_ollama_chat", lambda system_prompt, user_prompt: "Synthese LLM test")

    result = agent_graph.run_agent_query_sync("donne les kpi principaux", {"branch": "AUTO"})

    assert result["status"] == "ready"
    assert "kpi_tool" in result["invoked_tools"]
    assert len(result["tool_results"]) >= 1


def test_run_agent_query_sync_returns_decision_style_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", True)
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": "kpi_tool",
            "summary": "ok",
            "payload": {
                "source": "postgres_fallback",
                "branch": "ALL",
                "year_from": 2025,
                "year_to": 2025,
                "total_pnet": 12500000.0,
                "total_commission": 3200000.0,
                "sp_ratio_proxy_pct": 1.25,
                "taux_resiliation_pct": 5.4,
            },
        },
    )
    monkeypatch.setattr(agent_graph, "_call_ollama_chat", lambda system_prompt, user_prompt: "LLM path")

    result = agent_graph.run_agent_query_sync("donne une vue kpi assurance", {})

    assert result["status"] == "ready"
    assert "**Synthese decisionnelle**" in result["answer"]


def test_force_deterministic_keeps_llm_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", True)
    monkeypatch.setattr(agent_graph, "LLM_MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": tool_name,
            "summary": "result for tool",
            "payload": {},
        },
    )
    monkeypatch.setattr(agent_graph, "_call_ollama_chat", lambda system_prompt, user_prompt: "LLM should be bypassed")

    result = agent_graph.run_agent_query_sync("donne moi les kpi impaye par branche", {})

    assert result["status"] == "ready"
    assert result["llm_used"] is False


def test_hybrid_mode_uses_llm_when_quality_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", False)
    monkeypatch.setattr(agent_graph, "LLM_MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": "kpi_tool",
            "summary": "kpi specialist result",
            "payload": {
                "source": "postgres_fallback",
                "branch": "AUTO",
                "year_from": 2025,
                "year_to": 2025,
                "total_pnet": 9800000.0,
                "total_commission": 2100000.0,
                "sp_ratio_proxy_pct": 1.9,
                "taux_resiliation_pct": 6.2,
            },
        },
    )
    monkeypatch.setattr(
        agent_graph,
        "_call_ollama_chat",
        lambda system_prompt, user_prompt: (
            "1) Synthese decisionnelle: la branche AUTO reste sous tension, "
            "mais la trajectoire est pilotable avec les KPI observes. "
            "2) Chiffres cles: prime nette 9.8M TND, resiliation 6.2%, S/P proxy 1.9%. "
            "3) Actions recommandees: renforcer retention ciblee et suivi hebdomadaire."
        ),
    )

    result = agent_graph.run_agent_query_sync("donne une synthese kpi assurance auto", {})

    assert result["status"] == "ready"
    assert result["llm_used"] is True
    assert result["synthesis_mode"] == "llm"


def test_run_agent_query_exposes_specialist_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", True)
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": tool_name,
            "summary": "specialist result",
            "payload": {"sample": 1},
        },
    )

    result = agent_graph.run_agent_query_sync("donne les kpi principaux", {"branch": "AUTO"})

    assert result["status"] == "ready"
    assert len(result["specialist_reports"]) >= 1
    assert result["specialist_reports"][0]["agent"] == "kpi_specialist"


def test_run_agent_query_infers_branch_from_question(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_contexts: list[dict[str, object]] = []

    def _fake_run_tool(tool_name: str, question: str, context: dict[str, object]) -> dict[str, object]:
        captured_contexts.append(dict(context))
        return {
            "tool": tool_name,
            "summary": "ok",
            "payload": {},
        }

    monkeypatch.setattr(agent_graph, "run_tool", _fake_run_tool)
    monkeypatch.setattr(agent_graph, "_call_ollama_chat", lambda system_prompt, user_prompt: "Synthese LLM test")

    result = agent_graph.run_agent_query_sync(
        "donne moi les predictions du nombre des sinistres dans la branche auto",
        {"year_from": 2024, "year_to": 2025},
    )

    assert result["intent"] == "forecast"
    assert len(captured_contexts) >= 1
    assert captured_contexts[0].get("branch") == "AUTO"


def test_run_agent_query_infers_year_and_all_branches_from_question(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_contexts: list[dict[str, object]] = []

    def _fake_run_tool(tool_name: str, question: str, context: dict[str, object]) -> dict[str, object]:
        captured_contexts.append(dict(context))
        return {
            "tool": tool_name,
            "summary": "ok",
            "payload": {
                "source": "postgres_fallback",
                "branch": "ALL",
                "year_from": context.get("year_from"),
                "year_to": context.get("year_to"),
                "taux_resiliation_pct": 12.34,
            },
        }

    monkeypatch.setattr(agent_graph, "run_tool", _fake_run_tool)

    result = agent_graph.run_agent_query_sync(
        "donne moi le taux de resiliation dans l'annee 2025 dans toutes les branches",
        {"branch": "AUTO", "year_from": 2024, "year_to": 2026},
    )

    assert result["intent"] == "kpi"
    assert result["invoked_tools"] == ["kpi_tool"]
    assert len(captured_contexts) == 1
    assert captured_contexts[0].get("branch") is None
    assert captured_contexts[0].get("year_from") == 2025
    assert captured_contexts[0].get("year_to") == 2025
    assert "2025" in result["answer"]
    assert "12.34%" in result["answer"]


def test_run_agent_query_infers_history_range_and_horizon_from_forecast_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_contexts: list[dict[str, object]] = []

    def _fake_run_tool(tool_name: str, question: str, context: dict[str, object]) -> dict[str, object]:
        captured_contexts.append(dict(context))
        return {
            "tool": tool_name,
            "summary": "forecast ok",
            "payload": {
                "predictions": [{"period": "2026-01", "total_pnet_pred": 1234.0}],
            },
            "charts": [
                {
                    "type": "line",
                    "title": "Forecast Prime nette",
                    "x_key": "period",
                    "y_key": "total_pnet_pred",
                    "items": [{"period": "2026-01", "total_pnet_pred": 1234.0}],
                }
            ],
            "tables": [
                {
                    "title": "Forecast",
                    "columns": ["period", "total_pnet_pred"],
                    "rows": [{"period": "2026-01", "total_pnet_pred": 1234.0}],
                }
            ],
        }

    monkeypatch.setattr(agent_graph, "run_tool", _fake_run_tool)

    result = agent_graph.run_agent_query_sync(
        "Prevois la prime nette 2026 toutes branches en te basant sur l historique 2019-2025, horizon 12 mois",
        {"branch": "AUTO", "year_from": 2024, "year_to": 2026, "horizon_months": 3},
    )

    assert result["intent"] == "forecast"
    assert result["invoked_tools"] == ["forecast_tool"]
    assert len(captured_contexts) == 1
    assert captured_contexts[0].get("branch") is None
    assert captured_contexts[0].get("year_from") == 2019
    assert captured_contexts[0].get("year_to") == 2025
    assert captured_contexts[0].get("horizon_months") == 12


def test_run_agent_query_forecast_graph_request_does_not_invoke_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": tool_name,
            "summary": "forecast graph",
            "payload": {
                "report_mode": "graph_pref",
                "target_label": "Nombre d impayes",
                "target_unit": "count",
                "trend_pct": -5.0,
                "horizon_months": 4,
                "engine": "linear_regression_fallback",
                "predictions": [
                    {"period": "2026-01", "nb_impayes_pred": 10.0},
                    {"period": "2026-02", "nb_impayes_pred": 9.0},
                ],
            },
            "charts": [
                {
                    "type": "line",
                    "title": "Forecast Nombre d impayes",
                    "x_key": "period",
                    "y_key": "nb_impayes_pred",
                    "items": [
                        {"period": "2026-01", "nb_impayes_pred": 10.0},
                        {"period": "2026-02", "nb_impayes_pred": 9.0},
                    ],
                }
            ],
            "tables": [],
        },
    )

    result = agent_graph.run_agent_query_sync(
        "donne un graphique de prevision des impayes pour la branche auto sur 4 mois",
        {"branch": "AUTO", "horizon_months": 4},
    )

    assert result["intent"] == "forecast"
    assert result["invoked_tools"] == ["forecast_tool"]
    assert result["errors"] == []


def test_low_confidence_policy_kpi_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "INTENT_MIN_CONFIDENCE", 0.95)
    monkeypatch.setattr(agent_graph, "LOW_CONFIDENCE_POLICY", "kpi_guarded")
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": tool_name,
            "summary": f"result for {tool_name}",
            "payload": {},
        },
    )

    result = agent_graph.run_agent_query_sync("assurance", {})

    assert result["status"] == "ready"
    assert result["policy_applied"] == "low_confidence_kpi_guarded"
    assert "kpi_tool" in result["invoked_tools"]
    assert result["llm_used"] is False


def test_low_confidence_policy_sql_guarded_for_retrieval_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "INTENT_MIN_CONFIDENCE", 0.95)
    monkeypatch.setattr(agent_graph, "LOW_CONFIDENCE_POLICY", "kpi_guarded")
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": tool_name,
            "summary": f"result for {tool_name}",
            "payload": {},
        },
    )

    result = agent_graph.run_agent_query_sync("ratio impaye par branche", {})

    assert result["status"] == "ready"
    assert result["policy_applied"] == "low_confidence_sql_guarded"
    assert result["invoked_tools"] == ["sql_tool"]
    assert result["llm_used"] is False


def test_low_confidence_policy_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "INTENT_MIN_CONFIDENCE", 0.95)
    monkeypatch.setattr(agent_graph, "LOW_CONFIDENCE_POLICY", "ask")

    result = agent_graph.run_agent_query_sync("assurance", {})

    assert result["status"] == "needs_clarification"
    assert result["policy_applied"] == "low_confidence_ask"
    assert result["invoked_tools"] == []


def test_run_agent_query_sql_answer_omits_technical_sql_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": "sql_tool",
            "summary": "SQL agent (overall_resiliation_rate): technical summary",
            "payload": {
                "sql_id": "overall_resiliation_rate",
                "rows": [{"total_polices": 1000, "nb_resiliees": 50, "taux_resiliation_pct": 5.0}],
                "result_kind": "scalar",
                "kpis": [
                    {"key": "taux_resiliation_pct", "label": "Taux de resiliation", "value": 5.0, "unit": "%"}
                ],
                "context": "Vue globale de la resiliation sur toutes les branches (periode 2025).",
                "analysis": "Resiliation globale: 50 polices resiliees sur 1,000, soit 5.00%.",
                "decision": "Niveau de resiliation a surveiller.",
                "actions": ["Suivre mensuellement les motifs de resiliation."],
            },
            "charts": [],
            "tables": [],
        },
    )

    result = agent_graph.run_agent_query_sync("requete sql taux de resiliation global", {})

    assert result["intent"] == "sql"
    assert "SQL agent (overall_resiliation_rate)" not in result["answer"]
    assert "Vue globale de la resiliation" in result["answer"]


def test_run_agent_query_forecast_count_uses_count_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": "forecast_tool",
            "summary": "forecast count",
            "payload": {
                "target_label": "Nombre d impayes",
                "target_unit": "count",
                "trend_pct": 12.0,
                "horizon_months": 3,
                "engine": "linear_regression_fallback",
                "predictions": [
                    {"period": "2026-01", "nb_impayes_pred": 90.0},
                    {"period": "2026-02", "nb_impayes_pred": 100.0},
                    {"period": "2026-03", "nb_impayes_pred": 110.0},
                ],
                "kpis": [
                    {"key": "projection_totale", "label": "Projection cumulee", "value": 300.0, "unit": "count"},
                    {"key": "projection_moyenne", "label": "Projection moyenne mensuelle", "value": 100.0, "unit": "count"},
                ],
                "context": "Projection Nombre d impayes sur 3 mois.",
                "analysis": "Hausse projetee du nombre d impayes.",
                "decision": "Renforcer le recouvrement.",
                "actions": ["Suivre les dossiers critiques chaque semaine."],
            },
            "charts": [],
            "tables": [],
        },
    )

    result = agent_graph.run_agent_query_sync("prevision nombre impayes", {})

    assert result["intent"] == "forecast"


def test_context_force_llm_overrides_confidence_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", False)
    monkeypatch.setattr(agent_graph, "LLM_MIN_CONFIDENCE", 0.95)
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": "kpi_tool",
            "summary": "kpi specialist result",
            "payload": {
                "source": "postgres_fallback",
                "branch": "ALL",
                "year_from": 2025,
                "year_to": 2025,
                "total_pnet": 1000000.0,
                "total_commission": 200000.0,
                "sp_ratio_proxy_pct": 1.1,
                "taux_resiliation_pct": 4.2,
            },
        },
    )
    monkeypatch.setattr(
        agent_graph,
        "_call_ollama_chat",
        lambda system_prompt, user_prompt: (
            "1) Synthese decisionnelle: la situation reste pilotable avec une attention sur la retention. "
            "2) Chiffres cles: prime nette 1.0M TND, commission 0.2M TND, resiliation 4.2%. "
            "3) Actions recommandees: suivi mensuel et plan de prevention churn."
        ),
    )

    result = agent_graph.run_agent_query_sync(
        "donne les kpi",
        {"force_llm": True},
    )

    assert result["status"] == "ready"
    assert result["policy_applied"] == "request_force_llm"
    assert result["llm_used"] is True


def test_context_skip_llm_forces_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_graph, "FORCE_DETERMINISTIC", False)
    monkeypatch.setattr(agent_graph, "LLM_MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(
        agent_graph,
        "run_tool",
        lambda tool_name, question, context: {
            "tool": "kpi_tool",
            "summary": "kpi specialist result",
            "payload": {
                "source": "postgres_fallback",
                "branch": "ALL",
                "year_from": 2025,
                "year_to": 2025,
                "total_pnet": 1000000.0,
                "total_commission": 200000.0,
                "sp_ratio_proxy_pct": 1.1,
                "taux_resiliation_pct": 4.2,
            },
        },
    )
    monkeypatch.setattr(agent_graph, "_call_ollama_chat", lambda system_prompt, user_prompt: "This should not be used")

    result = agent_graph.run_agent_query_sync(
        "donne les kpi",
        {"skip_llm": True},
    )

    assert result["status"] == "ready"
    assert result["policy_applied"] == "request_skip_llm"
    assert result["llm_used"] is False
