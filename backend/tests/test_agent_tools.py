from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest  # type: ignore[reportMissingImports]

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent_tools
import tools.forecast as tools_forecast
import tools.sql as tools_sql


# ── Patch helpers ────────────────────────────────────────────────────────────
# The tool logic lives in the tools/ package (agent_tools is a thin shim), so the
# DB accessor must be patched in the module that actually owns it. For forecast
# we also neutralise the pre-trained prophet_service fast-path and the model
# import so the deterministic DWH → linear-regression fallback is exercised
# (no database, no pickle → CI-safe).
def _raise_import_error(name: str):
    raise ImportError(name)


def _patch_forecast(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> None:
    monkeypatch.setattr(tools_forecast, "_query_dataframe", lambda q, params=None: df)
    monkeypatch.setattr(tools_forecast, "_try_prophet_service", lambda *a, **k: None)
    monkeypatch.setattr(tools_forecast.importlib, "import_module", _raise_import_error)


def _patch_sql(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> None:
    monkeypatch.setattr(tools_sql, "_query_dataframe", lambda q, params=None: df)


# ── Forecast tool ────────────────────────────────────────────────────────────
def test_forecast_tool_sinistre_target(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = pd.date_range("2024-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "annee_echeance": [period.year for period in periods],
            "mois_echeance": [period.month for period in periods],
            "period": periods,
            "metric_value": [10 + i for i in range(12)],
        }
    )
    _patch_forecast(monkeypatch, df)

    result = agent_tools.forecast_tool(
        question="donne moi les predictions du nombre des sinistres dans la branche auto",
        context={"branch": "AUTO", "year_from": 2024, "year_to": 2025, "horizon_months": 3},
    )

    assert result["tool"] == "forecast_tool"
    assert result["payload"]["target_metric"] == "nb_sinistres"
    assert result["payload"]["target_unit"] == "count"
    assert result["payload"]["report_mode"] == "report"
    assert result["payload"]["result_kind"] == "timeseries"
    assert len(result["payload"].get("history", [])) == 12
    assert len(result["payload"].get("kpis", [])) >= 2
    assert result["payload"].get("decision")
    assert isinstance(result["payload"].get("actions"), list)
    assert result["charts"][0]["y_key"] == "combined_value"
    assert len(result["charts"][0]["items"]) == 15
    assert result["charts"][0]["series"][0]["key"] == "actual"
    assert result["charts"][0]["series"][1]["key"] == "nb_sinistres_pred"
    assert result["charts"][0]["series"][1]["color"] == "#dc2626"
    assert result["charts"][0]["series"][1]["strokeDasharray"] == "8 5"
    assert result["charts"][0]["forecast_start_period"] == "2025-01"
    assert result["charts"][0]["items"][0]["actual"] == pytest.approx(10.0)
    assert result["charts"][0]["items"][0]["combined_value"] == pytest.approx(10.0)
    assert result["charts"][0]["items"][-1]["actual"] is None
    assert result["charts"][0]["items"][-1]["combined_value"] == pytest.approx(24.0)
    assert len(result["tables"]) == 1
    assert len(result["payload"]["predictions"]) == 3


def test_forecast_tool_table_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = pd.date_range("2024-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "annee_echeance": [period.year for period in periods],
            "mois_echeance": [period.month for period in periods],
            "period": periods,
            "metric_value": [100 + i for i in range(12)],
        }
    )
    _patch_forecast(monkeypatch, df)

    result = agent_tools.forecast_tool(
        question="table only prevision prime nette",
        context={"year_from": 2024, "year_to": 2025, "horizon_months": 4},
    )

    assert result["tool"] == "forecast_tool"
    assert result["payload"]["target_metric"] == "total_pnet"
    assert result["payload"]["report_mode"] == "table_only"
    assert result["charts"] == []
    assert len(result["tables"]) == 1
    assert len(result["payload"]["predictions"]) == 4


def test_forecast_tool_graph_pref_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = pd.date_range("2024-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "annee_echeance": [period.year for period in periods],
            "mois_echeance": [period.month for period in periods],
            "period": periods,
            "metric_value": [80 + i for i in range(12)],
        }
    )
    _patch_forecast(monkeypatch, df)

    result = agent_tools.forecast_tool(
        question="donne un graphique de prevision des impayes",
        context={"year_from": 2024, "year_to": 2025, "horizon_months": 4},
    )

    assert result["tool"] == "forecast_tool"
    assert result["payload"]["target_metric"] == "nb_impayes"
    assert result["payload"]["report_mode"] == "graph_pref"
    assert len(result["charts"]) == 1
    assert result["tables"] == []


def test_forecast_tool_insufficient_data_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = pd.date_range("2025-01-01", periods=4, freq="MS")
    df = pd.DataFrame(
        {
            "annee_echeance": [period.year for period in periods],
            "mois_echeance": [period.month for period in periods],
            "period": periods,
            "metric_value": [12, 11, 10, 9],
        }
    )
    _patch_forecast(monkeypatch, df)

    result = agent_tools.forecast_tool(
        question="prevision nombre impayes",
        context={"year_from": 2025, "year_to": 2025, "horizon_months": 3},
    )

    assert result["tool"] == "forecast_tool"
    assert result["payload"]["predictions"] == []
    assert result["payload"]["target_unit"] == "count"
    assert result["payload"]["report_mode"] == "report"
    assert result["payload"].get("decision")
    assert isinstance(result["payload"].get("actions"), list)
    assert result["charts"] == []
    assert result["tables"] == []


# ── SQL tool ─────────────────────────────────────────────────────────────────
def test_sql_tool_graph_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "branche": ["AUTO", "SANTE", "IRDS"],
            "total_pnet": [1000.0, 700.0, 200.0],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="donne moi uniquement un graphique top branches prime nette",
        context={"year_from": 2024, "year_to": 2025},
    )

    assert result["tool"] == "sql_tool"
    assert result["payload"]["report_mode"] == "graph_only"
    assert result["payload"]["sql_id"] == "prime_by_branch"
    assert len(result["charts"]) == 1
    assert result["charts"][0]["title"] == "Part de production par branche"
    assert result["tables"] == []


def test_sql_tool_impaye_ratio_includes_decision_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "branche": ["AUTO", "SANTE"],
            "total_pnet": [10000.0, 8000.0],
            "total_impaye": [900.0, 240.0],
            "impaye_ratio_pct": [9.0, 3.0],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="donne le ratio impaye par branche",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["tool"] == "sql_tool"
    assert result["payload"]["sql_id"] == "impaye_rate_by_branch"
    assert result["payload"]["result_kind"] == "breakdown"
    assert "decision" in result["payload"]
    assert result["payload"]["decision"]
    assert isinstance(result["payload"].get("actions"), list)
    assert len(result["payload"]["actions"]) >= 1
    assert len(result["tables"]) == 1
    assert len(result["charts"]) == 1


def test_sql_tool_total_impayes_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "nb_impayes": [321],
            "total_impaye": [198765.43],
            "nb_polices_impactees": [205],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="donne moi le nombre total d impaye",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["tool"] == "sql_tool"
    assert result["payload"]["sql_id"] == "impaye_overview"
    assert result["payload"]["result_kind"] == "scalar"
    assert result["payload"]["rows"][0]["nb_impayes"] == 321
    assert result["charts"] == []
    assert result["tables"] == []
    assert len(result["payload"].get("kpis", [])) >= 3
    assert "decision" in result["payload"]
    assert result["payload"]["decision"]


def test_sql_tool_total_impayes_with_accented_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "nb_impayes": [87],
            "total_impaye": [76543.21],
            "nb_polices_impactees": [72],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="donne moi le nombre total d impayé",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["payload"]["sql_id"] == "impaye_overview"
    assert result["payload"]["rows"][0]["nb_impayes"] == 87


def test_sql_tool_avg_impaye_by_branche_semantic_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "branche": ["AUTO", "SANTE"],
            "avg_impaye": [950.0, 420.0],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="donne la moyenne impaye par branche",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["payload"]["sql_id"] == "impaye_rate_by_branch"
    assert result["payload"]["result_kind"] == "breakdown"
    assert len(result["charts"]) == 1


def test_sql_tool_scalar_graph_only_still_returns_kpi(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "nb_impayes": [200],
            "total_impaye": [150000.0],
            "avg_impaye": [750.0],
            "nb_polices_impactees": [130],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="graph only nombre total d impaye",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["payload"]["report_mode"] == "graph_only"
    assert result["payload"]["result_kind"] == "scalar"
    assert result["charts"] == []
    assert len(result["payload"].get("kpis", [])) >= 1


def test_sql_tool_top_n_clients_impaye_semantic_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "id_client": [101, 202],
            "nom": ["A", "B"],
            "prenom": ["X", "Y"],
            "total_impaye": [12000.0, 8000.0],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="top 5 clients impayes",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["payload"]["sql_id"] == "impaye_top_clients"
    assert len(result["charts"]) == 1
    assert result["charts"][0]["y_key"] == "total_impaye"


def test_sql_tool_resiliation_global_returns_scalar_kpi(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "total_polices": [5000],
            "nb_resiliees": [250],
            "taux_resiliation_pct": [5.0],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="requete sql taux de resiliation global",
        context={"year_from": 2025, "year_to": 2025},
    )

    assert result["payload"]["sql_id"] == "resiliation_overview"
    assert result["payload"]["result_kind"] == "scalar"
    assert result["charts"] == []
    assert len(result["payload"].get("kpis", [])) >= 1


def test_sql_tool_top_zones_risque_maps_to_impaye_gouvernorat(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "gouvernorat": ["SFAX", "TUNIS", "SOUSSE"],
            "total_impaye": [55620.0, 43210.0, 30100.0],
        }
    )
    _patch_sql(monkeypatch, df)

    result = agent_tools.sql_tool(
        question="donne moi top 3 zones risque",
        context={"year_from": 2025, "year_to": 2026},
    )

    assert result["payload"]["sql_id"] == "top_zones_risque"
    assert result["payload"]["result_kind"] == "breakdown"
    assert len(result["charts"]) == 1
    assert result["charts"][0]["x_key"] == "gouvernorat"
