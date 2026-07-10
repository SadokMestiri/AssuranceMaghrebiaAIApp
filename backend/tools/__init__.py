"""
tools package — one module per specialist tool.
Exports TOOL_REGISTRY and run_tool for use by agent_graph / agent_tools.
"""
from __future__ import annotations

from typing import Any

from tools.kpi import kpi_tool
from tools.rag import rag_tool
from tools.alerte import alerte_tool
from tools.forecast import forecast_tool
from tools.anomaly import anomaly_tool
from tools.drift import drift_tool
from tools.explain import explain_tool
from tools.segmentation import segmentation_tool
from tools.client import client_tool
from tools.sql import sql_tool, data_query_tool
from tools.ml_predict import ml_predict_tool
from tools.dim import dim_tool
from tools.comparison import comparison_tool

TOOL_REGISTRY: dict[str, Any] = {
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
    "comparison_tool": comparison_tool,
    # Compatibility aliases
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
