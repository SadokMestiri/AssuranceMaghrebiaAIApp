"""
Shared utility functions used across agent_tools, agent_graph, and indexer.
Import directly from here instead of duplicating logic.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_text(text_value: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Handles smart apostrophes."""
    text_value = text_value.replace("’", "'")
    ascii_normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", text_value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", ascii_normalized.strip().lower())


def format_amount_tnd(value: float) -> str:
    return f"{value:,.0f} TND"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def format_metric_value(value: float, unit: str) -> str:
    normalized_unit = unit.strip().upper()
    if normalized_unit == "TND":
        return format_amount_tnd(value)
    if normalized_unit == "%":
        return format_percent(value)
    if normalized_unit == "COUNT":
        return f"{int(round(value)):,.0f}"
    return f"{value:,.2f}"


def format_branch_label(branch_value: Any) -> str:
    branch = str(branch_value or "ALL").upper()
    return "toutes les branches" if branch == "ALL" else f"la branche {branch}"


# Report-mode markers shared by forecast_tool and sql_tool.
_GRAPH_ONLY_MARKERS = [
    "graphique uniquement", "graphe uniquement",
    "uniquement un graphique", "uniquement un graphe",
    "juste un graphe", "seulement un graphe", "seulement un graphique",
    "only graph", "graph only", "sans table", "without table", "data viz only",
]
_TABLE_ONLY_MARKERS = [
    "table uniquement", "tableau uniquement", "juste la table",
    "only table", "table only", "sans graphique", "without graph", "without chart",
]
_GRAPH_MARKERS = ["graph", "graphe", "graphique", "chart", "plot", "visual", "courbe", "diagramme"]
_TABLE_MARKERS = ["table", "tableau", "tabulaire", "lignes", "rows"]


def infer_report_mode(question: str) -> str:
    """Return 'graph_only' | 'table_only' | 'graph_pref' | 'table_pref' | 'report'."""
    lowered = normalize_text(question)

    def _any(markers: list[str]) -> bool:
        return any(m in lowered for m in markers)

    if _any(_GRAPH_ONLY_MARKERS):
        return "graph_only"
    if _any(_TABLE_ONLY_MARKERS):
        return "table_only"
    graph_req = _any(_GRAPH_MARKERS)
    table_req = _any(_TABLE_MARKERS)
    if graph_req and not table_req:
        return "graph_pref"
    if table_req and not graph_req:
        return "table_pref"
    return "report"
