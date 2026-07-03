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
