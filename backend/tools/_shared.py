from __future__ import annotations

from typing import Any

from utils import safe_int as _safe_int
from config import DATA_YEAR_FROM, DATA_YEAR_TO

VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}


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


def _to_markdown_table(columns: list[str], rows: list[dict[str, Any]], max_rows: int = 8) -> str:
    if not columns or not rows:
        return ""

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:max_rows]:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_chart_payload(chart_type: str, title: str, x_key: str, y_key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": chart_type,
        "title": title,
        "x_key": x_key,
        "y_key": y_key,
        "items": items,
    }
