from __future__ import annotations

import re
from typing import Any

from utils import normalize_text as _normalize_text, safe_float as _safe_float
from config import DATA_YEAR_FROM, DATA_YEAR_TO
from tools._shared import _normalize_branch, _build_chart_payload
from tools.kpi import _fetch_kpi_context_postgres, _detect_kpi_focus

_BRANCHES = ("AUTO", "IRDS", "SANTE")

# Maps a kpi_tool "focus" to the single field to diff between the two periods.
_COMPARISON_METRICS: dict[str, tuple[str, str, str]] = {
    "ratio_combine": ("ratio_combine_pct", "Ratio Combiné", "%"),
    "resiliation":   ("taux_resiliation_pct", "Taux de résiliation", "%"),
    "sinistre":      ("nb_sinistres", "Nb sinistres", "count"),
    "impaye":        ("nb_impayes", "Nb impayés", "count"),
    "prime":         ("total_pnet", "Prime nette", "TND"),
    "overview":      ("total_pnet", "Prime nette", "TND"),
}


def _extract_years(question: str) -> list[int]:
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", question)]
    return sorted({y for y in years if DATA_YEAR_FROM <= y <= DATA_YEAR_TO})


def _extract_branches(question: str) -> list[str]:
    normalized = _normalize_text(question).upper()
    return [b for b in _BRANCHES if b in normalized]


def comparison_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    years = _extract_years(question)
    branches = _extract_branches(question)
    focus = _detect_kpi_focus(question)
    metric_key, metric_label, metric_unit = _COMPARISON_METRICS.get(focus, _COMPARISON_METRICS["overview"])
    base_branch = _normalize_branch(context.get("branch"))

    if len(years) >= 2:
        # "compare 2023 vs 2024 ratio combine AUTO" — same branch, two years.
        year_a, year_b = years[0], years[1]
        label_a, label_b = str(year_a), str(year_b)
        ctx_a = {**context, "branch": base_branch, "year_from": year_a, "year_to": year_a}
        ctx_b = {**context, "branch": base_branch, "year_from": year_b, "year_to": year_b}
        scope_label = f"branche {base_branch or 'ALL'}"
    elif len(branches) >= 2:
        # "compare AUTO vs IRDS" — same period, two branches.
        branch_a, branch_b = branches[0], branches[1]
        label_a, label_b = branch_a, branch_b
        year_to = context.get("year_to") or DATA_YEAR_TO
        year_from = context.get("year_from") or year_to
        ctx_a = {**context, "branch": branch_a, "year_from": year_from, "year_to": year_to}
        ctx_b = {**context, "branch": branch_b, "year_from": year_from, "year_to": year_to}
        scope_label = f"periode {year_from}-{year_to}"
    else:
        # No explicit second period/branch named — default to "this year vs last".
        year_to = context.get("year_to") or DATA_YEAR_TO
        year_from_selected = context.get("year_from") or year_to
        year_a, year_b = year_from_selected - 1, year_to
        label_a, label_b = str(year_a), str(year_b)
        ctx_a = {**context, "branch": base_branch, "year_from": year_a, "year_to": year_a}
        ctx_b = {**context, "branch": base_branch, "year_from": year_b, "year_to": year_b}
        scope_label = f"branche {base_branch or 'ALL'}"

    payload_a = _fetch_kpi_context_postgres(ctx_a)
    payload_b = _fetch_kpi_context_postgres(ctx_b)

    value_a = _safe_float(payload_a.get(metric_key), 0.0)
    value_b = _safe_float(payload_b.get(metric_key), 0.0)
    delta = value_b - value_a
    delta_pct = (100.0 * delta / abs(value_a)) if value_a else None
    direction = "hausse" if delta > 0 else "baisse" if delta < 0 else "stable"
    delta_pct_text = f"{delta_pct:+.1f}%" if delta_pct is not None else "n/d"

    unit_suffix = f" {metric_unit}" if metric_unit not in {"%", ""} else metric_unit

    summary = (
        f"{metric_label} — {label_a}: {value_a:,.2f}{unit_suffix} vs {label_b}: {value_b:,.2f}{unit_suffix} "
        f"({scope_label}). Variation: {delta:+,.2f}{unit_suffix} ({delta_pct_text}, {direction})."
    )

    comparison_rows = [
        {"periode": label_a, metric_key: value_a},
        {"periode": label_b, metric_key: value_b},
    ]

    return {
        "tool": "comparison_tool",
        "summary": summary,
        "payload": {
            "metric": metric_key,
            "metric_label": metric_label,
            "unit": metric_unit,
            "label_a": label_a,
            "label_b": label_b,
            "value_a": value_a,
            "value_b": value_b,
            "delta": delta,
            "delta_pct": delta_pct,
            "direction": direction,
            "context": f"Comparaison {metric_label} entre {label_a} et {label_b} ({scope_label}).",
            "decision": (
                f"{metric_label} en {direction} de {abs(delta):,.2f}{unit_suffix} "
                f"({delta_pct_text}) entre {label_a} et {label_b}."
            ),
            "actions": (
                [f"Investiguer les causes de cette {direction} — variation superieure a 5%."]
                if delta_pct is not None and abs(delta_pct) > 5
                else []
            ),
            "kpis": [
                {"key": f"{metric_key}_a", "label": f"{metric_label} {label_a}", "value": value_a, "unit": metric_unit},
                {"key": f"{metric_key}_b", "label": f"{metric_label} {label_b}", "value": value_b, "unit": metric_unit},
                {"key": "delta", "label": "Variation", "value": delta, "unit": metric_unit},
            ],
        },
        "charts": [
            _build_chart_payload("bar", f"{metric_label}: {label_a} vs {label_b}", "periode", metric_key, comparison_rows)
        ],
        "tables": [{
            "title": f"Comparaison {metric_label}",
            "columns": ["periode", metric_key],
            "rows": comparison_rows,
        }],
    }
