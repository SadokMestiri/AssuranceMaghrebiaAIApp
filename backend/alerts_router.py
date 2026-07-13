"""
Standalone alert route — the same detection logic as agent_tools' alerte_tool,
but reachable without going through the agent/intent pipeline, so the
dashboard sidebar can poll it directly and cheaply.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from tools._shared import _normalize_branch
from tools.alerte import compute_alerts

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def get_alerts(
    branch: str | None = Query(default=None),
    months: int = Query(default=12, ge=3, le=36),
) -> dict[str, Any]:
    result = compute_alerts(_normalize_branch(branch), months=months)
    return {
        "status": "ok",
        **result,
        "count": len(result["alerts"]),
    }
