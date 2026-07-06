"""
Evaluation router — routing correctness suite + classifier management.

Endpoints (all under /api/v1/eval):
  GET  /classifier/status     — training report embedded in the current pkl
  POST /classifier/retrain    — retrain from synthetic dataset, hot-reload pkl
  POST /run-suite             — async smoke-test: 8 questions × (intent + tool check)
  POST /run-suite/sync        — same, blocking, returns full results immediately
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/eval", tags=["eval"])

# ── Canonical test suite ───────────────────────────────────────────────────────
# One representative question per intent. Checks that the right intent is
# classified and the expected tool is actually invoked — deterministic, no judge.
_SUITE: list[dict[str, Any]] = [
    {"intent": "kpi",          "question": "Quel est le taux de résiliation AUTO en 2024 ?",             "tool": "kpi_tool"},
    {"intent": "forecast",     "question": "Prévision de la prime nette sur 3 mois pour IRDS.",          "tool": "forecast_tool"},
    {"intent": "anomaly",      "question": "Y a-t-il des anomalies contractuelles en AUTO ?",             "tool": "anomaly_tool"},
    {"intent": "segmentation", "question": "Quels sont les segments clients et leur profil ?",            "tool": "segmentation_tool"},
    {"intent": "drift",        "question": "Détecte un drift statistique sur les données de prime.",      "tool": "drift_tool"},
    {"intent": "rag",          "question": "Comment est calculé le FGA ?",                               "tool": "rag_tool"},
    {"intent": "alerte",       "question": "Y a-t-il des alertes critiques en cours ?",                  "tool": "alerte_tool"},
    {"intent": "sql",          "question": "Top 5 gouvernorats par montant impayé en 2024.",              "tool": "sql_tool"},
]


def _run_suite_impl() -> list[dict[str, Any]]:
    from agent_graph import run_agent_query_sync

    results = []
    for case in _SUITE:
        try:
            out = run_agent_query_sync(case["question"], context={"skip_llm": True})
            intent_ok = out.get("intent") == case["intent"]
            tool_ok   = case["tool"] in out.get("invoked_tools", [])
            results.append({
                "intent_expected": case["intent"],
                "intent_got":      out.get("intent"),
                "tool_expected":   case["tool"],
                "tools_got":       out.get("invoked_tools", []),
                "intent_ok":       intent_ok,
                "tool_ok":         tool_ok,
                "pass":            intent_ok and tool_ok,
                "errors":          out.get("errors", []),
                "status":          "ok",
            })
        except Exception as exc:
            results.append({
                "intent_expected": case["intent"],
                "tool_expected":   case["tool"],
                "pass":            False,
                "status":          "error",
                "detail":          str(exc),
            })
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/run-suite")
def run_suite(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Fire-and-forget suite — results logged server-side."""
    def _task():
        _run_suite_impl()

    background_tasks.add_task(_task)
    return {
        "status":   "accepted",
        "message":  f"Suite de {len(_SUITE)} questions lancée en arrière-plan.",
        "n_cases":  len(_SUITE),
    }


@router.post("/run-suite/sync")
def run_suite_sync() -> dict[str, Any]:
    """Blocking — returns all pass/fail results immediately. Use in CI."""
    results = _run_suite_impl()
    passed  = sum(1 for r in results if r.get("pass"))
    return {
        "status":   "ok" if passed == len(_SUITE) else "warning",
        "passed":   passed,
        "total":    len(_SUITE),
        "results":  results,
    }


@router.get("/classifier/status")
def classifier_status() -> dict[str, Any]:
    """Training report embedded in the current intent_classifier.pkl."""
    from intent_classifier import is_available, model_report
    available = is_available()
    return {
        "status":    "ok",
        "available": available,
        "report":    model_report() if available else None,
    }


@router.post("/classifier/retrain")
def retrain_classifier(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Retrain from synthetic dataset and hot-reload without server restart."""
    def _task():
        try:
            from train_intent_classifier import train
            train(extra_data=None, dry_run=False)
            from intent_classifier import reload
            reload()
        except Exception as exc:
            import logging
            logging.getLogger("maghrebia.eval").error("Retrain failed: %s", exc)

    background_tasks.add_task(_task)
    return {
        "status":  "accepted",
        "message": "Retraining started. Check GET /eval/classifier/status once done.",
    }
