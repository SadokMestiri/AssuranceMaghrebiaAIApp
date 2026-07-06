from __future__ import annotations

from typing import Any

from tools._shared import _normalize_branch, _build_chart_payload, _to_markdown_table


def drift_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Delegates to ml_services.drift_service — Evidently/KS/Chi² + PSI on 7 features
    with a proper time-based split (reference=12m, current=6m), same as the MLOps tab.
    The previous implementation ran KS on 4 monthly aggregates with a naive half/half split,
    which diverged from service results and missed categorical features (BRANCHE, PERIODICITE).
    """
    from ml_services.drift_service import detect_drift

    branch = _normalize_branch(context.get("branch"))

    try:
        result = detect_drift(departement=branch or None)
    except Exception as exc:
        return {
            "tool": "drift_tool",
            "summary": f"Erreur detection drift: {exc}",
            "payload": {"metrics": [], "engine": "error"},
        }

    if "error" in result:
        return {
            "tool": "drift_tool",
            "summary": result["error"],
            "payload": {"metrics": [], "engine": "unavailable"},
        }

    niveau = result.get("niveau", "normal")
    nb_drifted = result.get("nb_drifted", 0)
    nb_features = result.get("nb_features", 0)
    message = result.get("message", "")

    # PSI table: highest-PSI features first
    psi_rows = result.get("psi_features", [])
    psi_cols = ["feature", "psi", "niveau"]

    # KS/Chi² drift table
    drift_rows = [
        {
            "feature":   d.get("feature", ""),
            "drift":     "oui" if d.get("drift_detecte") else "non",
            "p_value":   d.get("p_value", 1.0),
            "methode":   d.get("methode", ""),
        }
        for d in result.get("features", [])
    ]
    drift_cols = ["feature", "drift", "p_value", "methode"]

    # Mean comparison chart (numerical features)
    comparison = result.get("comparaison", [])
    chart_items = [
        {"feature": c["feature"], "variation_pct": c["variation_pct"]}
        for c in comparison
    ]

    summary = f"Drift {niveau}: {message} (ref={result.get('nb_mois_reference', 12)}m, courant={result.get('nb_mois_courant', 6)}m)."

    return {
        "tool": "drift_tool",
        "summary": summary,
        "payload": {
            "branch":          branch or "ALL",
            "engine":          "evidently_or_scipy",
            "niveau":          niveau,
            "dataset_drift":   result.get("dataset_drift", False),
            "nb_drifted":      nb_drifted,
            "nb_features":     nb_features,
            "share_drift":     result.get("share_drift", 0.0),
            "date_ref_debut":  result.get("date_ref_debut"),
            "date_cur_debut":  result.get("date_cur_debut"),
            "features":        result.get("features", []),
            "psi_features":    psi_rows,
            "comparaison":     comparison,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Variation moyenne (%) reference → courant",
                x_key="feature",
                y_key="variation_pct",
                items=chart_items,
            )
        ] if chart_items else [],
        "tables": [
            {
                "title": "PSI par feature (stabilite distribution)",
                "columns": psi_cols,
                "rows": psi_rows,
                "markdown": _to_markdown_table(psi_cols, psi_rows),
            },
            {
                "title": "Drift statistique (KS / Chi²)",
                "columns": drift_cols,
                "rows": drift_rows,
                "markdown": _to_markdown_table(drift_cols, drift_rows),
            },
        ] if psi_rows or drift_rows else [],
    }
