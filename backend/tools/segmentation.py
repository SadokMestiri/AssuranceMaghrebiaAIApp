from __future__ import annotations

from typing import Any

from tools._shared import _build_chart_payload, _to_markdown_table


def segmentation_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Delegates to ml_services.segmentation_service — RFM + K-Means k=5 on 15 features
    with RobustScaler and named segments, same model used by the MLOps tab.
    The previous implementation ran K-Means k=3 on 3 features without caching,
    producing different cluster assignments than the MLOps tab.
    """
    from ml_services.segmentation_service import get_segmentation_summary

    try:
        result = get_segmentation_summary()
    except Exception as exc:
        return {
            "tool": "segmentation_tool",
            "summary": f"Erreur segmentation: {exc}",
            "payload": {"segments": [], "engine": "kmeans_rfm"},
        }

    segments = result.get("segments", [])
    nb_clients = result.get("nb_clients", 0)
    nb_clusters = result.get("nb_clusters", 5)
    silhouette = result.get("silhouette")

    sil_str = f", silhouette={silhouette:.3f}" if silhouette else ""
    summary = (
        f"Segmentation RFM + K-Means k={nb_clusters}: {nb_clients} clients "
        f"classes en {len(segments)} segments{sil_str}."
    )

    table_rows = [
        {
            "segment":     s.get("name", "—"),
            "nb_clients":  s.get("count", 0),
            "part_%":      s.get("share_pct", 0),
            "prime_moy":   s.get("avg_prime", 0),
            "taux_impaye": s.get("avg_taux_impaye", 0),
            "action":      s.get("action", "—"),
        }
        for s in segments
    ]
    cols = ["segment", "nb_clients", "part_%", "prime_moy", "taux_impaye", "action"]

    chart_items = [
        {"segment": s.get("name", "—"), "nb_clients": s.get("count", 0)}
        for s in segments
    ]

    return {
        "tool": "segmentation_tool",
        "summary": summary,
        "payload": {
            "engine":       "kmeans_rfm",
            "nb_clients":   nb_clients,
            "nb_clusters":  nb_clusters,
            "silhouette":   silhouette,
            "model_source": result.get("model_source", ""),
            "segments":     segments,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Segments clients (RFM + KMeans k=5)",
                x_key="segment",
                y_key="nb_clients",
                items=chart_items,
            )
        ],
        "tables": (
            [
                {
                    "title": "Profils de segments clients",
                    "columns": cols,
                    "rows": table_rows,
                    "markdown": _to_markdown_table(cols, table_rows),
                }
            ]
            if table_rows
            else []
        ),
    }
