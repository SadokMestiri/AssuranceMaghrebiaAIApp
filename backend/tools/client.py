from __future__ import annotations

import re
from typing import Any

from db import query_dataframe as _query_dataframe
from tools._shared import (
    _normalize_branch,
    _build_chart_payload,
)


def _extract_client_name(question: str, context: dict[str, Any]) -> str | None:
    explicit = context.get("client_name")
    if explicit:
        candidate = str(explicit).strip()
        if candidate:
            return candidate

    quoted = re.search(r"['\"]([^'\"]{3,})['\"]", question)
    if quoted:
        return quoted.group(1).strip()

    named = re.search(r"client\s+([A-Za-z\-\s]{3,})", question, flags=re.IGNORECASE)
    if named:
        return named.group(1).strip()

    return None


def client_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    target_name = _extract_client_name(question, context)

    top_clients_sql = """
        SELECT
            c.id_client,
            COALESCE(c.nom, 'N/A') AS nom,
            COALESCE(c.prenom, 'N/A') AS prenom,
            COALESCE(c.ville, 'N/A') AS ville,
            COUNT(*) AS nb_impayes,
            COALESCE(SUM(i.mt_acp), 0) AS total_impaye,
            COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
        FROM dwh_fact_impaye i
        JOIN dim_police p ON p.id_police = i.id_police
        JOIN dim_client c ON c.id_client = p.id_client
        LEFT JOIN dwh_fact_emission e ON e.num_quittance = i.num_quittance
        WHERE (:branch IS NULL OR i.branche = :branch)
        GROUP BY c.id_client, COALESCE(c.nom, 'N/A'), COALESCE(c.prenom, 'N/A'), COALESCE(c.ville, 'N/A')
        ORDER BY total_impaye DESC
        LIMIT 15
    """

    top_clients = _query_dataframe(top_clients_sql, {"branch": branch}).to_dict(orient="records")

    profile_rows: list[dict[str, Any]] = []
    homonym_rows: list[dict[str, Any]] = []

    if target_name:
        profile_sql = """
            SELECT
                c.id_client,
                COALESCE(c.nom, 'N/A') AS nom,
                COALESCE(c.prenom, 'N/A') AS prenom,
                COALESCE(c.ville, 'N/A') AS ville,
                COUNT(DISTINCT p.id_police) AS nb_polices,
                COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
                COALESCE(SUM(i.mt_acp), 0) AS total_impaye,
                COUNT(i.num_quittance) AS nb_impayes
            FROM dim_client c
            LEFT JOIN dim_police p ON p.id_client = c.id_client
            LEFT JOIN dwh_fact_emission e ON e.id_police = p.id_police AND e.etat_quit IN ('E','P','A')
            LEFT JOIN dwh_fact_impaye i ON i.id_police = p.id_police
            WHERE CONCAT(COALESCE(c.nom, ''), ' ', COALESCE(c.prenom, '')) ILIKE :pattern
            GROUP BY c.id_client, COALESCE(c.nom, 'N/A'), COALESCE(c.prenom, 'N/A'), COALESCE(c.ville, 'N/A')
            ORDER BY total_impaye DESC
            LIMIT 20
        """
        profile_rows = _query_dataframe(profile_sql, {"pattern": f"%{target_name}%"}).to_dict(orient="records")

        homonym_sql = """
            SELECT
                COALESCE(nom, 'N/A') AS nom,
                COALESCE(prenom, 'N/A') AS prenom,
                COUNT(*) AS homonym_count
            FROM dim_client
            WHERE CONCAT(COALESCE(nom, ''), ' ', COALESCE(prenom, '')) ILIKE :pattern
            GROUP BY COALESCE(nom, 'N/A'), COALESCE(prenom, 'N/A')
            HAVING COUNT(*) > 1
            ORDER BY homonym_count DESC
            LIMIT 10
        """
        homonym_rows = _query_dataframe(homonym_sql, {"pattern": f"%{target_name}%"}).to_dict(orient="records")

    summary = (
        f"Client analytics: {len(top_clients)} clients top impaye retournes."
        if not target_name
        else f"Client analytics sur '{target_name}': {len(profile_rows)} profils et {len(homonym_rows)} homonymes."
    )

    return {
        "tool": "client_tool",
        "summary": summary,
        "payload": {
            "branch": branch or "ALL",
            "target_name": target_name,
            "top_claim_clients": top_clients,
            "named_client_profile": profile_rows,
            "homonym_candidates": homonym_rows,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title="Top clients par impaye",
                x_key="id_client",
                y_key="total_impaye",
                items=top_clients,
            )
        ],
    }
