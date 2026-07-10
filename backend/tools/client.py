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


def _is_agent_question(question: str) -> bool:
    """
    "client" intent's keyword list can still route agent-name lookups here
    (e.g. "trouve l'agent Ben Salah" has no client-related keyword but can
    still be misclassified). Detect that case so we search dim_agent instead
    of silently answering with unrelated client data.
    """
    lowered = question.lower()
    return "agent" in lowered and "client" not in lowered


def _extract_agent_name(question: str) -> str | None:
    named = re.search(r"agents?\s+([A-Za-z\-\s]{3,})", question, flags=re.IGNORECASE)
    if named:
        return named.group(1).strip()
    quoted = re.search(r"['\"]([^'\"]{3,})['\"]", question)
    if quoted:
        return quoted.group(1).strip()
    return None


def _agent_search_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    target_name = _extract_agent_name(question)
    rows: list[dict[str, Any]] = []
    if target_name:
        agent_sql = """
            SELECT
                a.id_agent,
                COALESCE(a.nom_agent, 'N/A') AS agent,
                COALESCE(a.groupe_agent, 'N/A') AS groupe,
                COALESCE(a.localite_agent, 'N/A') AS localite,
                CASE
                    WHEN a.etat_agent = 'A' THEN 'Actif'
                    WHEN a.etat_agent IN ('R', 'I') THEN 'Inactif'
                    ELSE 'N/A'
                END AS etat,
                COUNT(DISTINCT e.id_police) AS nb_polices,
                COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
            FROM dim_agent a
            LEFT JOIN dwh_fact_emission e ON e.id_agent = a.id_agent AND e.etat_quit IN ('E', 'P', 'A')
            WHERE a.nom_agent ILIKE :pattern
            GROUP BY a.id_agent, a.nom_agent, a.groupe_agent, a.localite_agent, a.etat_agent
            ORDER BY total_pnet DESC
            LIMIT 20
        """
        rows = _query_dataframe(agent_sql, {"pattern": f"%{target_name}%"}).to_dict(orient="records")

    summary = (
        f"Agent(s) trouve(s) pour '{target_name}': {len(rows)} resultat(s)."
        if target_name and rows
        else f"Aucun agent trouve pour '{target_name}'." if target_name
        else "Nom d'agent non identifie dans la question."
    )

    return {
        "tool": "client_tool",
        "summary": summary,
        "payload": {
            "target_name": target_name,
            "agent_matches": rows,
        },
        "charts": [
            _build_chart_payload(
                chart_type="bar",
                title=f"Agents correspondant a '{target_name}'",
                x_key="agent",
                y_key="total_pnet",
                items=rows,
            )
        ] if rows else [],
    }


def client_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    if _is_agent_question(question):
        return _agent_search_tool(question, context)

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
