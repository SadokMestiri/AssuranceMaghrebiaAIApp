from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from db import query_dataframe as _query_dataframe
from utils import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    normalize_text as _normalize_text,
    format_metric_value as _format_metric_value,
    format_branch_label as _format_scope_label,
    infer_report_mode as _infer_report_mode,
)
from config import DATA_YEAR_FROM, DATA_YEAR_TO
from tools._shared import (
    _normalize_branch,
    _resolve_period_context,
    _to_markdown_table,
    _build_chart_payload,
)

def _contains_any(text_value: str, keywords: list[str]) -> bool:
    return any(keyword in text_value for keyword in keywords)


def _contains_any_term(text_value: str, terms: list[str]) -> bool:
    for term in terms:
        normalized_term = term.strip().lower()
        if not normalized_term:
            continue

        if " " in normalized_term:
            if normalized_term in text_value:
                return True
            continue

        if re.search(rf"\b{re.escape(normalized_term)}\b", text_value):
            return True

    return False


def _extract_top_n(normalized_question: str, default_value: int = 10) -> int:
    patterns = [r"\btop\s*(\d{1,2})\b", r"\bpremiers?\s*(\d{1,2})\b", r"\bfirst\s*(\d{1,2})\b"]
    for pattern in patterns:
        match = re.search(pattern, normalized_question)
        if match:
            return min(50, max(1, _safe_int(match.group(1), default_value)))
    return default_value


def _detect_sql_metric(normalized_question: str) -> str:
    """
    Détecte la métrique/dimension demandée dans la question.
    """
    
    # ── Sinistre (dédicated table) ──
    sinistre_terms = ["sinistre", "sinistres", "montant paye sinistre", "provision", "provisions"]
    if any(term in normalized_question for term in sinistre_terms):
        if "nature" in normalized_question:
            return "sinistre_nature"
        if "etat" in normalized_question or "ouvert" in normalized_question or "clos" in normalized_question:
            return "sinistre_etat"
        if "responsabilite" in normalized_question:
            return "sinistre_responsabilite"
        if "gouvernorat" in normalized_question or "region" in normalized_question or "zone" in normalized_question:
            return "sinistre_by_gouvernorat"
        if "materiel" in normalized_question:
            return "sinistre_materiel"
        if "branche" in normalized_question:
            return "sinistre_by_branche"
        if "evolution" in normalized_question or "mensuel" in normalized_question or "tendance" in normalized_question:
            return "sinistre_monthly_trend"
        return "sinistre_overview"

    
    # ── Impayé ──
    if any(term in normalized_question for term in ["impaye", "impayes", "acp", "recouvrement"]):
        if "branche" in normalized_question:
            return "impaye_rate_by_branch"
        if "gouvernorat" in normalized_question or "region" in normalized_question:
            return "impaye_by_gouvernorat"
        if "evolution" in normalized_question or "mensuel" in normalized_question:
            return "impaye_monthly_trend"
        if "client" in normalized_question or "top" in normalized_question:
            return "impaye_top_clients"
        return "impaye_overview"

    # ── Résiliation / Churn ──
    if any(term in normalized_question for term in ["resiliation", "annulation", "resilie", "resiliee", "churn"]):
        if "branche" in normalized_question:
            return "resiliation_by_branch"
        if "sexe" in normalized_question:
            return "resiliation_by_sexe"
        if "agent" in normalized_question:
            return "resiliation_by_agent"
        return "resiliation_overview"
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIENT DIMENSIONS - Conditions SPÉCIFIQUES d'abord
    # ═══════════════════════════════════════════════════════════════════════
    
    # D'abord les questions très spécifiques qui pourraient être mal classées
    if "age moyen" in normalized_question or "âge moyen" in normalized_question:
        return "client_age_moyen"
    
    if "tranche" in normalized_question and "age" in normalized_question:
        return "client_age_tranches"
    
    if "sexe" in normalized_question:
        return "client_sexe"
    
    if "personne physique" in normalized_question or "personnes physiques" in normalized_question:
        return "client_type_personne"

    if "personne morale" in normalized_question or "personnes morales" in normalized_question:
        return "client_type_personne"
    
    if "natp" in normalized_question or "nationalite" in normalized_question:
        return "client_natp"
    
    # Ensuite les questions client génériques
    client_terms = ["client", "clients", "assure", "assures"]
    if any(term in normalized_question for term in client_terms):
        if "ville" in normalized_question and ("couvert" in normalized_question or "distinct" in normalized_question or "top" in normalized_question):
            return "client_top_villes"
        if "nombre total" in normalized_question or "total clients" in normalized_question:
            return "client_total"
        return "client_overview"
    
    # ── Agent dimensions ──
    agent_terms = ["agent", "agents", "reseau", "distributeur"]
    if any(term in normalized_question for term in agent_terms):
        if "etat" in normalized_question or "actif" in normalized_question or "inactif" in normalized_question:
            return "agent_etat"
        if "type agent" in normalized_question or "type d'agent" in normalized_question:
            return "agent_type"
        if "groupe" in normalized_question:
            return "agent_groupe"
        if "localite" in normalized_question or "ville" in normalized_question:
            return "agent_localite"
        if "top" in normalized_question and ("prime" in normalized_question or "pnet" in normalized_question):
            return "agent_top_prime"
        if "top" in normalized_question and ("police" in normalized_question or "nombre" in normalized_question):
            return "agent_top_polices"
        if "prime moyenne" in normalized_question or "prime nette moyenne" in normalized_question:
            return "agent_prime_moyenne"
        if "agent" in normalized_question and "nombre de polices" in normalized_question and "top" in normalized_question:
            return "agent_top_polices"
        # No known dimension keyword matched above — check whether "agent" is
        # followed by what looks like a name ("trouve l'agent Ben Salah") rather
        # than falling straight into the generic overview.
        stopwords = {"actif", "actifs", "inactif", "inactifs", "general", "generaux", "de", "des", "du", "le", "la", "les"}
        named = re.search(r"agents?\s+([a-z\-]{3,}(?:\s+[a-z\-]{2,})?)", normalized_question)
        if named and named.group(1).strip() not in stopwords:
            return "agent_search"
        return "agent_overview"
    
    
    # ── Produit dimensions ──
    if any(term in normalized_question for term in [
        "produit", "produits",
        "famille de risque", "familles de risque", "famille risque",
        "top produit", "top produits",
        "prime nette par famille",
        "nombre de produits", "produits distincts"
    ]):
        if "famille" in normalized_question:
            return "produit_familles"
        if "top" in normalized_question and "prime" in normalized_question:
            return "produit_top_prime"
        if "top" in normalized_question and "quittance" in normalized_question:
            return "produit_top_quittances"
        if "branche" in normalized_question:
            return "produit_by_branche"
        if "total" in normalized_question and "prime" in normalized_question:
            return "produit_total_pnet"
        if "top" in normalized_question:
            return "produit_top_prime"
        return "produit_count"
    
    # ── Véhicule dimensions ──
    vehicule_terms = ["vehicule", "vehicules", "voiture", "voitures", "parc auto"]
    if any(term in normalized_question for term in vehicule_terms):
        if "sinistralite" in normalized_question or ("sinistre" in normalized_question and "top" in normalized_question):
            return "vehicule_top_sinistralite"
        if "marque" in normalized_question:
            return "vehicule_top_marques"
        if "genre" in normalized_question or "type" in normalized_question:
            return "vehicule_genre"
        if "puissance" in normalized_question and "moyenne" in normalized_question:
            return "vehicule_puissance_moyenne"
        if "puissance" in normalized_question and "tranche" in normalized_question:
            return "vehicule_puissance_tranches"
        if "age" in normalized_question or "anciennete" in normalized_question:
            return "vehicule_age_moyen"
        if "total" in normalized_question or "nombre" in normalized_question:
            return "vehicule_total"
        return "vehicule_overview"
    if "vehicule" in normalized_question or "voiture" in normalized_question:
        if "particuliere" in normalized_question or "vp" in normalized_question:
            return "vehicule_vp"
        if "utilitaire" in normalized_question or "vu" in normalized_question:
            return "vehicule_vu"
        if "marque" in normalized_question and "distinct" in normalized_question:
            return "vehicule_marques_distinctes"
        if "puissance" in normalized_question and "tranche" in normalized_question:
            return "vehicule_puissance_tranches"
        if "bonus malus" in normalized_question or "bm" in normalized_question:
            return "police_bm_moyen"
    
    # ── Police dimensions ──
    police_terms = ["police", "polices", "portefeuille"]
    if any(term in normalized_question for term in police_terms):
        if "situation" in normalized_question:
            return "police_situation"
        if "periodicite" in normalized_question or "reglement" in normalized_question:
            return "police_periodicite"
        if "type police" in normalized_question or "individuel" in normalized_question or "flotte" in normalized_question:
            return "police_type"
        if "duree" in normalized_question:
            return "police_duree"
        if "bonus malus" in normalized_question and "moyen" in normalized_question:
            return "police_bm_moyen"
        if "bonus malus" in normalized_question and "distribution" in normalized_question:
            return "police_bm_distribution"
        if "total" in normalized_question or "nombre" in normalized_question:
            return "police_total"
        return "police_overview"

    # QUITTANCES EMISES
    if "quittance" in normalized_question and ("nombre" in normalized_question or "nb" in normalized_question or "émises" in normalized_question or "emises" in normalized_question):
        return "quittances_emises"
    
    # ── KPI transactionnels ──
    if any(term in normalized_question for term in ["prime", "pnet", "production", "commission"]):
        if "evolution" in normalized_question or "mensuel" in normalized_question or "tendance" in normalized_question:
            return "prime_monthly_trend"
        if "branche" in normalized_question:
            return "prime_by_branch"
        if "annee" in normalized_question or "annuel" in normalized_question:
            return "prime_yearly"
        return "prime_overview"

    if any(term in normalized_question for term in ["ratio combine", "ratio combin", "combined ratio"]):
        return "ratio_combine"

    # ── Géographique ──
    if "gouvernorat" in normalized_question or "region" in normalized_question:
        if "sinistre" in normalized_question:
            return "sinistre_by_gouvernorat"
        if "impaye" in normalized_question:
            return "impaye_by_gouvernorat"
        return "top_zones_risque"
    if ("ville" in normalized_question and "couvert" in normalized_question) or "nombre de villes" in normalized_question:
        return "client_top_villes"
    if ("zone" in normalized_question and "risque" in normalized_question) or "top zones" in normalized_question:
        return "top_zones_risque"
    
    
    # ── Fallback par défaut ──
    return "prime_overview"


def _detect_sql_aggregation(normalized_question: str, metric: str) -> str:
    if _contains_any_term(normalized_question, ["ratio", "taux"]) and metric in {"impaye", "resiliation"}:
        return "ratio"
    if _contains_any_term(normalized_question, ["moyenne", "average", "avg", "moyen"]):
        return "avg"
    if _contains_any_term(normalized_question, ["distinct", "unique", "uniques", "differents"]):
        return "count_distinct"
    if _contains_any_term(normalized_question, ["nombre", "nb", "count", "combien"]):
        return "count"
    if _contains_any_term(normalized_question, ["somme", "sum", "montant", "total", "cumule", "global"]):
        return "sum"
    if metric == "client":
        return "count"
    if metric == "resiliation":
        return "count"
    return "sum"


def _detect_sql_dimensions(normalized_question: str) -> list[str]:
    dimensions: list[str] = []

    if _contains_any(normalized_question, ["branche", "branches"]):
        dimensions.append("branche")

    if _contains_any(normalized_question, ["gouvernorat", "gouvernorats", "localite", "region", "zone", "ville"]):
        dimensions.append("gouvernorat")

    has_client_dimension = _contains_any(
        normalized_question,
        ["par client", "top client", "top clients", "classement client", "classement des clients"],
    ) or re.search(r"\btop\s*(\d{1,2}\s*)?clients?\b", normalized_question) is not None

    if has_client_dimension:
        dimensions.append("client")

    return dimensions


def _analyze_sql_request(normalized_question: str) -> dict[str, Any]:
    metric = _detect_sql_metric(normalized_question)
    aggregation = _detect_sql_aggregation(normalized_question, metric)
    dimensions = _detect_sql_dimensions(normalized_question)
    is_timeseries = _contains_any(
        normalized_question,
        ["evolution", "trend", "tendance", "mensuel", "mensuelle", "mois", "historique"],
    )
    is_ranking = _contains_any(
        normalized_question,
        ["top", "classement", "ranking", "premier", "premiers", "plus eleve", "plus grand"],
    )
    top_n = _extract_top_n(normalized_question, default_value=10)

    return {
        "metric": metric,
        "aggregation": aggregation,
        "dimensions": dimensions,
        "is_timeseries": is_timeseries,
        "is_ranking": is_ranking,
        "top_n": top_n,
        "normalized_question": normalized_question,
    }


def _build_semantic_sql_query_spec(semantic: dict[str, Any], params: dict[str, Any]) -> dict[str, Any] | None:
    """
    Construit la requête SQL en fonction de la métrique/dimension détectée.
    """
    metric = str(semantic.get("metric", "prime_overview"))
    is_timeseries = bool(semantic.get("is_timeseries", False))
    is_ranking = bool(semantic.get("is_ranking", False))
    top_n = _safe_int(semantic.get("top_n"), 10)
    limit_value = top_n if is_ranking else 50
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIENT DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "client_total":
        return {
            "sql_id": "client_total",
            "sql_query": "SELECT COUNT(*) AS total_clients FROM dim_client",
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "total_clients", "label": "Total clients", "unit": "count"}],
        }
    
    if metric == "client_sexe":
        return {
            "sql_id": "client_by_sexe",
            "sql_query": """
                SELECT
                    c.sexe,
                    COUNT(DISTINCT c.id_client) AS nb_clients,
                    ROUND(100.0 * COUNT(DISTINCT c.id_client) / SUM(COUNT(DISTINCT c.id_client)) OVER (), 1) AS pct
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                INNER JOIN dwh_fact_emission e ON e.id_police = p.id_police
                WHERE c.sexe IN ('F', 'M')
                AND p.situation = 'V'
                AND e.etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR e.branche = :branch)
                AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY c.sexe
                ORDER BY c.sexe
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "sexe", "label": "Sexe", "unit": ""},
                {"key": "nb_clients", "label": "Nombre de clients (portefeuille actif)", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Répartition par sexe (portefeuille actif)", "x_key": "sexe", "y_key": "nb_clients"},
        }
    
    if metric == "client_age_moyen":
        return {
            "sql_id": "client_age_moyen",
            "sql_query": """
                SELECT 
                    ROUND(AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance))), 1) AS age_moyen
                FROM dim_client
                WHERE date_naissance IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "age_moyen", "label": "Âge moyen des clients", "unit": "ans"}],
        }
    
    if metric == "client_age_tranches":
        return {
            "sql_id": "client_age_tranches",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 25 THEN '<25'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 35 THEN '25-34'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 45 THEN '35-44'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 55 THEN '45-54'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)) < 65 THEN '55-64'
                        ELSE '65+'
                    END AS tranche_age,
                    COUNT(*) AS nb_clients
                FROM dim_client
                WHERE date_naissance IS NOT NULL
                GROUP BY tranche_age
                ORDER BY MIN(EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_naissance)))
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "tranche_age", "label": "Tranche d'âge", "unit": ""},
                {"key": "nb_clients", "label": "Nombre de clients", "unit": "count"},
            ],
            "chart": {"type": "bar", "title": "Tranches d'âge des clients", "x_key": "tranche_age", "y_key": "nb_clients"},
        }
    
    if metric == "client_natp":
        return {
            "sql_id": "client_by_natp",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN natp = 'C' THEN 'Citoyen'
                        WHEN natp = 'P' THEN 'Permanent'
                        WHEN natp = 'R' THEN 'Résident'
                        WHEN natp = 'S' THEN 'Sans papiers'
                        ELSE natp
                    END AS nationalite,
                    COUNT(*) AS nb_clients,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_client
                WHERE natp IS NOT NULL
                GROUP BY natp
                ORDER BY nb_clients DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "nationalite", "label": "Nationalité", "unit": ""},
                {"key": "nb_clients", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Nationalité des clients", "x_key": "nationalite", "y_key": "nb_clients"},
        }
    
    if metric == "client_overview":
        return {
            "sql_id": "client_overview",
            "sql_query": """
                SELECT
                    COUNT(DISTINCT c.id_client) AS total_clients,
                    COUNT(DISTINCT c.ville) AS nb_villes,
                    COUNT(DISTINCT CASE WHEN c.type_personne = 'P' THEN c.id_client END) AS nb_personnes_physiques,
                    COUNT(DISTINCT CASE WHEN c.type_personne = 'M' THEN c.id_client END) AS nb_personnes_morales,
                    COUNT(DISTINCT CASE WHEN c.sexe = 'F' THEN c.id_client END) AS nb_femmes,
                    COUNT(DISTINCT CASE WHEN c.sexe = 'M' THEN c.id_client END) AS nb_hommes,
                    ROUND(AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.date_naissance))), 1) AS age_moyen
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                INNER JOIN dwh_fact_emission e ON e.id_police = p.id_police
                WHERE p.situation = 'V'
                AND c.date_naissance IS NOT NULL
                AND e.etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR e.branche = :branch)
                AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_clients", "label": "Total clients (portefeuille actif)", "unit": "count"},
                {"key": "nb_villes", "label": "Villes couvertes", "unit": "count"},
                {"key": "nb_personnes_physiques", "label": "Personnes physiques", "unit": "count"},
                {"key": "nb_personnes_morales", "label": "Personnes morales", "unit": "count"},
                {"key": "nb_femmes", "label": "Femmes", "unit": "count"},
                {"key": "nb_hommes", "label": "Hommes", "unit": "count"},
                {"key": "age_moyen", "label": "Âge moyen", "unit": "ans"},
            ],
        }
    if metric == "client_type_personne":
        return {
            "sql_id": "client_by_type",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN c.type_personne = 'P' THEN 'Personne Physique'
                        WHEN c.type_personne = 'M' THEN 'Personne Morale'
                        ELSE c.type_personne
                    END AS type_client,
                    COUNT(DISTINCT c.id_client) AS nb_clients,
                    ROUND(100.0 * COUNT(DISTINCT c.id_client) / SUM(COUNT(DISTINCT c.id_client)) OVER (), 1) AS pct
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                WHERE c.type_personne IN ('P', 'M')
                AND p.situation = 'V'
                AND (:branch IS NULL OR p.branche = :branch)
                GROUP BY c.type_personne
                ORDER BY c.type_personne
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "type_client", "label": "Type", "unit": ""},
                {"key": "nb_clients", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Type de personne (portefeuille actif)", "x_key": "type_client", "y_key": "nb_clients"},
        }
    if metric == "client_top_villes":
        limit_value = top_n if is_ranking else 10
        return {
            "sql_id": "client_top_villes",
            "sql_query": f"""
                SELECT 
                    COALESCE(c.ville, 'N/A') AS ville,
                    COUNT(DISTINCT c.id_client) AS nb_clients
                FROM dim_client c
                INNER JOIN dim_police p ON p.id_client = c.id_client
                WHERE c.ville IS NOT NULL 
                AND TRIM(c.ville) != ''
                AND p.situation = 'V'
                AND (:branch IS NULL OR p.branche = :branch)
                GROUP BY c.ville
                ORDER BY nb_clients DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "ville", "label": "Ville", "unit": ""},
                {"key": "nb_clients", "label": "Nombre de clients (portefeuille actif)", "unit": "count"},
            ],
            "chart": {"type": "bar", "title": "Top villes par concentration clients", "x_key": "ville", "y_key": "nb_clients"},
        }

    
    # ═══════════════════════════════════════════════════════════════════════
    # AGENT DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "agent_overview":
        return {
            "sql_id": "agent_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_agents,
                    SUM(CASE WHEN etat_agent = 'A' THEN 1 ELSE 0 END) AS agents_actifs,
                    SUM(CASE WHEN etat_agent IN ('R', 'I') THEN 1 ELSE 0 END) AS agents_inactifs,
                    COUNT(DISTINCT groupe_agent) AS nb_groupes,
                    COUNT(DISTINCT localite_agent) AS nb_localites
                FROM dim_agent
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_agents", "label": "Total agents", "unit": "count"},
                {"key": "agents_actifs", "label": "Agents actifs", "unit": "count"},
                {"key": "agents_inactifs", "label": "Agents inactifs", "unit": "count"},
                {"key": "nb_groupes", "label": "Groupes de distribution", "unit": "count"},
                {"key": "nb_localites", "label": "Localités couvertes", "unit": "count"},
            ],
        }
    
    if metric == "agent_search":
        question_text = str(semantic.get("normalized_question", ""))
        name_match = re.search(r"agents?\s+([a-z\-]{3,}(?:\s+[a-z\-]{2,})?)", question_text)
        target_name = name_match.group(1).strip() if name_match else ""
        return {
            "sql_id": "agent_search",
            "sql_query": """
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
            """,
            "params": {"pattern": f"%{target_name}%"},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "groupe", "label": "Groupe", "unit": ""},
                {"key": "localite", "label": "Localité", "unit": ""},
                {"key": "etat", "label": "État", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": f"Agents correspondant a '{target_name}'", "x_key": "agent", "y_key": "total_pnet"},
        }

    if metric == "agent_etat":
        return {
            "sql_id": "agent_by_etat",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN etat_agent = 'A' THEN 'Actif'
                        WHEN etat_agent = 'R' THEN 'Résilié'
                        WHEN etat_agent = 'I' THEN 'Inactif'
                        WHEN etat_agent = 'S' THEN 'Suspendu'
                        ELSE etat_agent
                    END AS statut,
                    COUNT(*) AS nb_agents,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_agent
                GROUP BY etat_agent
                ORDER BY nb_agents DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "statut", "label": "Statut", "unit": ""},
                {"key": "nb_agents", "label": "Nombre d'agents", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "État des agents", "x_key": "statut", "y_key": "nb_agents"},
        }
    
    if metric == "agent_type":
        return {
            "sql_id": "agent_by_type",
            "sql_query": """
                SELECT 
                    type_agent,
                    COUNT(*) AS nb_agents,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_agent
                WHERE type_agent IS NOT NULL
                GROUP BY type_agent
                ORDER BY nb_agents DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "type_agent", "label": "Type d'agent", "unit": ""},
                {"key": "nb_agents", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Type d'agent", "x_key": "type_agent", "y_key": "nb_agents"},
        }
    
    if metric == "agent_groupe":
        return {
            "sql_id": "agent_by_groupe",
            "sql_query": """
                SELECT 
                    groupe_agent,
                    COUNT(*) AS nb_agents,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_agent
                WHERE groupe_agent IS NOT NULL
                GROUP BY groupe_agent
                ORDER BY nb_agents DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "groupe_agent", "label": "Groupe", "unit": ""},
                {"key": "nb_agents", "label": "Nombre d'agents", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Agents par groupe", "x_key": "groupe_agent", "y_key": "nb_agents"},
        }
    
    if metric == "agent_localite":
        return {
            "sql_id": "agent_by_localite",
            "sql_query": f"""
                SELECT 
                    COALESCE(localite_agent, 'N/A') AS localite,
                    COUNT(*) AS nb_agents
                FROM dim_agent
                WHERE localite_agent IS NOT NULL
                GROUP BY localite_agent
                ORDER BY nb_agents DESC
                LIMIT {limit_value}
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "localite", "label": "Localité", "unit": ""},
                {"key": "nb_agents", "label": "Nombre d'agents", "unit": "count"},
            ],
            "chart": {"type": "bar", "title": "Top localités agents", "x_key": "localite", "y_key": "nb_agents"},
        }
    
    if metric == "agent_top_prime":
        return {
            "sql_id": "agent_top_prime",
            "sql_query": f"""
                SELECT 
                    COALESCE(a.nom_agent, 'N/A') AS agent,
                    COALESCE(a.groupe_agent, 'N/A') AS groupe,
                    COUNT(DISTINCT e.id_police) AS nb_polices,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_agent a
                JOIN dwh_fact_emission e ON e.id_agent = a.id_agent
                WHERE e.etat_quit IN ('E', 'P', 'A')
                  AND (:branch IS NULL OR e.branche = :branch)
                  AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY a.nom_agent, a.groupe_agent, a.id_agent
                ORDER BY total_pnet DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "groupe", "label": "Groupe", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top agents par prime nette", "x_key": "agent", "y_key": "total_pnet"},
        }
    
    if metric == "agent_prime_moyenne":
        return {
            "sql_id": "agent_prime_moyenne",
            "sql_query": """
                SELECT 
                    COALESCE(SUM(e.mt_pnet), 0) / NULLIF(SUM(CASE WHEN a.etat_agent = 'A' THEN 1 ELSE 0 END), 0) AS prime_moyenne_agent_actif
                FROM dim_agent a
                LEFT JOIN dwh_fact_emission e ON e.id_agent = a.id_agent
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR e.branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [{"key": "prime_moyenne_agent_actif", "label": "Prime nette moyenne par agent actif", "unit": "TND"}],
        }
    if metric == "agent_top_polices":
        limit_value = top_n if is_ranking else 10
        return {
            "sql_id": "agent_top_polices",
            "sql_query": f"""
                SELECT 
                    COALESCE(a.nom_agent, 'N/A') AS agent,
                    COALESCE(a.groupe_agent, 'N/A') AS groupe,
                    COUNT(DISTINCT e.id_police) AS nb_polices,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_agent a
                JOIN dwh_fact_emission e ON e.id_agent = a.id_agent
                WHERE e.etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR e.branche = :branch)
                AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY a.nom_agent, a.groupe_agent, a.id_agent
                ORDER BY nb_polices DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "groupe", "label": "Groupe", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top agents par nombre de polices", "x_key": "agent", "y_key": "nb_polices"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRODUIT DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "produit_count":
        return {
            "sql_id": "produit_count",
            "sql_query": """
                SELECT 
                    COUNT(DISTINCT code_produit) AS nb_produits,
                    COUNT(DISTINCT famille_risque) AS nb_familles,
                    COUNT(DISTINCT branche) AS nb_branches
                FROM dim_produit
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_produits", "label": "Produits distincts", "unit": "count"},
                {"key": "nb_familles", "label": "Familles de risque", "unit": "count"},
                {"key": "nb_branches", "label": "Branches couvertes", "unit": "count"},
            ],
        }
    
    if metric == "produit_familles":
        return {
            "sql_id": "produit_by_famille",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.famille_risque, 'N/A') AS famille,
                    COUNT(DISTINCT p.code_produit) AS nb_produits,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.famille_risque
                ORDER BY total_pnet DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "famille", "label": "Famille de risque", "unit": ""},
                {"key": "nb_produits", "label": "Nombre de produits", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Prime nette par famille de risque", "x_key": "famille", "y_key": "total_pnet"},
        }
    
    if metric == "produit_top_prime":
        return {
            "sql_id": "produit_top_prime",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.lib_produit, CAST(p.code_produit AS VARCHAR)) AS produit,
                    p.branche,
                    COUNT(DISTINCT pol.id_police) AS nb_polices,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.lib_produit, p.code_produit, p.branche
                ORDER BY total_pnet DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "produit", "label": "Produit", "unit": ""},
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_polices", "label": "Nb polices", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top produits par prime nette", "x_key": "produit", "y_key": "total_pnet"},
        }
    
    if metric == "produit_by_branche":
        return {
            "sql_id": "produit_by_branche",
            "sql_query": """
                SELECT 
                    branche,
                    COUNT(DISTINCT code_produit) AS nb_produits,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                GROUP BY branche
                ORDER BY total_pnet DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_produits", "label": "Nb produits", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "pie", "title": "Prime nette par branche", "x_key": "branche", "y_key": "total_pnet"},
        }
    
    if metric == "produit_top_quittances":
        return {
            "sql_id": "produit_top_quittances",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.lib_produit, CAST(p.code_produit AS VARCHAR)) AS produit,
                    p.branche,
                    COUNT(e.num_quittance) AS nb_quittances,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.lib_produit, p.code_produit, p.branche
                ORDER BY nb_quittances DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "produit", "label": "Produit", "unit": ""},
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_quittances", "label": "Quittances", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top produits par volume quittances", "x_key": "produit", "y_key": "nb_quittances"},
        }
    if metric == "produit_total_pnet":
        return {
            "sql_id": "produit_total_pnet",
            "sql_query": """
                SELECT 
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
                    COUNT(DISTINCT e.num_quittance) AS nb_quittances
                FROM dwh_fact_emission e
                JOIN dim_police p ON p.id_police = e.id_police
                WHERE e.etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR p.branche = :branch)
                AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_pnet", "label": "Prime nette totale des produits", "unit": "TND"},
                {"key": "nb_quittances", "label": "Nombre de quittances", "unit": "count"},
            ],
        }
    if metric == "produit_top_single":
        return {
            "sql_id": "produit_top_single",
            "sql_query": f"""
                SELECT 
                    COALESCE(p.lib_produit, CAST(p.code_produit AS VARCHAR)) AS produit,
                    p.branche,
                    COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                FROM dim_produit p
                LEFT JOIN dim_police pol ON pol.code_produit = p.code_produit
                LEFT JOIN dwh_fact_emission e ON e.id_police = pol.id_police
                    AND e.etat_quit IN ('E', 'P', 'A')
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                WHERE (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.lib_produit, p.code_produit, p.branche
                ORDER BY total_pnet DESC
                LIMIT 1
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "produit", "label": "Top produit", "unit": ""},
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
            ],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # VEHICULE DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "vehicule_total":
        return {
            "sql_id": "vehicule_total",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_vehicules,
                    COUNT(DISTINCT marque) AS nb_marques,
                    ROUND(AVG(puissance), 1) AS puissance_moyenne,
                    ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM date_mec)), 1) AS age_moyen
                FROM dim_vehicule
                WHERE puissance IS NOT NULL AND puissance > 0
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_vehicules", "label": "Total véhicules", "unit": "count"},
                {"key": "nb_marques", "label": "Marques distinctes", "unit": "count"},
                {"key": "puissance_moyenne", "label": "Puissance moyenne", "unit": "CV"},
                {"key": "age_moyen", "label": "Âge moyen", "unit": "ans"},
            ],
        }
    
    if metric == "vehicule_top_sinistralite":
        return {
            "sql_id": "vehicule_top_sinistralite",
            "sql_query": f"""
                SELECT
                    v.id_vehicule,
                    COALESCE(v.marque, 'N/A') AS marque,
                    COALESCE(v.immatriculation, 'N/A') AS immatriculation,
                    COALESCE(v.genre_vehicule, 'N/A') AS genre,
                    COUNT(s.id_sinistre) AS nb_sinistres,
                    COALESCE(SUM(s.mt_paye), 0) AS total_mt_paye
                FROM dim_vehicule v
                INNER JOIN dwh_fact_sinistre s ON s.id_vehicule = v.id_vehicule
                WHERE (:branch IS NULL OR s.branche = :branch)
                  AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR s.annee_survenance <= :year_to)
                GROUP BY v.id_vehicule, v.marque, v.immatriculation, v.genre_vehicule
                ORDER BY nb_sinistres DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "marque", "label": "Marque", "unit": ""},
                {"key": "immatriculation", "label": "Immatriculation", "unit": ""},
                {"key": "genre", "label": "Genre", "unit": ""},
                {"key": "nb_sinistres", "label": "Nb sinistres", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top véhicules par sinistralité", "x_key": "immatriculation", "y_key": "nb_sinistres"},
        }

    if metric == "vehicule_top_marques":
        return {
            "sql_id": "vehicule_top_marques",
            "sql_query": f"""
                SELECT 
                    COALESCE(marque, 'N/A') AS marque,
                    COUNT(*) AS nb_vehicules,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_vehicule
                WHERE marque IS NOT NULL
                GROUP BY marque
                ORDER BY nb_vehicules DESC
                LIMIT {limit_value}
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "marque", "label": "Marque", "unit": ""},
                {"key": "nb_vehicules", "label": "Nombre de véhicules", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Top marques par parc assuré", "x_key": "marque", "y_key": "nb_vehicules"},
        }
    
    if metric == "vehicule_genre":
        return {
            "sql_id": "vehicule_by_genre",
            "sql_query": """
                SELECT 
                    COALESCE(genre_vehicule, 'N/A') AS genre,
                    COUNT(*) AS nb_vehicules,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_vehicule
                WHERE genre_vehicule IS NOT NULL
                GROUP BY genre_vehicule
                ORDER BY nb_vehicules DESC
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "genre", "label": "Genre", "unit": ""},
                {"key": "nb_vehicules", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Genre de véhicule", "x_key": "genre", "y_key": "nb_vehicules"},
        }
    
    if metric == "vehicule_puissance_moyenne":
        return {
            "sql_id": "vehicule_puissance_moyenne",
            "sql_query": """
                SELECT 
                    ROUND(AVG(puissance), 1) AS puissance_moyenne
                FROM dim_vehicule
                WHERE puissance IS NOT NULL AND puissance > 0
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "puissance_moyenne", "label": "Puissance moyenne", "unit": "CV"}],
        }
    
    if metric == "vehicule_age_moyen":
        return {
            "sql_id": "vehicule_age_moyen",
            "sql_query": """
                SELECT 
                    ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM date_mec)), 1) AS age_moyen
                FROM dim_vehicule
                WHERE date_mec IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "age_moyen", "label": "Âge moyen des véhicules", "unit": "ans"}],
        }
    
    if metric == "vehicule_overview":
        return {
            "sql_id": "vehicule_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_vehicules,
                    SUM(CASE WHEN genre_vehicule = 'VP' THEN 1 ELSE 0 END) AS nb_vp,
                    SUM(CASE WHEN genre_vehicule IN ('VU', 'PL', 'TC', 'AR') THEN 1 ELSE 0 END) AS nb_utilitaires,
                    COUNT(DISTINCT marque) AS nb_marques,
                    ROUND(AVG(puissance), 1) AS puissance_moyenne,
                    ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - EXTRACT(YEAR FROM date_mec)), 1) AS age_moyen
                FROM dim_vehicule
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_vehicules", "label": "Total véhicules", "unit": "count"},
                {"key": "nb_vp", "label": "Voitures particulières", "unit": "count"},
                {"key": "nb_utilitaires", "label": "Véhicules utilitaires", "unit": "count"},
                {"key": "nb_marques", "label": "Marques distinctes", "unit": "count"},
                {"key": "puissance_moyenne", "label": "Puissance moyenne", "unit": "CV"},
                {"key": "age_moyen", "label": "Âge moyen", "unit": "ans"},
            ],
        }
    if metric == "vehicule_vp":
        return {
            "sql_id": "vehicule_vp",
            "sql_query": """
                SELECT COUNT(*) AS nb_vp
                FROM dim_vehicule
                WHERE genre_vehicule = 'VP'
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_vp", "label": "Voitures particulières", "unit": "count"}],
        }
    if metric == "vehicule_vu":
        return {
            "sql_id": "vehicule_vu",
            "sql_query": """
                SELECT COUNT(*) AS nb_vu
                FROM dim_vehicule
                WHERE genre_vehicule IN ('VU', 'PL', 'TC', 'AR')
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_vu", "label": "Véhicules utilitaires", "unit": "count"}],
        }
    if metric == "vehicule_marques_distinctes":
        return {
            "sql_id": "vehicule_marques_distinctes",
            "sql_query": """
                SELECT COUNT(DISTINCT marque) AS nb_marques
                FROM dim_vehicule
                WHERE marque IS NOT NULL
            """,
            "params": {},
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_marques", "label": "Marques distinctes", "unit": "count"}],
        }
    if metric == "vehicule_puissance_tranches":
        return {
            "sql_id": "vehicule_puissance_tranches",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN puissance <= 4 THEN '≤4 CV'
                        WHEN puissance <= 7 THEN '5-7 CV'
                        WHEN puissance <= 10 THEN '8-10 CV'
                        WHEN puissance <= 14 THEN '11-14 CV'
                        ELSE '15+ CV'
                    END AS tranche_puissance,
                    COUNT(*) AS nb_vehicules,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_vehicule
                WHERE puissance IS NOT NULL
                GROUP BY tranche_puissance
                ORDER BY MIN(puissance)
            """,
            "params": {},
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "tranche_puissance", "label": "Tranche de puissance", "unit": ""},
                {"key": "nb_vehicules", "label": "Nombre de véhicules", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Répartition par tranche de puissance", "x_key": "tranche_puissance", "y_key": "nb_vehicules"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # POLICE DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "police_total":
        return {
            "sql_id": "police_total",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_en_vigueur,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    SUM(CASE WHEN type_police = 'flotte' THEN 1 ELSE 0 END) AS polices_flotte,
                    ROUND(AVG(bonus_malus), 2) AS bm_moyen
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_en_vigueur", "label": "En vigueur", "unit": "count"},
                {"key": "polices_resiliees", "label": "Résiliées", "unit": "count"},
                {"key": "polices_flotte", "label": "Polices flotte", "unit": "count"},
                {"key": "bm_moyen", "label": "Bonus-Malus moyen", "unit": ""},
            ],
        }
    
    if metric == "police_situation":
        return {
            "sql_id": "police_by_situation",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN situation = 'V' THEN 'En vigueur'
                        WHEN situation = 'R' THEN 'Résiliée'
                        WHEN situation = 'T' THEN 'Terminée'
                        WHEN situation = 'S' THEN 'Suspendue'
                        WHEN situation = 'A' THEN 'Annulée'
                        ELSE situation
                    END AS statut,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                GROUP BY situation
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "statut", "label": "Situation", "unit": ""},
                {"key": "nb_polices", "label": "Nombre de polices", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Situation du portefeuille", "x_key": "statut", "y_key": "nb_polices"},
        }
    
    if metric == "police_periodicite":
        return {
            "sql_id": "police_by_periodicite",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN periodicite = 'A' THEN 'Annuelle'
                        WHEN periodicite = 'S' THEN 'Semestrielle'
                        WHEN periodicite = 'T' THEN 'Trimestrielle'
                        WHEN periodicite = 'C' THEN 'Comptant'
                        ELSE periodicite
                    END AS periodicite_label,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                AND periodicite IS NOT NULL
                GROUP BY periodicite
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "periodicite_label", "label": "Périodicité", "unit": ""},
                {"key": "nb_polices", "label": "Nombre de polices", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Périodicité de règlement", "x_key": "periodicite_label", "y_key": "nb_polices"},
        }
    
    if metric == "police_type":
        return {
            "sql_id": "police_by_type",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN type_police = 'individuel' THEN 'Individuelle'
                        WHEN type_police = 'flotte' THEN 'Flotte'
                        ELSE type_police
                    END AS type_label,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                  AND type_police IS NOT NULL
                GROUP BY type_police
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "type_label", "label": "Type", "unit": ""},
                {"key": "nb_polices", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Type de police", "x_key": "type_label", "y_key": "nb_polices"},
        }
    
    if metric == "police_duree":
        return {
            "sql_id": "police_by_duree",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN duree = 'R' THEN 'Renouvelable'
                        WHEN duree = 'F' THEN 'Ferme'
                        WHEN duree = 'STR' THEN 'STR'
                        ELSE duree
                    END AS duree_label,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                  AND duree IS NOT NULL
                GROUP BY duree
                ORDER BY nb_polices DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "duree_label", "label": "Durée", "unit": ""},
                {"key": "nb_polices", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Durée de police", "x_key": "duree_label", "y_key": "nb_polices"},
        }
    
    if metric == "police_bm_moyen":
        return {
            "sql_id": "police_bm_moyen",
            "sql_query": """
                SELECT 
                    ROUND(AVG(bonus_malus), 2) AS bm_moyen
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                AND bonus_malus IS NOT NULL
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [{"key": "bm_moyen", "label": "Bonus-Malus moyen", "unit": ""}],
        }
    
    if metric == "police_bm_distribution":
        return {
            "sql_id": "police_bm_distribution",
            "sql_query": """
                SELECT 
                    CASE 
                        WHEN bonus_malus <= 1 THEN '≤1 (Bonus)'
                        WHEN bonus_malus <= 1.5 THEN '1.01-1.5'
                        WHEN bonus_malus <= 2 THEN '1.51-2'
                        ELSE '>2 (Malus)'
                    END AS tranche_bm,
                    COUNT(*) AS nb_polices,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
                AND bonus_malus IS NOT NULL
                GROUP BY tranche_bm
                ORDER BY MIN(bonus_malus)
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "tranche_bm", "label": "Bonus-Malus", "unit": ""},
                {"key": "nb_polices", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Distribution Bonus-Malus", "x_key": "tranche_bm", "y_key": "nb_polices"},
        }
    
    if metric == "police_overview":
        return {
            "sql_id": "police_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_en_vigueur,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    SUM(CASE WHEN type_police = 'flotte' THEN 1 ELSE 0 END) AS polices_flotte,
                    ROUND(AVG(bonus_malus), 2) AS bm_moyen,
                    ROUND(100.0 * SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) / COUNT(*), 1) AS taux_vigueur
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_en_vigueur", "label": "En vigueur", "unit": "count"},
                {"key": "polices_resiliees", "label": "Résiliées", "unit": "count"},
                {"key": "polices_flotte", "label": "Polices flotte", "unit": "count"},
                {"key": "bm_moyen", "label": "BM moyen", "unit": ""},
                {"key": "taux_vigueur", "label": "Taux d'en vigueur", "unit": "%"},
            ],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SINISTRE DIMENSIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "sinistre_overview":
        return {
            "sql_id": "sinistre_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_sinistres,
                    SUM(CASE WHEN etat_sinistre = 'Ouvert' THEN 1 ELSE 0 END) AS sinistres_ouverts,
                    SUM(CASE WHEN etat_sinistre = 'Clos' THEN 1 ELSE 0 END) AS sinistres_clos,
                    COALESCE(SUM(mt_evaluation), 0) AS total_mt_evaluation,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                    SUM(CASE WHEN nature_sinistre = 'Matériel' THEN 1 ELSE 0 END) AS sinistres_materiels
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_sinistres", "label": "Total sinistres", "unit": "count"},
                {"key": "sinistres_ouverts", "label": "Sinistres ouverts", "unit": "count"},
                {"key": "sinistres_clos", "label": "Sinistres clos", "unit": "count"},
                {"key": "total_mt_evaluation", "label": "Montant évalué", "unit": "TND"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "sinistres_materiels", "label": "Sinistres matériels", "unit": "count"},
            ],
        }
    
    if metric == "sinistre_nature":
        return {
            "sql_id": "sinistre_by_nature",
            "sql_query": f"""
                SELECT 
                    COALESCE(nature_sinistre, 'N/A') AS nature,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                GROUP BY nature_sinistre
                ORDER BY nb_sinistres DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "nature", "label": "Nature", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Sinistres par nature", "x_key": "nature", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_etat":
        return {
            "sql_id": "sinistre_by_etat",
            "sql_query": """
                SELECT 
                    etat_sinistre,
                    COUNT(*) AS nb_sinistres,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                GROUP BY etat_sinistre
                ORDER BY nb_sinistres DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "etat_sinistre", "label": "État", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "État des sinistres", "x_key": "etat_sinistre", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_responsabilite":
        return {
            "sql_id": "sinistre_by_responsabilite",
            "sql_query": """
                SELECT 
                    responsabilite,
                    CASE 
                        WHEN responsabilite = 0 THEN '0% (Tiers responsable)'
                        WHEN responsabilite = 50 THEN '50% (Partage)'
                        WHEN responsabilite = 100 THEN '100% (Assuré)'
                        ELSE CAST(responsabilite AS VARCHAR) || '%'
                    END AS responsabilite_label,
                    COUNT(*) AS nb_sinistres,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                  AND responsabilite IS NOT NULL
                GROUP BY responsabilite
                ORDER BY responsabilite
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "responsabilite_label", "label": "Responsabilité", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Responsabilité engagée", "x_key": "responsabilite_label", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_by_branche":
        return {
            "sql_id": "sinistre_by_branche",
            "sql_query": """
                SELECT 
                    branche,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(mt_evaluation), 0) AS total_mt_evaluation,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM dwh_fact_sinistre
                WHERE (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                GROUP BY branche
                ORDER BY nb_sinistres DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre", "unit": "count"},
                {"key": "total_mt_evaluation", "label": "Montant évalué", "unit": "TND"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "pct", "label": "Pourcentage", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Sinistres par branche", "x_key": "branche", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_monthly_trend":
        return {
            "sql_id": "sinistre_monthly_trend",
            "sql_query": """
                SELECT 
                    make_date(annee_survenance, mois_survenance, 1) AS period,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(mt_evaluation), 0) AS total_mt_evaluation,
                    COALESCE(SUM(mt_paye), 0) AS total_mt_paye
                FROM dwh_fact_sinistre
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_survenance >= :year_from)
                  AND (:year_to IS NULL OR annee_survenance <= :year_to)
                  AND annee_survenance BETWEEN 1900 AND 2100
                  AND mois_survenance BETWEEN 1 AND 12
                GROUP BY make_date(annee_survenance, mois_survenance, 1)
                ORDER BY period
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "line", "title": "Évolution mensuelle sinistres", "x_key": "period", "y_key": "nb_sinistres"},
        }
    
    if metric == "sinistre_materiel":
        return {
            "sql_id": "sinistre_materiel",
            "sql_query": """
                SELECT COUNT(*) AS nb_sinistres_materiels
                FROM dwh_fact_sinistre
                WHERE nature_sinistre = 'Matériel'
                AND (:branch IS NULL OR branche = :branch)
                AND (:year_from IS NULL OR annee_survenance >= :year_from)
                AND (:year_to IS NULL OR annee_survenance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [{"key": "nb_sinistres_materiels", "label": "Sinistres matériels", "unit": "count"}],
        }
    if metric == "sinistre_by_gouvernorat":
        limit_value = top_n if is_ranking else 20
        return {
            "sql_id": "sinistre_by_gouvernorat",
            "sql_query": f"""
                SELECT 
                    TRIM(UPPER(c.ville)) AS gouvernorat,
                    COUNT(*) AS nb_sinistres,
                    COALESCE(SUM(s.mt_paye), 0) AS total_mt_paye,
                    COALESCE(SUM(s.mt_evaluation), 0) AS total_mt_evaluation
                FROM dwh_fact_sinistre s
                JOIN dim_client c ON c.id_client = s.id_client
                WHERE c.ville IS NOT NULL
                AND TRIM(c.ville) != ''
                AND (:branch IS NULL OR s.branche = :branch)
                AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
                AND (:year_to IS NULL OR s.annee_survenance <= :year_to)
                GROUP BY TRIM(UPPER(c.ville))
                ORDER BY nb_sinistres DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "gouvernorat", "label": "Gouvernorat", "unit": ""},
                {"key": "nb_sinistres", "label": "Nombre de sinistres", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Sinistres par gouvernorat", "x_key": "gouvernorat", "y_key": "nb_sinistres"},
        }
    
    if metric == "top_zones_risque":
        limit_value = min(10, top_n) if is_ranking else 10
        return {
            "sql_id": "top_zones_risque",
            "sql_query": f"""
                WITH emission AS (
                    SELECT
                        TRIM(UPPER(c.ville)) AS gouvernorat,
                        COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
                    FROM dwh_fact_emission e
                    JOIN dim_police p ON p.id_police = e.id_police        
                    JOIN dim_client c ON c.id_client = p.id_client        
                    WHERE e.etat_quit IN ('E','P','A')
                    AND e.mt_pnet >= 0
                    AND c.ville IS NOT NULL AND TRIM(c.ville) != ''
                    AND (:branch IS NULL OR e.branche = :branch)
                    AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
                    AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
                    GROUP BY TRIM(UPPER(c.ville))
                ),
                sinistres AS (
                    SELECT
                        TRIM(UPPER(c.ville)) AS gouvernorat,
                        COUNT(*) AS nb_sinistres,
                        COALESCE(SUM(s.mt_paye), 0) AS total_mt_paye
                    FROM dwh_fact_sinistre s
                    JOIN dim_client c ON c.id_client = s.id_client
                    WHERE c.ville IS NOT NULL AND TRIM(c.ville) != ''
                    AND (:branch IS NULL OR s.branche = :branch)
                    AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
                    AND (:year_to IS NULL OR s.annee_survenance <= :year_to)
                    GROUP BY TRIM(UPPER(c.ville))
                ),
                merged AS (
                    SELECT
                        COALESCE(e.gouvernorat, s.gouvernorat) AS gouvernorat,
                        COALESCE(e.total_pnet, 0) AS total_pnet,
                        COALESCE(s.nb_sinistres, 0) AS nb_sinistres,
                        COALESCE(s.total_mt_paye, 0) AS total_mt_paye,
                        ROUND(100.0 * COALESCE(s.total_mt_paye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0), 2) AS taux_sinistres_sur_pnet_pct
                    FROM emission e
                    FULL OUTER JOIN sinistres s ON s.gouvernorat = e.gouvernorat
                )
                SELECT
                    gouvernorat,
                    total_pnet,
                    nb_sinistres,
                    total_mt_paye,
                    COALESCE(taux_sinistres_sur_pnet_pct, 0) AS taux_sinistres_sur_pnet_pct,
                    ROUND(
                        0.70 * COALESCE(taux_sinistres_sur_pnet_pct, 0)
                        + 30.0 * COALESCE(nb_sinistres, 0) / NULLIF(MAX(nb_sinistres) OVER (), 0),
                        2
                    ) AS score_risque
                FROM merged
                WHERE COALESCE(total_pnet, 0) > 0 OR COALESCE(nb_sinistres, 0) > 0
                ORDER BY score_risque DESC NULLS LAST
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "gouvernorat", "label": "Gouvernorat", "unit": ""},
                {"key": "nb_sinistres", "label": "Nb sinistres", "unit": "count"},
                {"key": "total_mt_paye", "label": "Montant payé", "unit": "TND"},
                {"key": "taux_sinistres_sur_pnet_pct", "label": "Taux S/P", "unit": "%"},
                {"key": "score_risque", "label": "Score risque", "unit": ""},
            ],
            "chart": {"type": "bar", "title": "Top zones à risque", "x_key": "gouvernorat", "y_key": "score_risque"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # KPI TRANSACTIONNELS (existants, légèrement enrichis)
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "prime_overview":
        return {
            "sql_id": "prime_overview",
            "sql_query": """
                SELECT 
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COALESCE(AVG(mt_pnet), 0) AS avg_pnet,
                    COUNT(*) AS nb_quittances,
                    COUNT(DISTINCT id_police) AS nb_polices,
                    COALESCE(SUM(mt_commission), 0) AS total_commission
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_pnet", "label": "Prime nette totale", "unit": "TND"},
                {"key": "avg_pnet", "label": "Prime nette moyenne", "unit": "TND"},
                {"key": "nb_quittances", "label": "Nombre de quittances", "unit": "count"},
                {"key": "nb_polices", "label": "Polices distinctes", "unit": "count"},
                {"key": "total_commission", "label": "Commission totale", "unit": "TND"},
            ],
        }
    
    if metric == "ratio_combine":
        return {
            "sql_id": "ratio_combine",
            "sql_query": """
                WITH emission AS (
                    SELECT COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                           COALESCE(SUM(mt_commission), 0) AS total_commission
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:branch IS NULL OR branche = :branch)
                      AND (:year_from IS NULL OR annee_echeance >= :year_from)
                      AND (:year_to IS NULL OR annee_echeance <= :year_to)
                ),
                sinistre AS (
                    SELECT COALESCE(SUM(mt_paye), 0) AS total_mt_paye
                    FROM dwh_fact_sinistre
                    WHERE (:branch IS NULL OR branche = :branch)
                      AND (:year_from IS NULL OR annee_survenance >= :year_from)
                      AND (:year_to IS NULL OR annee_survenance <= :year_to)
                )
                SELECT 
                    e.total_pnet,
                    e.total_commission,
                    s.total_mt_paye,
                    ROUND(100.0 * (s.total_mt_paye + e.total_commission) / NULLIF(e.total_pnet, 0), 2) AS ratio_combine_pct,
                    ROUND(100.0 * s.total_mt_paye / NULLIF(e.total_pnet, 0), 2) AS sp_ratio_pct,
                    ROUND(100.0 * e.total_commission / NULLIF(e.total_pnet, 0), 2) AS expense_ratio_pct
                FROM emission e, sinistre s
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "ratio_combine_pct", "label": "Ratio Combiné", "unit": "%"},
                {"key": "sp_ratio_pct", "label": "S/P pur", "unit": "%"},
                {"key": "expense_ratio_pct", "label": "Expense ratio", "unit": "%"},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
                {"key": "total_mt_paye", "label": "Sinistres payés", "unit": "TND"},
                {"key": "total_commission", "label": "Commission", "unit": "TND"},
            ],
        }
    
    if metric == "prime_monthly_trend":
        return {
            "sql_id": "prime_monthly_trend",
            "sql_query": """
                SELECT 
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COUNT(*) AS nb_quittances
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
                ORDER BY period
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "line", "title": "Évolution mensuelle prime nette", "x_key": "period", "y_key": "total_pnet"},
        }
    
    if metric == "prime_by_branch":
        return {
            "sql_id": "prime_by_branch",
            "sql_query": """
                SELECT 
                    branche,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COUNT(*) AS nb_quittances,
                    ROUND(100.0 * COALESCE(SUM(mt_pnet), 0) / SUM(COALESCE(SUM(mt_pnet), 0)) OVER (), 2) AS pct
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P','A')
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                GROUP BY branche
                ORDER BY total_pnet DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
                {"key": "nb_quittances", "label": "Quittances", "unit": "count"},
                {"key": "pct", "label": "Part", "unit": "%"},
            ],
            "chart": {"type": "pie", "title": "Part de production par branche", "x_key": "branche", "y_key": "total_pnet"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # RÉSILIATION
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "resiliation_overview":
        return {
            "sql_id": "resiliation_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_actives,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / COUNT(*), 2) AS taux_resiliation_pct
                FROM dim_police
                WHERE (:branch IS NULL OR branche = :branch)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_actives", "label": "Polices actives", "unit": "count"},
                {"key": "polices_resiliees", "label": "Polices résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux de résiliation", "unit": "%"},
            ],
        }
    
    if metric == "resiliation_by_branch":
        return {
            "sql_id": "resiliation_by_branch",
            "sql_query": """
                SELECT 
                    branche,
                    COUNT(*) AS total_polices,
                    SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS nb_resiliees,
                    ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / COUNT(*), 2) AS taux_resiliation_pct
                FROM dim_police
                WHERE branche IN ('AUTO', 'IRDS', 'SANTE')
                  AND (:branch IS NULL OR branche = :branch)
                GROUP BY branche
                ORDER BY taux_resiliation_pct DESC
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "nb_resiliees", "label": "Polices résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux de résiliation", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Résiliation par branche", "x_key": "branche", "y_key": "taux_resiliation_pct"},
        }
    
    if metric == "resiliation_by_sexe":
        return {
            "sql_id": "resiliation_by_sexe",
            "sql_query": """
                SELECT 
                    COALESCE(c.sexe, 'N/A') AS sexe,
                    COUNT(DISTINCT p.id_police) AS total_polices,
                    SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
                    ROUND(100.0 * SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) / COUNT(DISTINCT p.id_police), 2) AS taux_resiliation_pct
                FROM dim_police p
                LEFT JOIN dim_client c ON c.id_client = p.id_client
                WHERE (:branch IS NULL OR p.branche = :branch)
                  AND c.sexe IN ('F', 'M')
                GROUP BY c.sexe
                ORDER BY c.sexe
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "sexe", "label": "Sexe", "unit": ""},
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "polices_resiliees", "label": "Polices résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Résiliation par sexe", "x_key": "sexe", "y_key": "polices_resiliees"},
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # IMPAYÉ
    # ═══════════════════════════════════════════════════════════════════════
    
    if metric == "impaye_overview":
        return {
            "sql_id": "impaye_overview",
            "sql_query": """
                SELECT 
                    COUNT(*) AS nb_impayes,
                    COALESCE(SUM(mt_acp), 0) AS total_impaye,
                    COALESCE(AVG(mt_acp), 0) AS avg_impaye,
                    COUNT(DISTINCT id_police) AS nb_polices_impactees
                FROM dwh_fact_impaye
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_impayes", "label": "Nombre d'impayés", "unit": "count"},
                {"key": "total_impaye", "label": "Montant total impayé", "unit": "TND"},
                {"key": "avg_impaye", "label": "Montant moyen", "unit": "TND"},
                {"key": "nb_polices_impactees", "label": "Polices impactées", "unit": "count"},
            ],
        }
    
    if metric == "impaye_rate_by_branch":
        return {
            "sql_id": "impaye_rate_by_branch",
            "sql_query": """
                WITH emission AS (
                    SELECT branche, COALESCE(SUM(mt_pnet), 0) AS total_pnet
                    FROM dwh_fact_emission
                    WHERE etat_quit IN ('E','P','A')
                      AND (:year_from IS NULL OR annee_echeance >= :year_from)
                      AND (:year_to IS NULL OR annee_echeance <= :year_to)
                    GROUP BY branche
                ),
                impayes AS (
                    SELECT branche, COALESCE(SUM(mt_acp), 0) AS total_impaye
                    FROM dwh_fact_impaye
                    WHERE (:year_from IS NULL OR annee_echeance >= :year_from)
                      AND (:year_to IS NULL OR annee_echeance <= :year_to)
                    GROUP BY branche
                )
                SELECT 
                    COALESCE(e.branche, i.branche) AS branche,
                    COALESCE(e.total_pnet, 0) AS total_pnet,
                    COALESCE(i.total_impaye, 0) AS total_impaye,
                    ROUND(100.0 * COALESCE(i.total_impaye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0), 2) AS taux_impaye_pct
                FROM emission e
                FULL OUTER JOIN impayes i ON i.branche = e.branche
                ORDER BY taux_impaye_pct DESC NULLS LAST
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "branche", "label": "Branche", "unit": ""},
                {"key": "total_pnet", "label": "Prime nette", "unit": "TND"},
                {"key": "total_impaye", "label": "Impayé", "unit": "TND"},
                {"key": "taux_impaye_pct", "label": "Taux impayé/prime", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Taux impayé par branche", "x_key": "branche", "y_key": "taux_impaye_pct"},
        }
    # ═══════════════════════════════════════════════════════════════════════
    # QUITTANCES EMISES
    # ═══════════════════════════════════════════════════════════════════════
    if metric == "quittances_emises":
        return {
            "sql_id": "quittances_emises",
            "sql_query": """
                SELECT 
                    COUNT(DISTINCT num_quittance) AS nb_quittances,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E', 'P', 'A')
                AND (:branch IS NULL OR branche = :branch)
                AND (:year_from IS NULL OR annee_echeance >= :year_from)
                AND (:year_to IS NULL OR annee_echeance <= :year_to)
            """,
            "params": params,
            "result_kind": "scalar",
            "kpi_fields": [
                {"key": "nb_quittances", "label": "Nombre de quittances émises", "unit": "count"},
                {"key": "total_pnet", "label": "Prime nette totale", "unit": "TND"},
            ],
        }
    # ── Handlers for metrics detected above that had no spec ──────────────────

    if metric == "impaye_by_gouvernorat":
        return {
            "sql_id": "impaye_by_gouvernorat",
            "sql_query": f"""
                SELECT
                    TRIM(UPPER(c.ville)) AS gouvernorat,
                    COUNT(i.num_quittance) AS nb_impayes,
                    COALESCE(SUM(i.mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye i
                JOIN dim_police p ON p.id_police = i.id_police
                JOIN dim_client c ON c.id_client = p.id_client
                WHERE c.ville IS NOT NULL AND TRIM(c.ville) != ''
                  AND (:branch IS NULL OR i.branche = :branch)
                  AND (:year_from IS NULL OR i.annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR i.annee_echeance <= :year_to)
                GROUP BY TRIM(UPPER(c.ville))
                ORDER BY total_impaye DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "gouvernorat", "label": "Gouvernorat", "unit": ""},
                {"key": "nb_impayes", "label": "Nombre d'impayés", "unit": "count"},
                {"key": "total_impaye", "label": "Montant impayé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Impayés par gouvernorat", "x_key": "gouvernorat", "y_key": "total_impaye"},
        }

    if metric == "impaye_monthly_trend":
        return {
            "sql_id": "impaye_monthly_trend",
            "sql_query": """
                SELECT
                    make_date(annee_echeance, mois_echeance, 1) AS period,
                    COUNT(*) AS nb_impayes,
                    COALESCE(SUM(mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye
                WHERE (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                  AND annee_echeance BETWEEN 1900 AND 2100
                  AND mois_echeance BETWEEN 1 AND 12
                GROUP BY make_date(annee_echeance, mois_echeance, 1)
                ORDER BY period
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "line", "title": "Évolution mensuelle des impayés", "x_key": "period", "y_key": "total_impaye"},
        }

    if metric == "impaye_top_clients":
        return {
            "sql_id": "impaye_top_clients",
            "sql_query": f"""
                SELECT
                    COALESCE(c.prenom || ' ' || c.nom, c.nom, CAST(i.id_client AS VARCHAR)) AS client,
                    COUNT(i.num_quittance) AS nb_impayes,
                    COALESCE(SUM(i.mt_acp), 0) AS total_impaye
                FROM dwh_fact_impaye i
                JOIN dim_police p ON p.id_police = i.id_police
                JOIN dim_client c ON c.id_client = p.id_client
                WHERE (:branch IS NULL OR i.branche = :branch)
                  AND (:year_from IS NULL OR i.annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR i.annee_echeance <= :year_to)
                GROUP BY i.id_client, c.nom, c.prenom
                ORDER BY total_impaye DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "client", "label": "Client", "unit": ""},
                {"key": "nb_impayes", "label": "Impayés", "unit": "count"},
                {"key": "total_impaye", "label": "Montant impayé", "unit": "TND"},
            ],
            "chart": {"type": "bar", "title": "Top clients par montant impayé", "x_key": "client", "y_key": "total_impaye"},
        }

    if metric == "resiliation_by_agent":
        return {
            "sql_id": "resiliation_by_agent",
            "sql_query": f"""
                SELECT
                    COALESCE(a.nom_agent, CAST(p.id_agent AS VARCHAR)) AS agent,
                    COUNT(p.id_police) AS total_polices,
                    SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) AS nb_resiliees,
                    ROUND(
                        100.0 * SUM(CASE WHEN p.situation = 'R' THEN 1 ELSE 0 END) / COUNT(p.id_police), 2
                    ) AS taux_resiliation_pct
                FROM dim_police p
                LEFT JOIN dim_agent a ON a.id_agent = p.id_agent
                WHERE p.id_agent IS NOT NULL
                  AND (:branch IS NULL OR p.branche = :branch)
                GROUP BY p.id_agent, a.nom_agent
                ORDER BY nb_resiliees DESC
                LIMIT {limit_value}
            """,
            "params": params,
            "result_kind": "breakdown",
            "kpi_fields": [
                {"key": "agent", "label": "Agent", "unit": ""},
                {"key": "total_polices", "label": "Total polices", "unit": "count"},
                {"key": "nb_resiliees", "label": "Résiliées", "unit": "count"},
                {"key": "taux_resiliation_pct", "label": "Taux résiliation", "unit": "%"},
            ],
            "chart": {"type": "bar", "title": "Résiliation par agent", "x_key": "agent", "y_key": "nb_resiliees"},
        }

    if metric == "prime_yearly":
        return {
            "sql_id": "prime_yearly",
            "sql_query": """
                SELECT
                    annee_echeance AS annee,
                    COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                    COUNT(DISTINCT id_police) AS nb_polices,
                    COUNT(*) AS nb_quittances
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E', 'P', 'A')
                  AND (:branch IS NULL OR branche = :branch)
                  AND (:year_from IS NULL OR annee_echeance >= :year_from)
                  AND (:year_to IS NULL OR annee_echeance <= :year_to)
                  AND annee_echeance BETWEEN 1900 AND 2100
                GROUP BY annee_echeance
                ORDER BY annee_echeance
            """,
            "params": params,
            "result_kind": "timeseries",
            "chart": {"type": "bar", "title": "Prime nette par année", "x_key": "annee", "y_key": "total_pnet"},
        }

    return None


def _infer_sql_report_mode(question: str) -> str:
    return _infer_report_mode(question)


def _build_sql_query_spec(question: str, context: dict[str, Any]) -> dict[str, Any]:
    lowered = _normalize_text(question)
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    params = {
        "branch": branch,
        "year_from": year_from,
        "year_to": year_to,
    }

    semantic = _analyze_sql_request(lowered)
    semantic_spec = _build_semantic_sql_query_spec(semantic, params)
    if semantic_spec:
        semantic_spec["semantic"] = semantic
        return semantic_spec

    return {
        "sql_id": "top_branches_prime",
        "sql_query": """
            SELECT
                branche,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND (:branch IS NULL OR branche = :branch)
              AND annee_echeance BETWEEN :year_from AND :year_to
            GROUP BY branche
            ORDER BY total_pnet DESC
            LIMIT 10
        """,
        "params": params,
        "chart": {
            "type": "bar",
            "title": "Top branches par prime nette",
            "x_key": "branche",
            "y_key": "total_pnet",
        },
        "result_kind": "breakdown",
    }


def _prepare_sql_rows(df: pd.DataFrame, sql_id: str) -> list[dict[str, Any]]:
    working_df = df.copy()

    if "period" in working_df.columns:
        working_df["period"] = pd.to_datetime(working_df["period"], errors="coerce").dt.strftime("%Y-%m")

    rows = working_df.to_dict(orient="records")

    if sql_id == "total_clients" and rows:
        total_clients = _safe_int(rows[0].get("total_clients"), 0)
        rows = [{"label": "clients", "total_clients": total_clients}]

    if sql_id == "total_impayes_overview" and rows:
        nb_impayes = _safe_int(rows[0].get("nb_impayes"), 0)
        total_impaye = _safe_float(rows[0].get("total_impaye"), 0.0)
        avg_impaye = _safe_float(rows[0].get("avg_impaye"), 0.0)
        nb_polices_impactees = _safe_int(rows[0].get("nb_polices_impactees"), 0)
        rows = [
            {
                "label": "impayes",
                "nb_impayes": nb_impayes,
                "total_impaye": total_impaye,
                "avg_impaye": avg_impaye,
                "nb_polices_impactees": nb_polices_impactees,
            }
        ]

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                normalized_row[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                normalized_row[key] = value.strftime("%Y-%m-%d")
            else:

                normalized_row[key] = value
        normalized_rows.append(normalized_row)

    return normalized_rows


def _build_sql_kpis(rows: list[dict[str, Any]], query_spec: dict[str, Any]) -> list[dict[str, Any]]:
    result_kind = str(query_spec.get("result_kind", "tabular")).lower()
    if result_kind != "scalar" or not rows:
        return []

    first_row = rows[0]
    kpi_fields = query_spec.get("kpi_fields")
    kpis: list[dict[str, Any]] = []

    if isinstance(kpi_fields, list):
        for field in kpi_fields:
            if not isinstance(field, dict):
                continue
            key = str(field.get("key", "")).strip()
            if not key or key not in first_row:
                continue

            value = first_row.get(key)
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (int, float)):
                continue

            kpis.append(
                {
                    "key": key,
                    "label": str(field.get("label", key.replace("_", " ").title())),
                    "value": value,
                    "unit": str(field.get("unit", "")),
                }
            )

    if kpis:
        return kpis[:4]

    for key, value in first_row.items():
        if key == "label":
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            kpis.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "value": value,
                    "unit": "",
                }
            )

    return kpis[:4]


def _first_numeric_item(row: dict[str, Any], excluded_keys: set[str] | None = None) -> tuple[str | None, float]:
    excluded = excluded_keys or set()
    for key, value in row.items():
        if key in excluded:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            return key, _safe_float(value)
    return None, 0.0


def _first_dimension_value(row: dict[str, Any]) -> str:
    preferred_keys = ["branche", "gouvernorat", "id_client", "nom", "prenom", "label", "period"]
    for key in preferred_keys:
        if key in row and row.get(key) is not None:
            raw = str(row.get(key)).strip()
            if raw:
                return raw

    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "N/A"


def _format_period_label(year_from: int, year_to: int) -> str:
    if year_from == year_to:
        return f"{year_from}"
    return f"{year_from}-{year_to}"


def _build_sql_context_line(sql_id: str, branch: str | None, year_from: int, year_to: int) -> str:
    scope_label = _format_scope_label(branch)
    period_label = _format_period_label(year_from, year_to)

    if sql_id == "overall_resiliation_rate":
        return f"Vue globale de la resiliation sur {scope_label} (periode {period_label})."
    if sql_id == "branch_resiliation_rate":
        return f"Comparaison des taux de resiliation par branche (periode {period_label})."
    if sql_id == "total_impayes_overview":
        return f"Vue globale des impayes sur {scope_label} (periode {period_label})."
    if sql_id == "prime_overview":
        return f"Vue globale de la production sur {scope_label} (periode {period_label})."
    if sql_id.startswith("monthly_prime_") or sql_id.startswith("monthly_impaye_"):
        return f"Evolution mensuelle sur {scope_label} (periode {period_label})."
    if sql_id.startswith("impaye_by_branche_"):
        return f"Comparaison des impayes par branche (periode {period_label})."
    if sql_id.startswith("impaye_by_gouvernorat_"):
        return f"Comparaison des impayes par gouvernorat (periode {period_label})."
    if sql_id.startswith("top_clients_impaye_") or sql_id.startswith("impaye_by_client_"):
        return f"Classement des clients par impaye (periode {period_label})."
    if sql_id.startswith("prime_by_branche_"):
        return f"Comparaison de la production par branche (periode {period_label})."

    return f"Analyse metier sur {scope_label} (periode {period_label})."


def _build_sql_report_details(
    *,
    sql_id: str,
    rows: list[dict[str, Any]],
    branch: str | None,
    year_from: int,
    year_to: int,
) -> dict[str, Any]:
    context_line = _build_sql_context_line(sql_id, branch, year_from, year_to)

    if not rows:
        return {
            "context": context_line,
            "analysis": "Aucune ligne retournee par la requete sur ce perimetre.",
            "decision": "Impossible de conclure sans donnees; elargir le scope temporel ou metier.",
            "actions": [
                "Elargir la periode d analyse (annees supplementaires).",
                "Verifier les filtres de branche et la qualite du chargement des tables sources.",
            ],
        }

    if sql_id == "top_branches_prime":
        total_prime = sum(_safe_float(item.get("total_pnet"), 0.0) for item in rows)
        leader = rows[0]
        leader_branch = str(leader.get("branche", "N/A"))
        leader_prime = _safe_float(leader.get("total_pnet"), 0.0)
        leader_share = (100.0 * leader_prime / total_prime) if total_prime > 0 else 0.0
        concentration = "elevee" if leader_share >= 65.0 else "moderee"
        return {
            "context": context_line,
            "analysis": (
                f"Branche leader {leader_branch} avec {leader_prime:,.0f} TND, soit {leader_share:.2f}% du total analyse."
            ),
            "decision": f"Concentration portefeuille {concentration}; pilotage commercial a ajuster par branche.",
            "actions": [
                "Renforcer la production sur les branches sous-ponderees si l objectif est la diversification.",
                "Fixer un seuil de concentration cible par branche et le suivre mensuellement.",
            ],
        }

    if sql_id == "monthly_prime_trend" or sql_id.startswith("monthly_prime_") or sql_id.startswith("monthly_impaye_"):
        first_item = rows[0]
        last_item = rows[-1]
        metric_key = "total_pnet" if "total_pnet" in first_item else None
        if metric_key is None:
            metric_key, _ = _first_numeric_item(first_item, excluded_keys={"annee_echeance", "mois_echeance"})

        first_value = _safe_float(first_item.get(metric_key or "", 0.0), 0.0)
        last_value = _safe_float(last_item.get(metric_key or "", 0.0), 0.0)
        growth_pct = (100.0 * (last_value - first_value) / first_value) if first_value > 0 else 0.0
        trend = "haussiere" if growth_pct >= 3.0 else "baissiere" if growth_pct <= -3.0 else "stable"
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": (
                f"Evolution {metric_label} de {first_item.get('period')} a {last_item.get('period')}: {growth_pct:.2f}% ({trend})."
            ),
            "decision": f"Tendance {trend}; ajuster capacite commerciale et objectifs de production.",
            "actions": [
                "Comparer la trajectoire mensuelle aux objectifs budgetaires et replanifier si ecart > 5%.",
                "Identifier les mois de rupture et lancer une analyse causale par branche.",
            ],
        }

    if sql_id == "branch_impaye_ratio":
        worst = rows[0]
        worst_branch = str(worst.get("branche", "N/A"))
        worst_ratio = _safe_float(worst.get("impaye_ratio_pct"), 0.0)
        return {
            "context": context_line,
            "analysis": f"Branche la plus exposee aux impayes: {worst_branch} avec ratio {worst_ratio:.2f}%.",
            "decision": "Prioriser le recouvrement sur les branches a ratio impaye eleve.",
            "actions": [
                "Definir des plans de recouvrement differencies par branche selon le ratio impaye.",
                "Reviser les regles de souscription sur la branche la plus risquee.",
            ],
        }

    if sql_id == "total_impayes_overview":
        snapshot = rows[0]
        nb_impayes = _safe_int(snapshot.get("nb_impayes"), 0)
        total_impaye = _safe_float(snapshot.get("total_impaye"), 0.0)
        avg_impaye = _safe_float(snapshot.get("avg_impaye"), 0.0)
        nb_polices_impactees = _safe_int(snapshot.get("nb_polices_impactees"), 0)
        return {
            "context": context_line,
            "analysis": (
                f"Stock global des impayes: {nb_impayes:,} impayes, montant cumule {total_impaye:,.0f} TND, "
                f"moyenne {avg_impaye:,.0f} TND, "
                f"touchant {nb_polices_impactees:,} polices."
            ),
            "decision": "Le volume impaye global justifie un pilotage recouvrement par priorite de montant et anciennete.",
            "actions": [
                "Segmenter le stock impaye par anciennete et lancer des vagues de recouvrement ciblees.",
                "Suivre mensuellement la reduction du nombre d impayes et du montant cumule.",
            ],
        }

    if sql_id == "prime_overview":
        snapshot = rows[0]
        total_pnet = _safe_float(snapshot.get("total_pnet"), 0.0)
        avg_pnet = _safe_float(snapshot.get("avg_pnet"), 0.0)
        nb_quittances = _safe_int(snapshot.get("nb_quittances"), 0)
        return {
            "context": context_line,
            "analysis": (
                f"Production globale: total {total_pnet:,.0f} TND, moyenne {avg_pnet:,.0f} TND sur "
                f"{nb_quittances:,} quittances."
            ),
            "decision": "Piloter la production avec un suivi conjoint du volume et de la valeur moyenne par quittance.",
            "actions": [
                "Comparer la valeur moyenne aux objectifs budgetaires par branche.",
                "Suivre mensuellement le couple volume-valeur pour prevenir la derive commerciale.",
            ],
        }

    if sql_id == "overall_resiliation_rate":
        snapshot = rows[0]
        total_polices = _safe_int(snapshot.get("total_polices"), 0)
        nb_resiliees = _safe_int(snapshot.get("nb_resiliees"), 0)
        taux_resiliation = _safe_float(snapshot.get("taux_resiliation_pct"), 0.0)
        return {
            "context": context_line,
            "analysis": (
                f"Resiliation globale: {nb_resiliees:,} polices resiliees sur {total_polices:,}, "
                f"soit {taux_resiliation:.2f}%."
            ),
            "decision": "Le taux de resiliation global doit etre compare a la cible retention et segmente par branche.",
            "actions": [
                "Lancer des actions retention sur les segments les plus contributes a la resiliation.",
                "Suivre l evolution mensuelle du taux pour mesurer l effet des plans d action.",
            ],
        }

    if sql_id == "branch_resiliation_rate":
        worst = rows[0]
        worst_branch = str(worst.get("branche", "N/A"))
        worst_rate = _safe_float(worst.get("taux_resiliation_pct"), 0.0)
        return {
            "context": context_line,
            "analysis": f"Taux de resiliation le plus eleve: {worst_branch} a {worst_rate:.2f}%.",
            "decision": "Risque retention cible par branche; intervention proactive requise.",
            "actions": [
                "Lancer un plan de retention sur les branches en tete de resiliation.",
                "Analyser les motifs d annulation pour corriger les causes recurrentes.",
            ],
        }

    if sql_id == "top_governorates_by_impaye" or sql_id.startswith("impaye_by_gouvernorat_"):
        first = rows[0]
        gov = _first_dimension_value(first)
        metric_key, metric_value = _first_numeric_item(first)
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": f"Gouvernorat prioritaire {gov} avec {metric_label} {metric_value:,.0f}.",
            "decision": "Concentration geographique du risque impaye; prioriser les equipes terrain.",
            "actions": [
                "Affecter les actions recouvrement en priorite sur les zones geographiques les plus exposees.",
                "Suivre hebdomadairement la baisse du stock impaye sur le gouvernorat leader.",
            ],
        }

    if sql_id == "top_clients_impaye" or sql_id.startswith("top_clients_impaye_") or sql_id.startswith("impaye_by_client_"):
        first = rows[0]
        client_label = f"{first.get('nom', 'N/A')} {first.get('prenom', '')}".strip()
        metric_key, metric_value = _first_numeric_item(first)
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": f"Client le plus expose: {client_label} avec {metric_label} {metric_value:,.0f}.",
            "decision": "Pilotage recouvrement cible sur les clients les plus materialites.",
            "actions": [
                "Mettre en priorite les dossiers top impayes dans le plan de recouvrement.",
                "Segmenter ces clients par anciennete et risque de defaut pour adapter la strategie.",
            ],
        }

    if sql_id.startswith("impaye_by_branche_") or sql_id.startswith("prime_by_branche_"):
        leader = rows[0]
        dimension_value = _first_dimension_value(leader)
        metric_key, metric_value = _first_numeric_item(leader)
        metric_label = str(metric_key or "metrique").replace("_", " ")
        return {
            "context": context_line,
            "analysis": f"Leader sur la repartition: {dimension_value} avec {metric_label} {metric_value:,.0f}.",
            "decision": "Concentrer les actions de pilotage sur les dimensions en tete et surveiller la concentration.",
            "actions": [
                "Suivre mensuellement le top des dimensions pour detecter les changements de structure.",
                "Definir des objectifs de reequilibrage si la concentration depasse le seuil cible.",
            ],
        }

    if sql_id == "total_clients":
        total_clients = _safe_int(rows[0].get("total_clients"), 0)
        return {
            "context": context_line,
            "analysis": f"Volume portefeuille clients: {total_clients:,} clients.",
            "decision": "Utiliser cette base comme referentiel de penetration des offres et retention.",
            "actions": [
                "Suivre le taux d activation et de retention sur ce referentiel clients.",
                "Croiser avec impayes/resiliation pour prioriser les segments a valeur.",
            ],
        }

    return {
        "context": context_line,
        "analysis": f"{len(rows)} lignes exploitables retournees.",
        "decision": "Resultat SQL disponible pour pilotage metier.",
        "actions": ["Exploiter ce resultat dans un cycle de suivi mensuel."]
    }


def sql_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    report_mode = _infer_sql_report_mode(question)
    query_spec = _build_sql_query_spec(question, context)

    sql_id = str(query_spec["sql_id"])
    sql_query = str(query_spec["sql_query"])
    params = dict(query_spec.get("params", {}))

    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    df = _query_dataframe(sql_query, params)
    rows = _prepare_sql_rows(df, sql_id)

    report_details = _build_sql_report_details(
        sql_id=sql_id,
        rows=rows,
        branch=branch,
        year_from=year_from,
        year_to=year_to,
    )

    result_kind = str(query_spec.get("result_kind", "tabular")).lower()
    kpis = _build_sql_kpis(rows, query_spec)
    is_scalar_result = result_kind == "scalar" and len(rows) == 1

    chart_spec = query_spec.get("chart", {}) if isinstance(query_spec.get("chart"), dict) else {}
    include_chart = report_mode in {"report", "graph_only", "graph_pref"} and not is_scalar_result
    include_table = report_mode in {"report", "table_only", "table_pref", "graph_pref"}

    # Scalar outputs are displayed as KPI cards in UI by default.
    if is_scalar_result and report_mode in {"report", "graph_only", "graph_pref"}:
        include_table = False

    charts: list[dict[str, Any]] = []
    if include_chart and rows and chart_spec:
        chart_items = rows[:24]
        charts.append(
            _build_chart_payload(
                chart_type=str(chart_spec.get("type", "bar")),
                title=str(chart_spec.get("title", f"SQL chart {sql_id}")),
                x_key=str(chart_spec.get("x_key", list(chart_items[0].keys())[0])),
                y_key=str(chart_spec.get("y_key", list(chart_items[0].keys())[1] if len(chart_items[0].keys()) > 1 else list(chart_items[0].keys())[0])),
                items=chart_items,
            )
        )

    tables: list[dict[str, Any]] = []
    if include_table:
        table_columns = list(rows[0].keys()) if rows else []
        tables.append(
            {
                "title": f"SQL result {sql_id}",
                "columns": table_columns,
                "rows": rows,
                "markdown": _to_markdown_table(table_columns, rows),
            }
        )

    period_label = _format_period_label(year_from, year_to)
    scope_label = _format_scope_label(branch)
    summary = f"{report_details['analysis']} ({scope_label}, periode {period_label})."

    return {
        "tool": "sql_tool",
        "summary": summary,
        "payload": {
            "sql_id": sql_id,
            "rows": rows,
            "branch": branch or "ALL",
            "year_from": year_from,
            "year_to": year_to,
            "report_mode": report_mode,
            "result_kind": result_kind,
            "kpis": kpis,
            "semantic": query_spec.get("semantic", {}),
            "context": report_details.get("context", ""),
            "analysis": report_details.get("analysis", ""),
            "decision": report_details.get("decision", ""),
            "actions": report_details.get("actions", []),
        },
        "charts": charts,
        "tables": tables,
    }


def data_query_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    lowered = question.lower()
    should_use_sql = (
        "sql" in lowered
        or "requete" in lowered
        or "gouvernorat" in lowered
        or ("total" in lowered and "client" in lowered)
    )
    if should_use_sql:
        return sql_tool(question=question, context=context)
    return kpi_tool(question=question, context=context)


