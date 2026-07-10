from __future__ import annotations

from typing import Any

import requests

from utils import normalize_text as _normalize_text
from config import NEXTJS_API_URL
from tools._shared import _normalize_branch, _resolve_period_context, _build_chart_payload

def dim_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Interroge l'API Next.js /api/dims pour les dimensions (clients, agents, produits, véhicules, polices, sinistres)
    """
    import requests
    
    # URL de l'API Next.js
    nextjs_api = NEXTJS_API_URL
    
    # Construire les paramètres
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    
    params = {}
    if branch:
        params["branch"] = branch
    if year_from:
        params["year_from"] = year_from
    if year_to:
        params["year_to"] = year_to
    
    try:
        response = requests.get(f"{nextjs_api}/api/dims", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {
            "tool": "dim_tool",
            "summary": f"Erreur d'accès aux données dimensions: {str(exc)}",
            "payload": {"error": str(exc)},
            "charts": [],
            "tables": [],
        }
    
    normalized = _normalize_text(question)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIENTS
    # ═══════════════════════════════════════════════════════════════════════
    if "client" in normalized or "assure" in normalized:
        clients_data = data.get("clients", {})
        kpis = clients_data.get("kpis", {})
        
        # Répartition par sexe
        if any(token in normalized for token in ("sexe", "femme", "femmes", "homme", "hommes")):
            rows = clients_data.get("sexe", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des clients par sexe: {kpis.get('nb_f', 0)} femmes ({kpis.get('pct_f', 0)}%), {kpis.get('nb_m', 0)} hommes ({kpis.get('pct_m', 0)}%)",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "nb_femmes", "label": "Femmes", "value": kpis.get("nb_f", 0), "unit": "count"},
                    {"key": "nb_hommes", "label": "Hommes", "value": kpis.get("nb_m", 0), "unit": "count"},
                    {"key": "pct_f", "label": "Pourcentage femmes", "value": kpis.get("pct_f", 0), "unit": "%"},
                    {"key": "pct_m", "label": "Pourcentage hommes", "value": kpis.get("pct_m", 0), "unit": "%"},
                ]},
                "charts": [_build_chart_payload("pie", "Répartition par sexe", "label", "count", rows)],
                "tables": [{"title": "Clients par sexe", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Âge moyen
        if "age" in normalized:
            age_tranches = clients_data.get("ageTranches", [])
            return {
                "tool": "dim_tool",
                "summary": f"Âge moyen des clients: {kpis.get('age_moyen', 0)} ans",
                "payload": {"result_kind": "scalar", "rows": [{"age_moyen": kpis.get("age_moyen", 0)}], "kpis": [
                    {"key": "age_moyen", "label": "Âge moyen", "value": kpis.get("age_moyen", 0), "unit": "ans"},
                ]},
                "charts": [_build_chart_payload("bar", "Tranches d'âge", "label", "count", age_tranches)] if age_tranches else [],
                "tables": [],
            }
        
        # Type de personne
        if "type personne" in normalized or "personne physique" in normalized or "personne morale" in normalized:
            rows = clients_data.get("typePersonne", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par type de personne: {kpis.get('pct_moral', 0)}% personnes morales",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "pct_moral", "label": "Personnes morales", "value": kpis.get("pct_moral", 0), "unit": "%"},
                ]},
                "charts": [_build_chart_payload("pie", "Type de personne", "label", "count", rows)],
                "tables": [{"title": "Type de personne", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Nationalité
        if "nationalite" in normalized or "natp" in normalized:
            rows = clients_data.get("natp", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par nationalité",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Nationalité", "label", "count", rows)],
                "tables": [{"title": "Nationalité", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Top villes
        if "ville" in normalized:
            rows = clients_data.get("topVilles", [])
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} villes par concentration clients",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "nb_villes", "label": "Villes couvertes", "value": kpis.get("nb_villes", 0), "unit": "count"},
                ]},
                "charts": [_build_chart_payload("bar", "Top villes", "label", "count", rows)],
                "tables": [{"title": "Top villes", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Vue globale clients
        return {
            "tool": "dim_tool",
            "summary": f"Total clients: {kpis.get('total', 0)}, {kpis.get('pct_f', 0)}% femmes, {kpis.get('age_moyen', 0)} ans moy.",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total clients", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "nb_f", "label": "Femmes", "value": kpis.get("nb_f", 0), "unit": "count"},
                {"key": "nb_m", "label": "Hommes", "value": kpis.get("nb_m", 0), "unit": "count"},
                {"key": "pct_moral", "label": "Personnes morales", "value": kpis.get("pct_moral", 0), "unit": "%"},
                {"key": "nb_villes", "label": "Villes couvertes", "value": kpis.get("nb_villes", 0), "unit": "count"},
                {"key": "age_moyen", "label": "Âge moyen", "value": kpis.get("age_moyen", 0), "unit": "ans"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    if "agent" in normalized:
        agents_data = data.get("agents", {})
        kpis = agents_data.get("kpis", {})
        
        # État des agents
        if "etat" in normalized:
            rows = agents_data.get("etat", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des agents par état: {kpis.get('actifs', 0)} actifs, {kpis.get('inactifs', 0)} inactifs",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "État des agents", "label", "count", rows)],
                "tables": [{"title": "État des agents", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Groupe
        if "groupe" in normalized:
            rows = agents_data.get("groupes", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des agents par groupe",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Agents par groupe", "label", "count", rows)],
                "tables": [{"title": "Agents par groupe", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Type d'agent
        if "type" in normalized:
            rows = agents_data.get("typeAgent", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition des agents par type",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Type d'agent", "label", "count", rows)],
                "tables": [{"title": "Type d'agent", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Localités
        if "localite" in normalized or "ville" in normalized:
            rows = agents_data.get("localites", [])[:10]
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} localités agents",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Top localités agents", "label", "count", rows)],
                "tables": [{"title": "Top localités", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Vue globale agents
        return {
            "tool": "dim_tool",
            "summary": f"Total agents: {kpis.get('total', 0)}, actifs: {kpis.get('actifs', 0)}",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total agents", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "actifs", "label": "Agents actifs", "value": kpis.get("actifs", 0), "unit": "count"},
                {"key": "inactifs", "label": "Agents inactifs", "value": kpis.get("inactifs", 0), "unit": "count"},
                {"key": "nb_groupes", "label": "Groupes", "value": kpis.get("nb_groupes", 0), "unit": "count"},
                {"key": "nb_localites", "label": "Localités", "value": kpis.get("nb_localites", 0), "unit": "count"},
                {"key": "avg_pnet", "label": "Prime nette moyenne", "value": kpis.get("avg_pnet", 0), "unit": "TND"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # FAMILLES DE RISQUE (standalone, no "produit" required)
    # ═══════════════════════════════════════════════════════════════════════

    if "famille" in normalized or "familles" in normalized:
        produits_data = data.get("produits", {})
        if "familles" in normalized and "nombre" in normalized:
            # Question: "nombre de familles de risque"
            kpis = produits_data.get("kpis", {})
            return {
                "tool": "dim_tool",
                "summary": f"Nombre de familles de risque: {kpis.get('nb_familles', 0)}",
                "payload": {
                    "result_kind": "scalar",
                    "rows": [{"nb_familles": kpis.get("nb_familles", 0)}],
                    "kpis": [{"key": "nb_familles", "label": "Familles de risque", "value": kpis.get("nb_familles", 0), "unit": "count"}]
                },
                "charts": [],
                "tables": [],
            }
        else:
            rows = produits_data.get("byFamille", [])
            return {
                "tool": "dim_tool",
                "summary": f"Prime nette par famille de risque",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Familles de risque", "label", "pnet", rows)],
                "tables": [{"title": "Familles de risque", "columns": ["label", "pnet"], "rows": rows}],
            }

    # ═══════════════════════════════════════════════════════════════════════
    # PRODUITS
    # ═══════════════════════════════════════════════════════════════════════
    if "produit" in normalized:
        produits_data = data.get("produits", {})
        kpis = produits_data.get("kpis", {})
        
        # Nombre de produits distincts
        if "nombre de produits" in normalized or "distinct" in normalized:
            return {
                "tool": "dim_tool",
                "summary": f"Produits distincts: {kpis.get('nb_produits', 0)}, Familles: {kpis.get('nb_familles', 0)}, Branches: {kpis.get('nb_branches', 0)}",
                "payload": {
                    "result_kind": "scalar",
                    "rows": [kpis],
                    "kpis": [
                        {"key": "nb_produits", "label": "Produits distincts", "value": kpis.get("nb_produits", 0), "unit": "count"},
                        {"key": "nb_familles", "label": "Familles de risque", "value": kpis.get("nb_familles", 0), "unit": "count"},
                        {"key": "nb_branches", "label": "Branches couvertes", "value": kpis.get("nb_branches", 0), "unit": "count"},
                        {"key": "total_pnet", "label": "Prime nette totale", "value": kpis.get("total_pnet", 0), "unit": "TND"},
                    ],
                },
                "charts": [],
                "tables": [],
            }
        
        # Familles de risque
        if "famille" in normalized:
            rows = produits_data.get("byFamille", [])
            return {
                "tool": "dim_tool",
                "summary": f"Prime nette par famille de risque",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Familles de risque", "label", "pnet", rows)],
                "tables": [{"title": "Familles de risque", "columns": ["label", "pnet"], "rows": rows}],
            }
        
        # Top produits
        if "top" in normalized:
            rows = produits_data.get("topProduits", [])
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} produits par prime nette",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Top produits", "label", "pnet", rows)],
                "tables": [{"title": "Top produits", "columns": ["label", "pnet"], "rows": rows}],
            }
    
    
    # ═══════════════════════════════════════════════════════════════════════
    # VÉHICULES
    # ═══════════════════════════════════════════════════════════════════════
    if "vehicule" in normalized or "voiture" in normalized:
        vehicules_data = data.get("vehicules", {})
        kpis = vehicules_data.get("kpis", {})
        
        # Top marques
        if "marque" in normalized:
            rows = vehicules_data.get("topMarques", [])
            return {
                "tool": "dim_tool",
                "summary": f"Top {len(rows)} marques par parc assuré",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Top marques", "label", "count", rows)],
                "tables": [{"title": "Top marques", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Genre
        if "genre" in normalized:
            rows = vehicules_data.get("byGenre", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par genre de véhicule",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Genre de véhicule", "label", "count", rows)],
                "tables": [{"title": "Genre de véhicule", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Puissance
        if "puissance" in normalized:
            rows = vehicules_data.get("byPuissance", [])
            return {
                "tool": "dim_tool",
                "summary": f"Puissance moyenne: {kpis.get('avg_puissance', 0)} CV",
                "payload": {"result_kind": "scalar", "rows": [{"puissance_moyenne": kpis.get("avg_puissance", 0)}], "kpis": [
                    {"key": "avg_puissance", "label": "Puissance moyenne", "value": kpis.get("avg_puissance", 0), "unit": "CV"},
                ]},
                "charts": [_build_chart_payload("bar", "Puissance fiscale", "label", "count", rows)] if rows else [],
                "tables": [],
            }
        
        # Vue globale véhicules
        return {
            "tool": "dim_tool",
            "summary": f"Total véhicules: {kpis.get('total', 0)}, Puissance moyenne: {kpis.get('avg_puissance', 0)} CV",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total véhicules", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "nb_vp", "label": "Voitures particulières", "value": kpis.get("nb_vp", 0), "unit": "count"},
                {"key": "nb_vu", "label": "Véhicules utilitaires", "value": kpis.get("nb_vu", 0), "unit": "count"},
                {"key": "nb_marques", "label": "Marques distinctes", "value": kpis.get("nb_marques", 0), "unit": "count"},
                {"key": "avg_puissance", "label": "Puissance moyenne", "value": kpis.get("avg_puissance", 0), "unit": "CV"},
                {"key": "avg_age", "label": "Âge moyen", "value": kpis.get("avg_age", 0), "unit": "ans"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # POLICES
    # ═══════════════════════════════════════════════════════════════════════
    if "police" in normalized:
        polices_data = data.get("polices", {})
        kpis = polices_data.get("kpis", {})
        
        # Situation
        if "situation" in normalized:
            rows = polices_data.get("bySituation", [])
            return {
                "tool": "dim_tool",
                "summary": f"Situation du portefeuille: {kpis.get('en_vigueur', 0)} polices en vigueur",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Situation du portefeuille", "label", "count", rows)],
                "tables": [{"title": "Situation", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Périodicité
        if "periodicite" in normalized:
            rows = polices_data.get("byPeriodicite", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par périodicité",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Périodicité", "label", "count", rows)],
                "tables": [{"title": "Périodicité", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Type de police
        if "type" in normalized:
            rows = polices_data.get("byType", [])
            return {
                "tool": "dim_tool",
                "summary": f"Répartition par type de police: {kpis.get('pct_indiv', 0)}% individuelles, {kpis.get('pct_flotte', 0)}% flotte",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Type de police", "label", "count", rows)],
                "tables": [{"title": "Type de police", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Bonus-Malus
        if "bonus malus" in normalized:
            rows = polices_data.get("bonusMalus", [])
            return {
                "tool": "dim_tool",
                "summary": f"Distribution Bonus-Malus, moyenne: {kpis.get('avg_bm', 0)}",
                "payload": {"result_kind": "breakdown", "rows": rows, "kpis": [
                    {"key": "avg_bm", "label": "Bonus-Malus moyen", "value": kpis.get("avg_bm", 0), "unit": ""},
                ]},
                "charts": [_build_chart_payload("bar", "Distribution Bonus-Malus", "label", "count", rows)],
                "tables": [{"title": "Bonus-Malus", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Vue globale polices
        return {
            "tool": "dim_tool",
            "summary": f"Total polices: {kpis.get('total', 0)}, En vigueur: {kpis.get('en_vigueur', 0)}",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total polices", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "en_vigueur", "label": "En vigueur", "value": kpis.get("en_vigueur", 0), "unit": "count"},
                {"key": "resiliees", "label": "Résiliées", "value": kpis.get("resiliees", 0), "unit": "count"},
                {"key": "individuelles", "label": "Individuelles", "value": kpis.get("individuelles", 0), "unit": "count"},
                {"key": "flottes", "label": "Flotte", "value": kpis.get("flottes", 0), "unit": "count"},
                {"key": "avg_bm", "label": "BM moyen", "value": kpis.get("avg_bm", 0), "unit": ""},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SINISTRES
    # ═══════════════════════════════════════════════════════════════════════
    if "sinistre" in normalized:
        sinistres_data = data.get("sinistres", {})
        kpis = sinistres_data.get("kpis", {})
        
        # Par nature
        if "nature" in normalized:
            rows = sinistres_data.get("byNature", [])
            return {
                "tool": "dim_tool",
                "summary": f"Sinistres par nature",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Sinistres par nature", "label", "count", rows)],
                "tables": [{"title": "Sinistres par nature", "columns": ["label", "count"], "rows": rows}],
            }
        
        # État
        if "etat" in normalized:
            rows = sinistres_data.get("byEtat", [])
            return {
                "tool": "dim_tool",
                "summary": f"État des sinistres: {kpis.get('ouverts', 0)} ouverts, {kpis.get('clos', 0)} clos",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "État des sinistres", "label", "count", rows)],
                "tables": [{"title": "État des sinistres", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Responsabilité
        if "responsabilite" in normalized:
            rows = sinistres_data.get("byResponsabilite", [])
            return {
                "tool": "dim_tool",
                "summary": f"Responsabilité engagée",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("pie", "Responsabilité engagée", "label", "count", rows)],
                "tables": [{"title": "Responsabilité", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Par branche
        if "branche" in normalized:
            rows = sinistres_data.get("byBranche", [])
            return {
                "tool": "dim_tool",
                "summary": f"Sinistres par branche",
                "payload": {"result_kind": "breakdown", "rows": rows},
                "charts": [_build_chart_payload("bar", "Sinistres par branche", "label", "count", rows)],
                "tables": [{"title": "Sinistres par branche", "columns": ["label", "count"], "rows": rows}],
            }
        
        # Évolution mensuelle
        if "evolution" in normalized or "mensuel" in normalized:
            rows = sinistres_data.get("monthly", [])
            return {
                "tool": "dim_tool",
                "summary": f"Évolution mensuelle des sinistres",
                "payload": {"result_kind": "timeseries", "rows": rows},
                "charts": [_build_chart_payload("line", "Évolution mensuelle sinistres", "label", "count", rows)],
                "tables": [],
            }
        
        # Vue globale sinistres
        return {
            "tool": "dim_tool",
            "summary": f"Total sinistres: {kpis.get('total', 0)}, Ouverts: {kpis.get('ouverts', 0)}, Montant payé: {kpis.get('total_paye', 0):,.0f} TND",
            "payload": {"result_kind": "scalar", "rows": [kpis], "kpis": [
                {"key": "total", "label": "Total sinistres", "value": kpis.get("total", 0), "unit": "count"},
                {"key": "ouverts", "label": "Ouverts", "value": kpis.get("ouverts", 0), "unit": "count"},
                {"key": "clos", "label": "Clos", "value": kpis.get("clos", 0), "unit": "count"},
                {"key": "total_eval", "label": "Montant évalué", "value": kpis.get("total_eval", 0), "unit": "TND"},
                {"key": "total_paye", "label": "Montant payé", "value": kpis.get("total_paye", 0), "unit": "TND"},
                {"key": "nb_materiel", "label": "Sinistres matériels", "value": kpis.get("nb_materiel", 0), "unit": "count"},
            ]},
            "charts": [],
            "tables": [],
        }
    
    # Fallback
    return {
        "tool": "dim_tool",
        "summary": "Je peux vous fournir des informations sur les clients, agents, produits, véhicules, polices et sinistres. Que souhaitez-vous exactement ?",
        "payload": {"result_kind": "text", "message": "Spécifiez la dimension (clients, agents, produits, véhicules, polices, sinistres) et l'information souhaitée."},
        "charts": [],
        "tables": [],
    }

