from __future__ import annotations

import asyncio
import json
import os
import re
from contextvars import ContextVar
from typing import Any, Callable, TypedDict

import requests
from langgraph.graph import END, StateGraph

from agent_tools import run_tool
from config import (
    OLLAMA_HOST, OLLAMA_CHAT_MODEL, OLLAMA_TIMEOUT_SECONDS,
    AGENT_INTENT_MIN_CONFIDENCE, AGENT_LLM_MIN_CONFIDENCE,
    AGENT_LOW_CONFIDENCE_POLICY, AGENT_FORCE_DETERMINISTIC,
)
from utils import (
    safe_float as _as_float,
    normalize_text as _normalize_text,
    format_amount_tnd as _format_amount_tnd,
    format_percent as _format_percent,
    format_metric_value as _format_metric_value,
    format_branch_label as _format_branch_label,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Per-request progress callback (set by the streaming endpoint).
# Thread-safe: run_in_executor copies the context, so the callback set in the
# async generator is visible inside the sync worker thread.
_progress_cb: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "_progress_cb", default=None
)


def _emit(event: dict[str, Any]) -> None:
    cb = _progress_cb.get(None)
    if cb is not None:
        cb(event)


OUT_OF_SCOPE_MESSAGE = (
    "Question hors scope assurance/data. Je peux aider sur KPI, risques, impayes, "
    "clients, forecast, anomalies, drift ou analyse SQL metier."
)

TECHNICAL_LIMIT_MESSAGE = (
    "Limitation technique: un outil requis n'a pas pu etre execute correctement. "
    "Je fournis un resultat partiel et les details de limitation."
)

DOMAIN_KEYWORDS = {
    # Core insurance terms
    "assurance", "assure", "assures", "assureur",
    "police", "polices",
    "prime", "primes", "pnet", "ptt",
    "commission", "commissions",
    "impaye", "impayes",
    "resiliation", "resilie", "resiliee", "churn",
    "sinistre", "sinistres",
    "quittance", "quittances",
    "cotisation", "couverture",
    "garantie", "garanties",
    "contrat", "contrats",
    "souscription", "recouvrement", "annulation",
    # Dimensions / entities
    "agent", "agents",
    "branche", "branches",
    "auto", "irds", "sante",
    "client", "clients",
    "vehicule", "vehicules",
    "voiture", "voitures",
    "marque", "marques",
    "produit", "produits",
    "usage", "portefeuille",
    "periode", "periodes",
    "gouvernorat", "gouvernorats",
    "localite", "ville", "villes", "region", "zone",
    # KPI terms
    "kpi", "kpis",
    "ratio", "taux", "total", "montant", "nombre", "nb",
    "moyenne", "somme", "volume",
    # Analytics / ML
    "drift", "derive", "anomal", "anomalie",
    "forecast", "prevision", "previsionnel", "projection",
    "risque", "fraude", "segment", "segmentation",
    "model", "modele", "ml", "rag",
    "stabilite", "degradation", "alerte",
    # Reporting intent
    "global", "situation", "synthese",
    "tableau de bord", "dashboard",
    "bilan", "rapport",
    "analyse", "analyser", "comparaison",
    "repartition", "distribution",
    "evolution", "tendance", "trend",
    "classement", "top",
    "performance", "productivite",
    "sql", "requete", "donnees",
    # Tunisian insurance specifics
    "maghrebia", "bonus", "malus",
    "flotte", "individuel", "responsabilite",
    "periodicite", "nature", "etat",
    "emis", "emises", "emise",
    "tranche", "tranches",
    "provisions", "provision",
    "responsabilite",
    "sexe", "femme", "femmes", "homme", "hommes",
    "produits distincts", "familles",
    # Regulatory bodies & acronyms
    "fga", "cga", "cnam", "cnss", "bipar", "iard",
    "arrete", "decret", "loi", "circulaire", "code",
    "reglementaire", "reglementation", "regle", "regles",
    # Financial / actuarial terms
    "tarif", "tarification", "tarifs",
    "bareme", "baremes",
    "franchise", "franchises",
    "solvabilite", "marge",
    "indemnisation", "indemnite", "indemnites",
    "expertise", "expert", "experts",
    "norme", "normes",
    "calcul", "calcule", "calculer",
    "definition", "defini", "signifie",
    # Definition-question anchors
    "comment", "pourquoi", "qu'est-ce", "c'est quoi",
    "expliquer", "explication",
}

INTENT_MIN_CONFIDENCE = AGENT_INTENT_MIN_CONFIDENCE
LLM_MIN_CONFIDENCE    = AGENT_LLM_MIN_CONFIDENCE
LOW_CONFIDENCE_POLICY = AGENT_LOW_CONFIDENCE_POLICY
FORCE_DETERMINISTIC   = AGENT_FORCE_DETERMINISTIC
DETERMINISTIC_INTENTS = {
    item.strip().lower()
    for item in os.getenv("AGENT_DETERMINISTIC_INTENTS", "").split(",")
    if item.strip()
}
LLM_MIN_ANSWER_CHARS = _env_int("AGENT_LLM_MIN_ANSWER_CHARS", 90)

TOOL_FAMILY_TO_TOOL = {
    "data_query": "data_query_tool",
    "kpi": "kpi_tool",
    "rag": "rag_tool",
    "alerte": "alerte_tool",
    "forecast": "forecast_tool",
    "anomaly": "anomaly_tool",
    "drift": "drift_tool",
    "explain": "explain_tool",
    "segmentation": "segmentation_tool",
    "client": "client_tool",
    "predict": "ml_predict_tool",
    "sql": "sql_tool",
    "dimension": "dim_tool",
}

TOOL_SPECIALIST_AGENTS = {
    "data_query_tool": "data_query_specialist",
    "kpi_tool": "kpi_specialist",
    "rag_tool": "rag_specialist",
    "alerte_tool": "alerte_specialist",
    "forecast_tool": "forecast_specialist",
    "anomaly_tool": "anomaly_specialist",
    "drift_tool": "drift_specialist",
    "explain_tool": "explain_specialist",
    "segmentation_tool": "segmentation_specialist",
    "client_tool": "client_specialist",
    "ml_predict_tool": "ml_specialist",
    "sql_tool": "sql_specialist",
    "dim_tool": "dim_specialist",
}

INTENT_RULES: list[dict[str, Any]] = [
    {
        "intent": "explain",
        "keywords": ["explain", "explic", "shap", "pourquoi", "facteur"],
        "tool_families": ["explain", "kpi", "rag"],
        "required": ["explain"],
        "skip_llm": False,
    },
    {
        "intent": "predict",
        "keywords": ["fraude", "modele ml", "ml predict", "random forest", "gradient boosting", "ml model", "machine learning", "fais tourner le modele", "lancer le modele", "modelisation", "modele de prediction"],
        "tool_families": ["predict", "kpi"],
        "required": ["predict"],
        "skip_llm": False,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSION 
    # ═══════════════════════════════════════════════════════════════════════
    {
        "intent": "dimension",
        "keywords": [
            # Clients
            "client", "clients", "assure", "assures",
            "sexe", "femme", "femmes", "homme", "hommes",
            "age moyen", "âge moyen", "tranche d'age", "tranche âge",
            "type personne", "personne physique", "personne morale",
            "natp", "nationalite",
            "ville client", "top villes",
            # Agents
            "agent", "agents", "reseau",
            "etat agent", "type agent", "groupe agent",
            "localite agent", "top localites",
            # Produits
            "produit", "produits", "catalogue",
            "famille de risque",
            "familles de risque", 
            "nombre de familles",
            "prime nette par famille",
            "par famille de risque",
            "nombre de produits", "produits distincts", "produit distinct", "nombre de produit"
            "top produit", "top 10 produits", "prime nette par famille",
            # Véhicules
            "vehicule", "vehicules", "voiture", "voitures",
            "marque", "marques", "genre vehicule", "puissance",
            # Polices
            "police", "polices", "portefeuille",
            "situation police", "periodicite", "bonus malus",
            # Sinistres
            "sinistre", "sinistres",
            "nature sinistre", "etat sinistre", "responsabilite"
        ],
        "tool_families": ["dimension"],
        "required": ["dimension"],
        "skip_llm": True,
    },
    {
        "intent": "overview",
        "keywords": [
            "toutes les infos",
            "toutes les informations",
            "vue globale",
            "synthese complete",
            "synthese decisionnelle",
            "etat global",
            "situation globale",
            "diagnostic complet",
            "tableau de bord complet",
        ],
        "tool_families": ["kpi", "forecast", "anomaly", "drift", "alerte", "client", "segmentation", "sql", "rag"],
        "required": ["kpi"],
        "skip_llm": False,
    },
    {
        "intent": "forecast",
        "keywords": [
            "forecast",
            "prevoir",
            "prevois",
            "prevision",
            "previsionnel",
            "predire",
            "projection",
            "futur",
            "future",
            "horizon",
            "prochain mois",
            "next month",
        ],
        "tool_families": ["forecast"],
        "required": ["forecast"],
        "skip_llm": False,
    },
    {
        "intent": "anomaly",
        "keywords": ["anomal", "outlier", "pic", "rupture", "inhabituel"],
        "tool_families": ["anomaly", "alerte", "kpi"],
        "required": ["anomaly"],
        "skip_llm": False,
    },
    {
        "intent": "drift",
        "keywords": ["drift", "derive", "distribution", "stabilite", "degradation"],
        "tool_families": ["drift", "explain", "kpi"],
        "required": ["drift"],
        "skip_llm": False,
    },
    {
        "intent": "segmentation",
        "keywords": ["segment", "cluster", "profil", "persona", "portefeuille"],
        "tool_families": ["segmentation", "kpi"],
        "required": ["segmentation"],
        "skip_llm": False,
    },
    {
        "intent": "client",
        "keywords": ["client", "homonyme", "identite", "top client", "assure"],
        "tool_families": ["client"],
        "required": ["client"],
        "skip_llm": False,
    },
    {
        "intent": "alerte",
        "keywords": ["alerte", "seuil", "incident", "surveillance", "monitoring"],
        "tool_families": ["alerte", "kpi"],
        "required": ["alerte"],
        "skip_llm": False,
    },
    {
        "intent": "rag",
        "keywords": [
            "regle", "regles", "documentation", "policy", "rag", "contexte",
            # definition / explanation triggers
            "qu'est-ce", "c'est quoi", "definition", "defini", "signifie", "signifient",
            "expliquer", "explication", "comment fonctionne", "comment est calcule",
            "comment calcule", "comment calculer",
            # regulatory entities
            "fga", "cga", "cnam", "cnss",
            # regulatory concepts
            "norme", "normes", "reglementaire", "reglementation",
            "loi", "decret", "circulaire", "bareme", "baremes",
            "tarif", "tarification", "solvabilite",
            "provision", "psap", "ppna", "prc",
            "bonus malus", "malus",
            "franchise", "indemnisation",
            "delai", "delais", "preavis",
        ],
        "tool_families": ["rag"],
        "required": ["rag"],
        "skip_llm": False,
    },
    {
        "intent": "kpi",
        "keywords": ["kpi", "prime", "commission", "resiliation", "performance", "sinistre", "sinistres", "risque", "taux", "ratio combine", "ratio_combine", "ratio combin", "combined ratio", "combine"],
        "tool_families": ["kpi"],
        "required": ["kpi"],
        "skip_llm": False,
    },
    # ═══════════════════════════════════════════════════════════════════════
    # SQL 
    # ═══════════════════════════════════════════════════════════════════════
    {
        "intent": "sql",
        "keywords": [
            "sql", "requete", "query", "table", "tableau", "lignes",
            "dataset", "base de donnees", "bdd",
            "graphique", "graphe", "chart", "visualisation", "dataviz",
            "top", "classement", "evolution", "trend", "tendance",
            "repartition", "distribution", "historique",
            "nombre de", "combien de", "compter", "lister", "liste",
            "gouvernorat", "gouvernorats",
            "marque", "marques",
            "genre", "genres",
            "puissance",
            "famille",
            "nature",
            "vehicule", "vehicules", "voiture", "voitures", "parc",
            "sinistre", "sinistres",
            "police", "polices",
            "quittance", "quittances",
            "periodicite", "bonus malus",
            "situation", "en vigueur",
            "age client", "tranche",
            "assure", "assures",
            "emis", "emises",
            "tranche", "tranches",
            "provisions", "provision",
            "sexe", "femme", "femmes",
        ],
        "tool_families": ["sql"],
        "required": ["sql"],
        "skip_llm": False,
    },
]


class AgentState(TypedDict, total=False):
    question: str
    context: dict[str, Any]
    history: list[dict[str, Any]]   # prior turns from session_store
    intent: str
    intent_confidence: float
    tool_families: list[str]
    selected_tools: list[str]
    required_tools: list[str]
    skip_llm: bool
    policy_applied: str
    llm_used: bool
    status: str
    answer: str
    guardrails: dict[str, Any]
    tool_results: list[dict[str, Any]]
    specialist_reports: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    errors: list[str]


def _keyword_score(normalized_question: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in normalized_question)


# Patterns that signal a follow-up filter change rather than a new topic.
# "et pour IRDS ?", "et la branche SANTE ?", "même chose pour 2023", "idem pour AUTO"
_FOLLOWUP_PREFIXES = re.compile(
    r"^(et\s+(pour|la|le|les|en|sur)|meme\s+chose|idem|pareil|qu'en\s+est[- ]il|same\s+for)\b",
    re.IGNORECASE,
)


def _is_followup_question(normalized: str, history: list[dict[str, Any]]) -> bool:
    """
    Return True when the question is a short filter-change follow-up
    (e.g. 'et pour IRDS ?') and there is a prior turn to inherit from.
    """
    if not history:
        return False
    # Short questions only — a long question is almost always a new topic.
    if len(normalized.split()) > 8:
        return False
    return bool(_FOLLOWUP_PREFIXES.search(normalized))


def _is_predictive_question(normalized_question: str) -> bool:
    predictive_tokens = [
        "forecast",
        "prevoir",
        "prevois",
        "prevision",
        "previsionnel",
        "prediction",
        "predire",
        "projection",
        "horizon",
        "futur",
        "future",
        "prochain mois",
        "next month",
    ]
    return any(token in normalized_question for token in predictive_tokens)


def _is_sql_retrieval_question(normalized_question: str) -> bool:
    if "famille de risque" in normalized_question or "familles de risque" in normalized_question:
        return False
    if _is_predictive_question(normalized_question):
        return False

    sql_terms = [
        "sql",
        "requete",
        "query",
        "base de donnees",
        "bdd",
        "table",
        "tableau",
        "dataset",
        "lignes",
        "colonnes",
    ]
    viz_terms = [
        "graphique",
        "graphe",
        "chart",
        "plot",
        "visualisation",
        "dataviz",
    ]
    retrieval_verbs = [
        "donne",
        "montre",
        "affiche",
        "liste",
        "calcule",
        "combien",
        "top",
        "classement",
        "compare",
        "comparaison",
        "repartition",
        "distribution",
        "evolution",
        "trend",
    ]
    aggregation_terms = [
        "ratio",
        "taux",
        "total",
        "moyenne",
        "somme",
        "count",
        "nombre",
        "nb",
        "part",
        "top",
        "classement",
    ]
    metric_terms = [
        "prime", "pnet", "commission",
        "impaye", "impayes",
        "resiliation", "annulation",
        "sinistre", "sinistres",
        "client", "clients",
        "police", "polices",
        "quittance", "quittances",
        "montant", "montants",
        "vehicule", "vehicules", "voiture", "voitures",
        "marque", "marques",
        "produit", "produits",
        "agent", "agents",
        "assure", "assures",
        "emis", "emises",
    ]
    dimension_terms = [
        "branche", "branches",
        "gouvernorat", "gouvernorats",
        "mois", "mensuel", "annee", "periode",
        "agent", "ville", "zone", "segment",
        "marque", "marques",
        "genre", "genres", "puissance",
        "vehicule", "vehicules", "voiture",
        "produit", "produits", "famille",
        "usage", "type", "nature",
        "police", "polices", "client", "clients",
        "localite", "localites",
        "sinistre", "sinistres",
        "assure", "assures",
        "periodicite", "situation", "etat", "tranche",
        "quittance", "quittances",
    ]

    has_sql_term = any(token in normalized_question for token in sql_terms)
    has_viz_term = any(token in normalized_question for token in viz_terms)
    has_retrieval_verb = any(token in normalized_question for token in retrieval_verbs)
    has_aggregation = any(token in normalized_question for token in aggregation_terms)
    has_metric = any(token in normalized_question for token in metric_terms)
    has_dimension = any(token in normalized_question for token in dimension_terms)
    has_sql_cue = has_sql_term or has_viz_term

    # Keep direct KPI retention questions routed to KPI unless SQL/BI cues are explicit.
    if (
        "resiliation" in normalized_question
        and "impaye" not in normalized_question
        and not has_sql_cue
        and not any(token in normalized_question for token in ["top", "classement", "evolution", "trend"])
    ):
        return False

    if has_sql_cue and (has_metric or has_dimension or has_aggregation):
        return True

    if "impaye" in normalized_question and (has_aggregation or has_dimension):
        return True

    if "gouvernorat" in normalized_question and ("impaye" in normalized_question or "sinistre" in normalized_question):
        return True

    if "client" in normalized_question and (has_aggregation or "top" in normalized_question or "classement" in normalized_question):
        return True

    if "prime" in normalized_question and any(token in normalized_question for token in ["evolution", "trend", "mensuel", "mensuelle"]):
        return True

    if has_retrieval_verb and has_aggregation and has_dimension and (has_metric or "branche" in normalized_question):
        return True

    # Any question with a metric term + aggregation term is a SQL retrieval
    if has_metric and has_aggregation:
        return True

    # "nombre de X" where X is a known table entity
    if "nombre de" in normalized_question and has_metric:
        return True

    # "top N X par Y" pattern
    if any(t in normalized_question for t in ["top", "classement"]):
        if has_metric or has_dimension:
            return True

    return False


def is_domain_question(question: str) -> bool:
    normalized = _normalize_text(question)
    # Fast path: any domain keyword present
    if any(keyword in normalized for keyword in DOMAIN_KEYWORDS):
        return True
    # Analytical verbs with counting/listing intent are always domain questions
    # for an insurance analytics platform
    analytical_count_verbs = [
        "nombre de", "combien de", "donne moi", "donne-moi",
        "top ", "liste des", "lister", "affiche", "montre",
        "quel est le", "quelle est la",
    ]
    if any(verb in normalized for verb in analytical_count_verbs):
        return True
    return False


_OLLAMA_INTENT_SYSTEM = """
Tu es un classificateur d'intentions pour un agent analytique d'assurance (Maghrebia).
Ton seul rôle : identifier les intents présents dans la question et retourner un JSON.

Intents disponibles (choisis UNIQUEMENT parmi cette liste) :
- kpi          : KPI globaux, prime nette, commission, ratio combiné, taux résiliation, chiffres de performance
- sql          : données brutes, tableaux, graphes, top N, évolution, répartition, nombre de X, impayés, sinistres
- forecast     : prévision, projection, futur, prochain mois
- anomaly      : anomalie, outlier, pic inhabituel, rupture
- drift        : dérive modèle, stabilité, dégradation distribution
- alerte       : alerte, seuil critique, surveillance, monitoring
- explain      : explication facteurs, SHAP, pourquoi une prédiction
- predict      : lancer modèle ML, prédire fraude, modélisation
- segmentation : segmentation, cluster, profil client
- client       : recherche client, identité, homonyme
- rag          : définition, explication, réglementation, règles métier, documentation, normes, calcul méthodologique.
                 UTILISER pour : "qu'est-ce que X", "comment est calculé X", "c'est quoi X",
                 "quelle est la norme de X", "quelles sont les règles de X",
                 questions sur FGA, CNAM, CGA, CNSS, bonus-malus, provisions, tarification,
                 délais réglementaires, barèmes, solvabilité, code des assurances.
- dimension    : catalogue agents, produits, véhicules, polices
- overview     : vue globale, tableau de bord complet, diagnostic

Règles :
- Retourne 1 à 3 intents MAX, ordonnés par pertinence décroissante.
- Ne retourne QUE les intents clairement présents dans la question.
- confidence entre 0.50 et 0.97.
- Les questions de définition/explication/réglementation → intent "rag" en PREMIER.
- Réponds UNIQUEMENT avec ce JSON exact, rien d'autre :
{"intents": ["intent1"], "confidence": 0.9}
""".strip()


def _classify_with_ollama(
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    Ask Ollama to classify the question into 1-3 intents.
    If history is provided, the last 2 turns are prepended as prior messages so
    the model can resolve anaphoric questions like "et pour IRDS ?".
    Returns {"intents": [...], "confidence": float} or None on any failure.
    Uses a short timeout so the keyword fallback kicks in quickly if Ollama is slow.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _OLLAMA_INTENT_SYSTEM}]

    for turn in (history or [])[-2:]:
        messages.append({"role": "user", "content": str(turn.get("user", ""))})
        prior_intent = turn.get("intent") or "kpi"
        messages.append({"role": "assistant", "content": f'{{"intents":["{prior_intent}"],"confidence":0.9}}'})

    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 80},
            },
            timeout=8.0,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        parsed = json.loads(content)
        intents = parsed.get("intents")
        if isinstance(intents, list) and intents:
            return {
                "intents": [str(i) for i in intents],
                "confidence": float(parsed.get("confidence", 0.75)),
            }
    except Exception:
        pass
    return None


def _classify_from_ollama(
    result: dict[str, Any],
) -> tuple[str, float, list[str], list[str], bool]:
    """Build the classify_question return tuple from a validated Ollama result."""
    valid_intents = {r["intent"] for r in INTENT_RULES}
    intents = [i for i in result["intents"] if i in valid_intents]
    if not intents:
        intents = ["kpi"]

    confidence = round(min(0.97, max(0.50, result.get("confidence", 0.75))), 3)

    # Accumulate tool families from ALL detected intents (true multi-intent).
    tool_families: list[str] = []
    for intent_name in intents:
        rule = next((r for r in INTENT_RULES if r["intent"] == intent_name), None)
        if rule:
            for family in rule["tool_families"]:
                if family not in tool_families:
                    tool_families.append(family)

    # Primary intent drives required tools and skip_llm.
    primary_rule = next(
        (r for r in INTENT_RULES if r["intent"] == intents[0]),
        next(r for r in INTENT_RULES if r["intent"] == "kpi"),
    )
    required_families = list(primary_rule.get("required", []))
    skip_llm = bool(primary_rule.get("skip_llm", False))

    return intents[0], confidence, tool_families, required_families, skip_llm


def _classify_from_keywords(
    normalized: str,
) -> tuple[str, float, list[str], list[str], bool]:
    """Original keyword-scoring classifier — used as fallback when Ollama is unavailable."""
    scored: list[tuple[int, dict[str, Any]]] = []

    for rule in INTENT_RULES:
        score = _keyword_score(normalized, rule["keywords"])
        if score > 0:
            scored.append((score, rule))

    if _is_sql_retrieval_question(normalized):
        sql_rule = next(rule for rule in INTENT_RULES if rule["intent"] == "sql")
        sql_found = False
        for index, (score, rule) in enumerate(scored):
            if str(rule.get("intent")) == "sql":
                scored[index] = (max(score, 3), rule)
                sql_found = True
                break
        if not sql_found:
            scored.append((3, sql_rule))

    if _is_predictive_question(normalized):
        forecast_rule = next(rule for rule in INTENT_RULES if rule["intent"] == "forecast")
        forecast_found = False
        for index, (score, rule) in enumerate(scored):
            if str(rule.get("intent")) == "forecast":
                scored[index] = (max(score, 4), rule)
                forecast_found = True
                break
        if not forecast_found:
            scored.append((4, forecast_rule))

        explicit_sql_cues = ["sql", "requete", "query", "base de donnees", "bdd"]
        if not any(token in normalized for token in explicit_sql_cues):
            scored = [(s, r) for s, r in scored if str(r.get("intent")) != "sql"]

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        primary_score, primary_rule = scored[0]
        matched_rules = [entry[1] for entry in scored]
    else:
        primary_score = 0
        primary_rule = next(rule for rule in INTENT_RULES if rule["intent"] == "kpi")
        matched_rules = [primary_rule]

    tool_families = list(primary_rule["tool_families"])
    primary_intent = str(primary_rule["intent"])

    for _, rule in scored[1:3]:
        for family in rule["tool_families"]:
            if primary_intent == "sql" and family != "sql":
                continue
            if primary_intent == "forecast" and family in {"kpi", "rag", "sql"}:
                continue
            if family not in tool_families:
                tool_families.append(family)
                break

    if "sql" in normalized and "sql" not in tool_families:
        tool_families.insert(0, "sql")

    confidence = 0.45
    if primary_score > 0:
        confidence = min(0.96, 0.56 + (0.1 * primary_score) + (0.02 * (len(matched_rules) - 1)))

    required_families = list(primary_rule.get("required", []))
    skip_llm = bool(primary_rule.get("skip_llm", False))

    return (
        str(primary_rule["intent"]),
        round(confidence, 3),
        tool_families,
        required_families,
        skip_llm,
    )


def _classify_from_model(
    question: str,
) -> tuple[str, float, list[str], list[str], bool] | None:
    """
    Try the trained TF-IDF + LinearSVC classifier first.
    Returns the full classify_question tuple on success, None if the pkl is
    absent, confidence is too low, or the predicted intent is unknown.
    """
    try:
        from intent_classifier import classify as _clf_classify
        result = _clf_classify(question, min_confidence=0.60)
        if result is None:
            return None
        intent, confidence, _ = result
        rule = next((r for r in INTENT_RULES if r["intent"] == intent), None)
        if rule is None:
            return None
        tool_families = list(rule.get("tool_families", [intent]))
        required      = list(rule.get("required", []))
        skip_llm      = bool(rule.get("skip_llm", False))
        return intent, round(confidence, 3), tool_families, required, skip_llm
    except Exception:
        return None


def classify_question(
    question: str,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, float, list[str], list[str], bool]:
    """
    Classify the user question into an intent + tool families.

    Path 1 (trained model): TF-IDF + LinearSVC with calibrated probabilities.
                            Instant (<1ms), no network call. Used when the pkl
                            exists and confidence ≥ 0.60.
    Path 2 (Ollama):        Structured JSON prompt via llama3.  Handles anaphoric
                            follow-ups ("et pour IRDS ?") using conversation history.
    Path 3 (keywords):      Deterministic keyword-scoring fallback.
                            Always available even when Ollama is down.

    Set AGENT_FORCE_DETERMINISTIC=true to skip Paths 1 and 2.
    """
    normalized = _normalize_text(question)

    if not FORCE_DETERMINISTIC:
        model_result = _classify_from_model(question)
        if model_result is not None:
            return model_result

        ollama_result = _classify_with_ollama(question, history=history)
        if ollama_result:
            return _classify_from_ollama(ollama_result)

    return _classify_from_keywords(normalized)


def detect_requested_tools(question: str) -> list[str]:
    _, _, tool_families, _, _ = classify_question(question)
    return tool_families


def _infer_branch_from_question(question: str) -> str | None:
    normalized = _normalize_text(question)
    if re.search(r"\b(toutes?\s+(?:les\s+)?branches|all\s+branches)\b", normalized):
        return "ALL"
    branch_match = re.search(r"\b(auto|irds|sante)\b", normalized)
    if not branch_match:
        return None
    return str(branch_match.group(1)).upper()


def _infer_year_range_from_question(question: str) -> tuple[int, int] | None:
    matches = re.findall(r"\b(20\d{2})\b", question)
    if not matches:
        return None

    years = sorted({int(item) for item in matches})
    if len(years) == 1:
        return years[0], years[0]
    return years[0], years[-1]


def _infer_history_year_range_from_question(question: str) -> tuple[int, int] | None:
    normalized = _normalize_text(question)
    patterns = [
        r"\bhistorique[^0-9]*(20\d{2})\s*[-/]\s*(20\d{2})\b",
        r"\bhistorique[^0-9]*(?:de\s*)?(20\d{2})\s*(?:a|au|to)\s*(20\d{2})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        return (first, second) if first <= second else (second, first)

    return None


def _infer_horizon_months_from_question(question: str) -> int | None:
    normalized = _normalize_text(question)
    patterns = [
        r"\bhorizon[^0-9]*(\d{1,2})\s*(?:mois|month|months)\b",
        r"\b(?:sur|pour|prochains?)\s*(\d{1,2})\s*(?:mois|month|months)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        horizon = int(match.group(1))
        return max(1, min(horizon, 12))
    return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _map_families_to_tools(families: list[str]) -> list[str]:
    tools: list[str] = []
    for family in families:
        tool_name = TOOL_FAMILY_TO_TOOL.get(family)
        if tool_name and tool_name not in tools:
            tools.append(tool_name)
    return tools


def _extract_visual_artifacts(
    tool_results: list[dict[str, Any]],
    *,
    intent: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    charts: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    intent_normalized = (intent or "").strip().lower()
    allowed_chart_tools: set[str] | None = None
    if intent_normalized == "forecast":
        # Keep forecast visuals focused and avoid KPI side charts unrelated to projection output.
        allowed_chart_tools = {"forecast_tool"}

    for result in tool_results:
        result_tool = str(result.get("tool", "")).strip().lower()
        charts_allowed = allowed_chart_tools is None or result_tool in allowed_chart_tools
        tables_allowed = allowed_chart_tools is None or result_tool in allowed_chart_tools

        for chart in result.get("charts", []) or []:
            if charts_allowed and isinstance(chart, dict):
                charts.append(chart)
        for table in result.get("tables", []) or []:
            if tables_allowed and isinstance(table, dict):
                tables.append(table)
    return charts, tables


def _compact_json(value: Any, max_chars: int = 1600) -> str:
    try:
        serialized = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        serialized = str(value)
    if len(serialized) <= max_chars:
        return serialized
    return serialized[:max_chars] + "..."


def _build_specialist_report(result: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(result.get("tool", "unknown_tool"))
    specialist_name = TOOL_SPECIALIST_AGENTS.get(tool_name, "general_specialist")
    has_error = "error" in result

    report: dict[str, Any] = {
        "agent": specialist_name,
        "tool": tool_name,
        "status": "error" if has_error else "ok",
        "summary": str(result.get("summary", "")).strip() or ("Erreur tool" if has_error else "Execution terminee."),
    }

    if has_error:
        report["error"] = str(result.get("error", "unknown error"))

    payload = result.get("payload")
    if isinstance(payload, dict) and payload:
        report["payload_excerpt"] = _compact_json(payload, max_chars=700)

    return report


def _has_successful_tool_results(state: AgentState) -> bool:
    return any(
        isinstance(result, dict) and "error" not in result
        for result in state.get("tool_results", [])
    )


def _is_llm_answer_usable(answer: str) -> bool:
    content = str(answer or "").strip()
    if len(content) < max(30, LLM_MIN_ANSWER_CHARS):
        return False

    words = re.findall(r"\w+", content)
    return len(words) >= 18


def _should_use_llm(state: AgentState) -> bool:
    if FORCE_DETERMINISTIC:
        return False
    if state.get("skip_llm", False):
        return False
    if not _has_successful_tool_results(state):
        return False

    intent = str(state.get("intent", "")).strip().lower()
    if intent in DETERMINISTIC_INTENTS:
        return False
    return True


def _append_unique(items: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in items:
        items.append(normalized)


def _format_period_label(payload: dict[str, Any]) -> str:
    year_from = payload.get("year_from")
    year_to = payload.get("year_to")
    if isinstance(year_from, int) and isinstance(year_to, int):
        return str(year_from) if year_from == year_to else f"{year_from}-{year_to}"
    return "la periode demandee"


def _compose_decision_answer(state: AgentState) -> str | None:
    tool_results = [
        result
        for result in state.get("tool_results", [])
        if isinstance(result, dict) and "error" not in result
    ]
    if not tool_results:
        return None

    intent = str(state.get("intent", "kpi"))
    confidence = _as_float(state.get("intent_confidence"), 0.0)

    synthese_items: list[str] = []
    chiffres_cles: list[str] = []
    decision_items: list[str] = []
    actions: list[str] = []
    tools_used: list[str] = []

    for result in tool_results:
        tool_name = str(result.get("tool", "unknown_tool"))
        summary = str(result.get("summary", "")).strip()
        payload = result.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        _append_unique(tools_used, tool_name)

        if tool_name == "kpi_tool":
            period_label = _format_period_label(payload)
            branch_label = _format_branch_label(payload.get("branch"))
            focus = str(payload.get("focus", "overview"))
            kpis = payload.get("kpis") if isinstance(payload.get("kpis"), list) else []

            # Build synthese line based on what was asked (focus)
            if focus == "ratio_combine":
                ratio_combine = _as_float(payload.get("ratio_combine_pct"), 0.0)
                sp_reel = _as_float(payload.get("sp_ratio_reel_pct"), 0.0)
                total_pnet = _as_float(payload.get("total_pnet"), 0.0)
                total_commission = _as_float(payload.get("total_commission"), 0.0)
                expense = (100.0 * total_commission / total_pnet) if total_pnet > 0 else 0.0
                _append_unique(synthese_items,
                    f"Ratio Combine sur {period_label} ({branch_label}): {_format_percent(ratio_combine)} "
                    f"(S/P pur {_format_percent(sp_reel)} + expense {_format_percent(expense)}).")
                _append_unique(chiffres_cles, f"- Ratio Combine: {_format_percent(ratio_combine)}")
                _append_unique(chiffres_cles, f"- S/P pur (sinistres/pnet): {_format_percent(sp_reel)}")
                _append_unique(chiffres_cles, f"- Expense ratio (commission/pnet): {_format_percent(expense)}")
                if ratio_combine >= 100.0:
                    _append_unique(decision_items, "Ratio combine au-dessus de 100%: activite deficitaire.")
                    _append_unique(actions, "Revoir la politique tarifaire et les reserves sinistres.")
                elif ratio_combine >= 80.0:
                    _append_unique(decision_items, "Ratio combine eleve: marge technique sous pression.")
                    _append_unique(actions, "Analyser les branches les plus sinistrees et ajuster les tarifs.")
                else:
                    _append_unique(decision_items, "Ratio combine sain: activite rentable techniquement.")

            elif focus == "resiliation":
                taux_resiliation = _as_float(payload.get("taux_resiliation_pct"), 0.0)
                _append_unique(synthese_items,
                    f"Taux resiliation sur {period_label} ({branch_label}): {_format_percent(taux_resiliation)}.")
                _append_unique(chiffres_cles, f"- Taux de resiliation: {_format_percent(taux_resiliation)}")
                _append_unique(chiffres_cles, f"- Polices resiliees: {_as_float(payload.get('polices_resiliees'),0.0):,.0f}")
                if taux_resiliation >= 8.0:
                    _append_unique(decision_items, "Risque retention eleve: resiliation depasse 8%.")
                    _append_unique(actions, "Lancer un plan anti-churn cible sur les segments a forte resiliation.")
                elif taux_resiliation >= 4.0:
                    _append_unique(decision_items, "Risque retention modere: surveillance requise.")
                else:
                    _append_unique(decision_items, "Retention maitrisee sur la periode analysee.")

            elif focus in {"sinistre", "impaye", "prime"}:
                # Surface the focused KPI cards directly
                for kpi in kpis[:4]:
                    if not isinstance(kpi, dict):
                        continue
                    label = str(kpi.get("label", "KPI"))
                    value = _as_float(kpi.get("value"), 0.0)
                    unit = str(kpi.get("unit", "")).upper()
                    _append_unique(chiffres_cles, f"- {label}: {_format_metric_value(value, unit)}")
                if kpis:
                    first = kpis[0]
                    _append_unique(synthese_items,
                        f"{first.get('label','KPI')} sur {period_label} ({branch_label}): "
                        f"{_format_metric_value(_as_float(first.get('value'),0.0), str(first.get('unit','')).upper())}.")

            else:
                # Full overview
                total_pnet = _as_float(payload.get("total_pnet"), 0.0)
                total_commission = _as_float(payload.get("total_commission"), 0.0)
                ratio_combine = _as_float(payload.get("ratio_combine_pct"), 0.0)
                taux_resiliation = _as_float(payload.get("taux_resiliation_pct"), 0.0)
                _append_unique(synthese_items,
                    f"Vue globale KPI {period_label} ({branch_label}): "
                    f"prime nette {_format_amount_tnd(total_pnet)}, "
                    f"ratio combine {_format_percent(ratio_combine)}, "
                    f"resiliation {_format_percent(taux_resiliation)}.")
                _append_unique(chiffres_cles, f"- Prime nette: {_format_amount_tnd(total_pnet)}")
                _append_unique(chiffres_cles, f"- Commission: {_format_amount_tnd(total_commission)}")
                _append_unique(chiffres_cles, f"- Ratio Combine: {_format_percent(ratio_combine)}")
                _append_unique(chiffres_cles, f"- Taux de resiliation: {_format_percent(taux_resiliation)}")
                if taux_resiliation >= 4.0:
                    _append_unique(decision_items, "Risque retention a surveiller.")
                if ratio_combine >= 80.0:
                    _append_unique(decision_items, "Ratio combine eleve: marge technique sous pression.")

        elif tool_name == "forecast_tool":
            predictions = payload.get("predictions") if isinstance(payload.get("predictions"), list) else []
            target_label = str(payload.get("target_label", "metrique"))
            target_unit = str(payload.get("target_unit", "TND"))
            trend_pct = _as_float(payload.get("trend_pct"), 0.0)
            horizon_months = int(_as_float(payload.get("horizon_months"), 0.0))
            engine = str(payload.get("engine", "unknown"))

            context_line = str(payload.get("context", "")).strip()
            analysis_line = str(payload.get("analysis", "")).strip()
            decision_line = str(payload.get("decision", "")).strip()
            forecast_actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
            kpis = payload.get("kpis") if isinstance(payload.get("kpis"), list) else []

            if context_line:
                _append_unique(synthese_items, context_line)
            if analysis_line:
                _append_unique(synthese_items, analysis_line)

            for item in kpis[:4]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "KPI")).strip() or "KPI"
                value = _as_float(item.get("value"), 0.0)
                unit = str(item.get("unit", "")).strip()
                _append_unique(chiffres_cles, f"- {label}: {_format_metric_value(value, unit)}")

            if decision_line:
                _append_unique(decision_items, decision_line)
                for action in forecast_actions[:3]:
                    if isinstance(action, str) and action.strip():
                        _append_unique(actions, action.strip())

            prediction_key = ""
            if predictions and isinstance(predictions[0], dict):
                prediction_key = next((key for key in predictions[0].keys() if key != "period"), "")

            total_forecast = 0.0
            average_forecast = 0.0
            first_period = ""
            last_period = ""
            if prediction_key and predictions:
                total_forecast = sum(_as_float(item.get(prediction_key), 0.0) for item in predictions if isinstance(item, dict))
                average_forecast = total_forecast / len(predictions)
                first_period = str(predictions[0].get("period", ""))
                last_period = str(predictions[-1].get("period", ""))

            if not context_line and not analysis_line:
                _append_unique(
                    synthese_items,
                    (
                        f"Forecast {target_label} sur {horizon_months} mois ({engine}): "
                        f"tendance {_format_percent(trend_pct)} vs dernier mois observe."
                    ),
                )

            if prediction_key and predictions and not kpis:
                _append_unique(
                    chiffres_cles,
                    (
                        f"- Projection cumulee ({first_period} a {last_period}): "
                        f"{_format_metric_value(total_forecast, target_unit)}"
                    ),
                )
                _append_unique(
                    chiffres_cles,
                    f"- Projection moyenne mensuelle: {_format_metric_value(average_forecast, target_unit)}",
                )

            if not decision_line:
                if trend_pct <= -5.0:
                    _append_unique(decision_items, "Baisse anticipee: scenario de contraction a traiter rapidement.")
                    _append_unique(actions, "Activer un plan commercial de relance sur les branches en baisse projetee.")
                elif trend_pct >= 5.0:
                    _append_unique(decision_items, "Croissance anticipee: potentiel de developpement a capter.")
                    _append_unique(actions, "Renforcer la capacite commerciale sur les segments les plus dynamiques.")
                else:
                    _append_unique(decision_items, "Tendance stable: maintien operationnel avec pilotage mensuel.")

        elif tool_name == "alerte_tool":
            alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
            high_count = sum(1 for alert in alerts if isinstance(alert, dict) and str(alert.get("severity", "")).lower() == "high")
            medium_count = sum(1 for alert in alerts if isinstance(alert, dict) and str(alert.get("severity", "")).lower() == "medium")

            _append_unique(chiffres_cles, f"- Alertes critiques: {high_count}")
            _append_unique(chiffres_cles, f"- Alertes moyennes: {medium_count}")
            if high_count > 0:
                _append_unique(decision_items, "Signal de risque operationnel eleve (alertes critiques presentes).")
                _append_unique(actions, "Traiter en priorite les alertes critiques et suivre la resolution sous 48h.")

        elif tool_name == "anomaly_tool":
            anomalies = payload.get("anomalies") if isinstance(payload.get("anomalies"), list) else []
            _append_unique(chiffres_cles, f"- Anomalies detectees: {len(anomalies)}")
            if anomalies:
                _append_unique(decision_items, "Anomalies detectees sur les impayes: revue ciblee recommandee.")
                _append_unique(actions, "Analyser les periodes anormales avec les equipes recouvrement et souscription.")

        elif tool_name == "drift_tool":
            drift_status = str(payload.get("status", "unknown")).lower()
            _append_unique(chiffres_cles, f"- Statut drift: {drift_status}")
            if drift_status in {"high", "medium"}:
                _append_unique(decision_items, "Derive de donnees/modeles detectee: calibration a planifier.")
                _append_unique(actions, "Declencher une verification de stabilite modeles et des distributions d entree.")

        elif tool_name == "sql_tool":
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            result_kind = str(payload.get("result_kind", "tabular")).lower()
            kpis = payload.get("kpis") if isinstance(payload.get("kpis"), list) else []

            if kpis:
                for item in kpis[:4]:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label", "KPI")).strip() or "KPI"
                    value = _as_float(item.get("value"), 0.0)
                    unit = str(item.get("unit", "")).strip()
                    _append_unique(chiffres_cles, f"- {label}: {_format_metric_value(value, unit)}")
            elif result_kind == "scalar" and rows:
                _append_unique(chiffres_cles, "- Valeur metier calculee sur la periode demandee.")
            else:
                _append_unique(chiffres_cles, f"- Lignes retournees: {len(rows)}")

            context_line = str(payload.get("context", "")).strip()
            analysis_line = str(payload.get("analysis", "")).strip()
            decision_line = str(payload.get("decision", "")).strip()
            sql_actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []

            if context_line:
                _append_unique(synthese_items, context_line)
            if analysis_line:
                _append_unique(synthese_items, analysis_line)
            if decision_line:
                _append_unique(decision_items, decision_line)

            for action in sql_actions[:3]:
                if isinstance(action, str) and action.strip():
                    _append_unique(actions, action.strip())

        elif tool_name == "client_tool":
            top_clients = payload.get("top_claim_clients") if isinstance(payload.get("top_claim_clients"), list) else []
            _append_unique(chiffres_cles, f"- Clients a risque impaye (top): {len(top_clients)}")

        elif tool_name == "segmentation_tool":
            segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
            _append_unique(chiffres_cles, f"- Segments clients identifies: {len(segments)}")

        elif tool_name == "explain_tool":
            feature_importance = payload.get("feature_importance") if isinstance(payload.get("feature_importance"), list) else []
            _append_unique(chiffres_cles, f"- Facteurs explicatifs disponibles: {len(feature_importance)}")

        elif tool_name == "rag_tool":
            documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
            _append_unique(chiffres_cles, f"- Snippets de contexte metier: {len(documents)}")

        if summary and tool_name not in {"kpi_tool", "forecast_tool", "sql_tool"}:
            _append_unique(synthese_items, summary)

    if not synthese_items:
        synthese_items = ["Analyse terminee sur la base des donnees disponibles."]

    lines: list[str] = []
    lines.append("**Synthese decisionnelle**")
    lines.append(" ".join(synthese_items[:3]))

    if chiffres_cles:
        lines.append("\n**Chiffres cles**")
        lines.extend(chiffres_cles[:8])

    if decision_items:
        lines.append("\n**Points de decision**")
        for item in decision_items[:4]:
            lines.append(f"- {item}")

    if actions:
        lines.append("\n**Actions recommandees**")
        for action in actions[:3]:
            lines.append(f"- {action}")

    return "\n".join(lines)


def _compose_deterministic_answer(state: AgentState) -> str:
    decision_answer = _compose_decision_answer(state)
    precise_answer = _compose_precise_metric_answer(state)
    if precise_answer and decision_answer:
        return f"{precise_answer}\n\n{decision_answer}"
    if precise_answer:
        return precise_answer
    if decision_answer:
        return decision_answer

    intent = state.get("intent", "kpi")
    confidence = state.get("intent_confidence", 0.0)
    lines = [f"Intent detecte: {intent} (confiance {confidence:.2f})."]

    tool_results = state.get("tool_results", [])
    if not tool_results:
        lines.append("Aucun resultat outil disponible.")
    else:
        lines.append("Synthese tools:")
        for result in tool_results:
            summary = str(result.get("summary", "")).strip() or "Resultat sans resume."
            lines.append(f"- {result.get('tool', 'tool')}: {summary}")

    if state.get("errors"):
        lines.append("Limites techniques:")
        for error in state["errors"][:4]:
            lines.append(f"- {error}")

    if state.get("charts"):
        lines.append(f"Visualisations disponibles: {len(state['charts'])}.")
    if state.get("tables"):
        lines.append(f"Tables disponibles: {len(state['tables'])}.")

    return "\n".join(lines)


def _compose_precise_metric_answer(state: AgentState) -> str | None:
    """Return a direct precise answer for single-KPI questions using the kpi_tool focus."""
    tool_results = state.get("tool_results", [])

    # ── 1. kpi_tool with focus ─────────────────────────────────────────────
    kpi_result = next(
        (r for r in tool_results if isinstance(r, dict)
         and r.get("tool") == "kpi_tool" and "error" not in r),
        None,
    )
    if kpi_result:
        payload = kpi_result.get("payload", {}) if isinstance(kpi_result, dict) else {}
        if isinstance(payload, dict):
            focus = str(payload.get("focus", "overview"))
            kpis = payload.get("kpis") if isinstance(payload.get("kpis"), list) else []
            period_label = _format_period_label(payload)
            branch_label = _format_branch_label(payload.get("branch"))

            if focus == "ratio_combine" and kpis:
                ratio = _as_float(payload.get("ratio_combine_pct"), 0.0)
                sp = _as_float(payload.get("sp_ratio_reel_pct"), 0.0)
                total_pnet = _as_float(payload.get("total_pnet"), 0.0)
                total_commission = _as_float(payload.get("total_commission"), 0.0)
                expense = (100.0 * total_commission / total_pnet) if total_pnet > 0 else 0.0
                lines = [
                    f"**Ratio Combiné** — {branch_label}, {period_label}",
                    f"",
                    f"**{ratio:.1f}%**",
                    f"",
                    f"Détail:",
                    f"- S/P pur (sinistres payés / prime nette) : {sp:.2f}%",
                    f"- Expense ratio (commission / prime nette) : {expense:.2f}%",
                    f"- Sinistres payés : {_format_amount_tnd(_as_float(payload.get('total_mt_paye_sinistres'),0.0))}",
                    f"- Commission : {_format_amount_tnd(total_commission)}",
                    f"- Prime nette : {_format_amount_tnd(total_pnet)}",
                ]
                return "\n".join(lines)

            if focus == "resiliation" and kpis:
                taux = _as_float(payload.get("taux_resiliation_pct"), 0.0)
                resiliees = _as_float(payload.get("polices_resiliees"), 0.0)
                total = _as_float(payload.get("total_polices_churn"), 0.0)
                return (
                    f"**Taux de résiliation** — {branch_label}, {period_label}\n\n"
                    f"**{taux:.2f}%** ({int(resiliees):,} polices résiliées / {int(total):,} totales)"
                )

            # Generic focus with kpi cards
            if kpis and focus != "overview":
                lines = [f"**{kpis[0].get('label','KPI')}** — {branch_label}, {period_label}", ""]
                for kpi in kpis:
                    if not isinstance(kpi, dict):
                        continue
                    label = str(kpi.get("label", "KPI"))
                    value = _as_float(kpi.get("value"), 0.0)
                    unit = str(kpi.get("unit", "")).upper()
                    lines.append(f"- **{label}** : {_format_metric_value(value, unit)}")
                return "\n".join(lines)

    # ── 2. sql_tool scalar result ──────────────────────────────────────────
    sql_result = next(
        (r for r in tool_results if isinstance(r, dict)
         and r.get("tool") == "sql_tool" and "error" not in r),
        None,
    )
    if sql_result:
        payload = sql_result.get("payload", {})
        if isinstance(payload, dict):
            result_kind = str(payload.get("result_kind", "")).lower()
            kpis = payload.get("kpis") if isinstance(payload.get("kpis"), list) else []
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            analysis = str(payload.get("analysis", "")).strip()
            context_text = str(payload.get("context", "")).strip()

            if result_kind == "scalar" and kpis:
                lines = []
                if context_text:
                    lines.append(context_text)
                for kpi in kpis[:6]:
                    if not isinstance(kpi, dict):
                        continue
                    label = str(kpi.get("label", "Valeur")).strip()
                    value = _as_float(kpi.get("value"), 0.0)
                    unit = str(kpi.get("unit", "")).strip().upper()
                    formatted = _format_metric_value(value, unit) if unit else f"{int(round(value)):,}".replace(",", " ")
                    lines.append(f"**{label}**: {formatted}")
                if analysis:
                    lines.append(f"\n{analysis}")
                return "\n".join(lines)

            if result_kind in {"breakdown", "timeseries"} and rows:
                lines = []
                if context_text:
                    lines.append(context_text)
                if analysis:
                    lines.append(analysis)
                return "\n".join(lines) if lines else None

    return None


def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    host  = OLLAMA_HOST
    model = OLLAMA_CHAT_MODEL
    cb    = _progress_cb.get(None)
    use_stream = cb is not None

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": use_stream,
            "options": {"temperature": 0.2},
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
        stream=use_stream,
    )
    response.raise_for_status()

    if use_stream:
        parts: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = (chunk.get("message") or {}).get("content", "")
            if token:
                parts.append(token)
                cb({"type": "llm_token", "token": token})  # type: ignore[misc]
            if chunk.get("done"):
                break
        content = "".join(parts)
    else:
        content = (response.json().get("message") or {}).get("content", "")

    if not content or not str(content).strip():
        raise RuntimeError("Empty response from Ollama chat endpoint")
    return str(content).strip()


def _build_llm_prompt(state: AgentState) -> tuple[str, str]:
    question = state.get("question", "")
    context_payload = state.get("context", {})
    grounded_draft = _compose_deterministic_answer(state)

    tool_context = []
    for result in state.get("tool_results", []):
        tool_context.append(
            {
                "tool": result.get("tool"),
                "summary": result.get("summary"),
                "payload": _compact_json(result.get("payload", {}), max_chars=2000),
            }
        )

    specialist_reports = [
        report
        for report in state.get("specialist_reports", [])
        if isinstance(report, dict)
    ]
    specialist_context = []
    for report in specialist_reports[:10]:
        specialist_context.append(
            {
                "agent": report.get("agent"),
                "tool": report.get("tool"),
                "status": report.get("status"),
                "summary": report.get("summary"),
                "payload_excerpt": report.get("payload_excerpt"),
            }
        )

    system_prompt = (
        "Tu es l'agent de synthese d'un systeme multi-agent assurance pour Maghrebia. Reponds en francais.\n"
        "Ton role est de consolider les donnees issues d'une generation NLP->SQL et du contexte documentaire (RAG).\n"
        "Tu dois OBLIGATOIREMENT structurer ta reponse en exactement 4 parties avec ces en-tetes exacts:\n"
        "\n**Contexte**: (Resume UNIQUEMENT LA QUESTION ACTUELLE et la source des donnees utilisees. "
        "Ne parle PAS des questions precedentes de la conversation. "
        "Ex: 'La presente analyse repond a la question sur X, basee sur la documentation interne RAG.')\n"
        "\n**Analyse**: (Analyse les donnees recuperees depuis les outils et le brouillon. Explique les tendances et les chiffres.)\n"
        "\n**Decision**: (Propose des recommandations metier basees sur l'analyse, ex: 'Il est recommande de cibler telle branche...')\n"
        "\n**Graphs/Tableaux**: (Si tu as recu des donnees formattes ou de serie temporelle, presente un resume tabulaire ou une suggestion claire du graphique a tracer. "
        "Si la question est une definition, indique simplement 'Pas de graphique applicable.')\n"
        "REGLES IMPORTANTES:\n"
        "- Si rag_tool a retourne des documents, cite leur contenu EN PRIORITE pour les questions de definition/reglementation.\n"
        "- Ne reponds PAS avec des chiffres KPI si la question est une question de definition.\n"
        "- La section **Contexte** doit decrire la question ACTUELLE uniquement, pas l'historique.\n"
    )

    history = state.get("history") or []
    history_block = ""
    if history:
        lines = ["## Historique (tours precedents — NE PAS confondre avec la question actuelle)"]
        for turn in history[-3:]:
            lines.append(f"[Tour precedent] Utilisateur: {turn.get('user', '')}")
            lines.append(f"[Tour precedent] Assistant: {str(turn.get('assistant', ''))[:200]}")
        history_block = "\n".join(lines) + "\n\n"

    user_prompt = (
        f"{history_block}"
        f"=== QUESTION ACTUELLE A TRAITER ===\n"
        f"{question}\n"
        f"===================================\n"
        f"Intent detecte: {state.get('intent')} (confiance={state.get('intent_confidence')})\n"
        f"Contexte filtres: {_compact_json(context_payload, max_chars=900)}\n"
        f"Rapports agents specialises: {_compact_json(specialist_context, max_chars=4000)}\n"
        f"Sorties outils: {_compact_json(tool_context, max_chars=7000)}\n"
        f"Brouillon deterministe fiable: {_compact_json(grounded_draft, max_chars=3000)}\n"
        "Instruction: produire une synthese de la QUESTION ACTUELLE uniquement, sans perdre les faits."
    )
    return system_prompt, user_prompt


async def _execute_tool(tool_name: str, question: str, context: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(run_tool, tool_name, question, context)
        if "tool" not in result:
            result["tool"] = tool_name
        return result
    except Exception as exc:
        return {
            "tool": tool_name,
            "error": str(exc),
            "summary": f"Erreur d'execution pour {tool_name}.",
            "payload": {},
        }


def _guardrails_node(state: AgentState) -> AgentState:
    _emit({"type": "progress", "step": "guardrails", "label": "Vérification du périmètre..."})
    question = state["question"]
    if not is_domain_question(question):
        return {
            "status": "blocked",
            "answer": OUT_OF_SCOPE_MESSAGE,
            "guardrails": {
                "domain_allowed": False,
                "reason": "out_of_scope",
            },
            "steps": state.get("steps", []) + [{"step": "guardrails", "status": "blocked"}],
        }

    return {
        "status": "ok",
        "guardrails": {
            "domain_allowed": True,
            "reason": "domain_match",
        },
        "steps": state.get("steps", []) + [{"step": "guardrails", "status": "passed"}],
    }


def _low_confidence_message(confidence: float) -> str:
    return (
        "Je n'ai pas bien compris votre demande. Pourriez-vous préciser ce que vous souhaitez ?\n\n"
        "Par exemple :\n"
        "- **KPI / indicateurs** : \"Donne-moi le taux de résiliation AUTO pour 2024\"\n"
        "- **Prévision** : \"Prévision de la prime nette sur 3 mois pour la branche IRDS\"\n"
        "- **Requête de données** : \"Top 5 gouvernorats par montant impayé en 2024\"\n"
        "- **Prédiction ML (fraude/résiliation)** : \"Lance le modèle ML sur les données 2024\"\n"
        "- **Anomalies** : \"Détecte les anomalies sur les impayés 2023\"\n\n"
        "Quelle analyse souhaitez-vous effectuer ?"
    )


def _intent_node(state: AgentState) -> AgentState:
    _emit({"type": "progress", "step": "intent", "label": "Classification de l'intention..."})
    context = dict(state.get("context", {}))
    inferred_branch = _infer_branch_from_question(state["question"])
    if inferred_branch == "ALL":
        context["branch"] = None
    elif inferred_branch:
        context["branch"] = inferred_branch

    inferred_history_range = _infer_history_year_range_from_question(state["question"])
    if inferred_history_range:
        context["year_from"], context["year_to"] = inferred_history_range
    else:
        inferred_year_range = _infer_year_range_from_question(state["question"])
        if inferred_year_range:
            context["year_from"], context["year_to"] = inferred_year_range

    inferred_horizon = _infer_horizon_months_from_question(state["question"])
    if inferred_horizon:
        context["horizon_months"] = inferred_horizon

    history = state.get("history") or []
    normalized_q = _normalize_text(state["question"])

    if _is_followup_question(normalized_q, history):
        # Inherit the previous turn's intent so "et pour IRDS ?" keeps calling kpi_tool,
        # not sql_tool, when the prior question was a KPI query.
        prior_intent = history[-1].get("intent", "") if history else ""
        prior_rule = next((r for r in INTENT_RULES if r["intent"] == prior_intent), None)
        if prior_rule:
            intent = prior_intent
            confidence = 0.92
            tool_families = list(prior_rule["tool_families"])
            required_families = list(prior_rule.get("required", []))
            skip_llm = bool(prior_rule.get("skip_llm", False))
        else:
            intent, confidence, tool_families, required_families, skip_llm = classify_question(
                state["question"], history=history
            )
    else:
        intent, confidence, tool_families, required_families, skip_llm = classify_question(
            state["question"], history=history
        )
    selected_tools = _map_families_to_tools(tool_families)
    required_tools = _map_families_to_tools(required_families)

    policy_applied = "intent_default"
    status = state.get("status", "ok")
    answer = state.get("answer", "")
    requested_skip_llm = _as_bool(context.get("skip_llm"), False)
    requested_force_llm = _as_bool(context.get("force_llm"), False)

    if confidence < INTENT_MIN_CONFIDENCE:
        if LOW_CONFIDENCE_POLICY == "block":
            status = "blocked"
            policy_applied = "low_confidence_block"
            answer = _low_confidence_message(confidence)
            selected_tools = []
            required_tools = []
        elif LOW_CONFIDENCE_POLICY == "ask":
            status = "needs_clarification"
            policy_applied = "low_confidence_ask"
            answer = _low_confidence_message(confidence)
            selected_tools = []
            required_tools = []
        else:
            # Guarded fallback keeps response deterministic and grounded.
            if _is_sql_retrieval_question(_normalize_text(state["question"])):
                intent = "sql"
                tool_families = ["sql"]
                selected_tools = _map_families_to_tools(tool_families)
                required_tools = ["sql_tool"]
                skip_llm = True
                policy_applied = "low_confidence_sql_guarded"
            else:
                intent = "kpi"
                tool_families = ["kpi", "rag"]
                selected_tools = _map_families_to_tools(tool_families)
                required_tools = ["kpi_tool"]
                skip_llm = True
                policy_applied = "low_confidence_kpi_guarded"

    if confidence < LLM_MIN_CONFIDENCE:
        skip_llm = True
        if policy_applied == "intent_default":
            policy_applied = "llm_gated_by_confidence"

    if requested_skip_llm:
        skip_llm = True
        if policy_applied == "intent_default":
            policy_applied = "request_skip_llm"
    elif (
        requested_force_llm
        and status not in {"blocked", "needs_clarification"}
        and not policy_applied.startswith("low_confidence")
        and intent not in DETERMINISTIC_INTENTS
    ):
        skip_llm = False
        if policy_applied in {"intent_default", "llm_gated_by_confidence"}:
            policy_applied = "request_force_llm"

    if intent in DETERMINISTIC_INTENTS:
        skip_llm = True
        if policy_applied == "intent_default":
            policy_applied = "intent_deterministic"

    return {
        "status": status,
        "intent": intent,
        "context": context,
        "intent_confidence": confidence,
        "tool_families": tool_families,
        "selected_tools": selected_tools,
        "required_tools": required_tools,
        "skip_llm": skip_llm,
        "policy_applied": policy_applied,
        "answer": answer,
        "steps": state.get("steps", [])
        + [
            {
                "step": "intent",
                "status": "done",
                "intent": intent,
                "confidence": confidence,
                "tool_families": tool_families,
                "policy_applied": policy_applied,
            }
        ],
    }


async def _run_tools_node(state: AgentState) -> AgentState:
    question = state["question"]
    context = state.get("context", {})
    selected_tools = state.get("selected_tools", [])

    if not selected_tools:
        return {
            "tool_results": [],
            "specialist_reports": [],
            "errors": state.get("errors", []) + ["No selected tools for intent"],
            "steps": state.get("steps", []) + [{"step": "run_tools", "status": "no_tools"}],
        }

    for tool_name in selected_tools:
        _emit({"type": "progress", "step": f"tool:{tool_name}", "label": f"Outil {tool_name} en cours..."})

    tasks = [
        _execute_tool(tool_name=tool_name, question=question, context=context)
        for tool_name in selected_tools
    ]
    results = await asyncio.gather(*tasks)

    errors = list(state.get("errors", []))
    steps = list(state.get("steps", []))
    specialist_reports: list[dict[str, Any]] = []

    for result in results:
        tool_name = str(result.get("tool", "unknown_tool"))
        specialist_report = _build_specialist_report(result)
        specialist_reports.append(specialist_report)
        specialist_name = str(specialist_report.get("agent", "general_specialist"))

        if "error" in result:
            errors.append(f"{tool_name}: {result['error']}")
            steps.append({"step": f"tool:{tool_name}", "status": "error"})
            steps.append({"step": f"agent:{specialist_name}", "tool": tool_name, "status": "error"})
        else:
            steps.append({"step": f"tool:{tool_name}", "status": "ok"})
            steps.append({"step": f"agent:{specialist_name}", "tool": tool_name, "status": "ok"})

    charts, tables = _extract_visual_artifacts(results, intent=state.get("intent"))

    return {
        "tool_results": results,
        "specialist_reports": specialist_reports,
        "errors": errors,
        "charts": charts,
        "tables": tables,
        "steps": steps,
    }


def _validate_node(state: AgentState) -> AgentState:
    required_tools = state.get("required_tools", [])
    tool_results = state.get("tool_results", [])
    invoked_ok = {
        str(result.get("tool"))
        for result in tool_results
        if "error" not in result
    }
    missing_required = [tool for tool in required_tools if tool not in invoked_ok]

    if missing_required:
        errors = state.get("errors", []) + [f"missing required tools: {', '.join(missing_required)}"]
        return {
            "status": "technical_limit",
            "answer": TECHNICAL_LIMIT_MESSAGE,
            "errors": errors,
            "steps": state.get("steps", [])
            + [
                {
                    "step": "validate",
                    "status": "technical_limit",
                    "missing_required": missing_required,
                }
            ],
        }

    return {
        "status": "ready",
        "steps": state.get("steps", []) + [{"step": "validate", "status": "ready"}],
    }


def _deterministic_answer_node(state: AgentState) -> AgentState:
    answer = _compose_deterministic_answer(state)
    return {
        "answer": answer,
        "llm_used": False,
        "steps": state.get("steps", []) + [{"step": "compose_deterministic", "status": "done"}],
    }


def _llm_synthesis_node(state: AgentState) -> AgentState:
    _emit({"type": "progress", "step": "llm", "label": "Génération de la réponse..."})
    if not _has_successful_tool_results(state):
        answer = _compose_deterministic_answer(state)
        return {
            "answer": answer,
            "llm_used": False,
            "steps": state.get("steps", []) + [{"step": "compose_llm", "status": "skipped_no_tools"}],
        }

    system_prompt, user_prompt = _build_llm_prompt(state)
    deterministic_answer = _compose_deterministic_answer(state)
    try:
        answer = _call_ollama_chat(system_prompt, user_prompt)
        if not _is_llm_answer_usable(answer):
            return {
                "answer": deterministic_answer,
                "llm_used": False,
                "steps": state.get("steps", []) + [{"step": "compose_llm", "status": "fallback_weak_output"}],
            }
        return {
            "answer": answer,
            "llm_used": True,
            "steps": state.get("steps", []) + [{"step": "compose_llm", "status": "done"}],
        }
    except Exception as exc:
        fallback_answer = deterministic_answer
        return {
            "answer": fallback_answer,
            "llm_used": False,
            "steps": state.get("steps", [])
            + [{"step": "compose_llm", "status": "fallback", "reason": str(exc)}],
        }


def _route_after_guardrails(state: AgentState) -> str:
    if state.get("status") in {"blocked", "needs_clarification"}:
        return "end"
    return "intent"


def _route_after_intent(state: AgentState) -> str:
    if state.get("status") in {"blocked", "needs_clarification"}:
        return "end"
    return "run_tools"


def _route_after_validate(state: AgentState) -> str:
    status = state.get("status")
    if status in {"technical_limit", "blocked", "needs_clarification", "error"}:
        return "end"
    return "llm" if _should_use_llm(state) else "deterministic"


def _build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("guardrails", _guardrails_node)
    workflow.add_node("intent", _intent_node)
    workflow.add_node("run_tools", _run_tools_node)
    workflow.add_node("validate", _validate_node)
    workflow.add_node("deterministic", _deterministic_answer_node)
    workflow.add_node("llm", _llm_synthesis_node)

    workflow.set_entry_point("guardrails")

    workflow.add_conditional_edges(
        "guardrails",
        _route_after_guardrails,
        {
            "intent": "intent",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "run_tools": "run_tools",
            "end": END,
        },
    )
    workflow.add_edge("run_tools", "validate")
    workflow.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "deterministic": "deterministic",
            "llm": "llm",
            "end": END,
        },
    )
    workflow.add_edge("deterministic", END)
    workflow.add_edge("llm", END)

    return workflow.compile()


AGENT_GRAPH = _build_agent_graph()


async def run_agent_query(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    ctx = dict(context or {})
    history: list[dict[str, Any]] = ctx.pop("history", None) or []

    state: AgentState = {
        "question": question,
        "context": ctx,
        "history": history,
        "status": "ok",
        "tool_results": [],
        "specialist_reports": [],
        "charts": [],
        "tables": [],
        "steps": [],
        "errors": [],
        "llm_used": False,
    }

    final_state = await AGENT_GRAPH.ainvoke(state)
    return {
        "question": question,
        "status": final_state.get("status", "ok"),
        "intent": final_state.get("intent", "kpi"),
        "intent_confidence": final_state.get("intent_confidence", 0.0),
        "tool_families": final_state.get("tool_families", []),
        "policy_applied": final_state.get("policy_applied", "intent_default"),
        "invoked_tools": [
            str(item.get("tool"))
            for item in final_state.get("tool_results", [])
            if isinstance(item, dict)
        ],
        "answer": final_state.get("answer", ""),
        "tool_results": final_state.get("tool_results", []),
        "specialist_reports": final_state.get("specialist_reports", []),
        "guardrails": final_state.get("guardrails", {}),
        "charts": final_state.get("charts", []),
        "tables": final_state.get("tables", []),
        "steps": final_state.get("steps", []),
        "errors": final_state.get("errors", []),
        "llm_used": bool(final_state.get("llm_used", False)),
        "synthesis_mode": "llm" if bool(final_state.get("llm_used", False)) else "deterministic",
    }


def run_agent_query_sync(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return asyncio.run(run_agent_query(question, context=context))


def get_agent_capabilities() -> dict[str, Any]:
    return {
        "orchestrator": "langgraph",
        "architecture": {
            "mode": "hybrid_multi_agent",
            "routing": "heuristic_intent_and_keywords",
            "execution": "parallel_specialist_tools",
            "synthesis": "ollama_llm_with_deterministic_fallback",
        },
        "llm": {
            "provider": "ollama",
            "model": OLLAMA_CHAT_MODEL,
            "host": OLLAMA_HOST,
        },
        "intents": [rule["intent"] for rule in INTENT_RULES],
        "tools": sorted(set(TOOL_FAMILY_TO_TOOL.values())),
        "specialist_agents": sorted(set(TOOL_SPECIALIST_AGENTS.values())),
        "policy": {
            "intent_min_confidence": INTENT_MIN_CONFIDENCE,
            "llm_min_confidence": LLM_MIN_CONFIDENCE,
            "llm_min_answer_chars": LLM_MIN_ANSWER_CHARS,
            "low_confidence_policy": LOW_CONFIDENCE_POLICY,
            "force_deterministic": FORCE_DETERMINISTIC,
            "deterministic_intents": sorted(DETERMINISTIC_INTENTS),
        },
        "guardrails": {
            "domain_filter": "insurance_data_scope",
            "out_of_scope_policy": "block",
        },
    }