from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata
from typing import Any, TypedDict

import requests
from langgraph.graph import END, StateGraph

from agent_tools import run_tool


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


OUT_OF_SCOPE_MESSAGE = (
    "Question hors scope assurance/data. Je peux aider sur KPI, risques, impayes, "
    "clients, forecast, anomalies, drift ou analyse SQL metier."
)

TECHNICAL_LIMIT_MESSAGE = (
    "Limitation technique: un outil requis n'a pas pu etre execute correctement. "
    "Je fournis un resultat partiel et les details de limitation."
)

DOMAIN_KEYWORDS = {
    "assurance",
    "police",
    "prime",
    "commission",
    "impaye",
    "resiliation",
    "sinistre",
    "agent",
    "branche",
    "auto",
    "irds",
    "sante",
    "client",
    "kpi",
    "drift",
    "anomal",
    "forecast",
    "prevision",
    "risque",
    "gouvernorat",
    "sql",
    "model",
    "ml",
    "rag",
    "derive",
    "stabilite",
    "global",        # "vue globale"
    "situation",     # "situation globale"
    "synthese",
    "tableau de bord",
    "dashboard",
    "bilan",
    "rapport",
}

INTENT_MIN_CONFIDENCE = float(os.getenv("AGENT_INTENT_MIN_CONFIDENCE", "0.62"))
LLM_MIN_CONFIDENCE = float(os.getenv("AGENT_LLM_MIN_CONFIDENCE", "0.62"))
LOW_CONFIDENCE_POLICY = os.getenv("AGENT_LOW_CONFIDENCE_POLICY", "ask").strip().lower()
FORCE_DETERMINISTIC = os.getenv("AGENT_FORCE_DETERMINISTIC", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
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
        # Removed "risque" and "resilie" — too generic; they match KPI/SQL questions.
        # Removed "predi" — too broad; "prediction" without "fraude"/"ml" should go to forecast.
        # Only trigger predict when the user explicitly asks for ML model execution.
        "keywords": ["fraude", "modele ml", "ml predict", "random forest", "gradient boosting", "ml model", "machine learning", "fais tourner le modele", "lancer le modele", "modelisation", "modele de prediction"],
        "tool_families": ["predict", "kpi"],
        "required": ["predict"],
        "skip_llm": False,
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
            # Note: "sinistre" and "sinistres" removed — routing to sql_tool is more accurate
            # since there is no dedicated sinistres table (proxy via dwh_fact_impaye).
            # Note: "prediction"/"predic" removed — too close to ml_predict intent.
            # Note: "tendance" removed — "tendance" alone should go to sql (evolution query).
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
        "intent": "sql",
        "keywords": [
            "sql",
            "requete",
            "query",
            "table",
            "tableau",
            "lignes",
            "dataset",
            "base de donnees",
            "bdd",
            "gouvernorat",
            "graphique",
            "graphe",
            "chart",
            "visualisation",
            "dataviz",
            "top",
            "classement",
            "evolution",
            "trend",
        ],
        "tool_families": ["sql"],
        "required": ["sql"],
        "skip_llm": True,
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
        "keywords": ["regle", "documentation", "policy", "rag", "contexte"],
        "tool_families": ["rag", "kpi"],
        "required": ["rag"],
        "skip_llm": False,
    },
    {
        "intent": "kpi",
        # Added "sinistre"/"risque" — when used without forecast/predictive verbs, they are KPI queries.
        "keywords": ["kpi", "prime", "commission", "resiliation", "performance", "sinistre", "sinistres", "risque", "taux"],
        "tool_families": ["kpi"],
        "required": ["kpi"],
        "skip_llm": False,
    },
]


class AgentState(TypedDict, total=False):
    question: str
    context: dict[str, Any]
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


def _normalize_text(text_value: str) -> str:
    text_value = text_value.replace("\u2019", "'")
    ascii_normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", text_value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", ascii_normalized.strip().lower())


def _keyword_score(normalized_question: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in normalized_question)


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
        "prime",
        "pnet",
        "commission",
        "impaye",
        "resiliation",
        "annulation",
        "sinistre",
        "client",
        "police",
        "quittance",
        "montant",
    ]
    dimension_terms = [
        "branche",
        "gouvernorat",
        "mois",
        "mensuel",
        "annee",
        "periode",
        "agent",
        "ville",
        "zone",
        "segment",
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

    return False


def is_domain_question(question: str) -> bool:
    normalized = _normalize_text(question)
    return any(keyword in normalized for keyword in DOMAIN_KEYWORDS)


def classify_question(question: str) -> tuple[str, float, list[str], list[str], bool]:
    normalized = _normalize_text(question)
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
        asks_explicit_sql = any(token in normalized for token in explicit_sql_cues)
        if not asks_explicit_sql:
            scored = [
                (score, rule)
                for score, rule in scored
                if str(rule.get("intent")) != "sql"
            ]

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

    # Merge one family from additional matched intents to support multi-signal questions.
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def _format_amount_tnd(value: float) -> str:
    return f"{value:,.0f} TND"


def _format_percent(value: float) -> str:
    return f"{value:.2f}%"


def _format_metric_value(value: float, unit: str) -> str:
    normalized_unit = unit.strip().upper()
    if normalized_unit == "TND":
        return _format_amount_tnd(value)
    if normalized_unit == "%":
        return _format_percent(value)
    if normalized_unit == "COUNT":
        return f"{int(round(value)):,.0f}"
    return f"{value:,.2f}"


def _format_branch_label(branch_value: Any) -> str:
    branch = str(branch_value or "ALL").upper()
    return "toutes les branches" if branch == "ALL" else f"la branche {branch}"


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
            total_pnet = _as_float(payload.get("total_pnet"), 0.0)
            total_commission = _as_float(payload.get("total_commission"), 0.0)
            sp_ratio = _as_float(payload.get("sp_ratio_proxy_pct"), 0.0)
            taux_resiliation = _as_float(payload.get("taux_resiliation_pct"), 0.0)

            _append_unique(
                synthese_items,
                (
                    f"KPI sur {period_label} ({branch_label}): prime nette {_format_amount_tnd(total_pnet)}, "
                    f"S/P proxy {_format_percent(sp_ratio)}, resiliation {_format_percent(taux_resiliation)}."
                ),
            )
            _append_unique(chiffres_cles, f"- Prime nette: {_format_amount_tnd(total_pnet)}")
            _append_unique(chiffres_cles, f"- Commission: {_format_amount_tnd(total_commission)}")
            _append_unique(chiffres_cles, f"- Ratio S/P proxy: {_format_percent(sp_ratio)}")
            _append_unique(chiffres_cles, f"- Taux de resiliation: {_format_percent(taux_resiliation)}")

            if taux_resiliation >= 8.0:
                _append_unique(decision_items, "Risque de retention eleve: la resiliation est au-dessus de 8%.")
                _append_unique(actions, "Lancer un plan anti-churn cible sur les segments a forte resiliation.")
            elif taux_resiliation >= 4.0:
                _append_unique(decision_items, "Risque de retention modere: la resiliation doit etre surveillee.")
                _append_unique(actions, "Mettre en place un suivi hebdomadaire des motifs d annulation par branche.")
            else:
                _append_unique(decision_items, "Retention globalement maitrisee sur la periode analysee.")

            if sp_ratio >= 2.0:
                _append_unique(decision_items, "Pression risque elevee: le S/P proxy depasse 2%." )
                _append_unique(actions, "Prioriser les actions de recouvrement sur les portefeuilles les plus exposes.")

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
    if not decision_items:
        decision_items = ["Situation globalement stable sur les indicateurs disponibles."]
    if not actions:
        actions = ["Maintenir un suivi mensuel des KPI critiques et des ecarts vs objectifs."]

    lines: list[str] = []
    lines.append("**Synthese decisionnelle**")
    lines.append(" ".join(synthese_items[:3]))

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
    question = _normalize_text(str(state.get("question", "")))
    if not ("taux" in question and "resiliation" in question):
        return None

    kpi_result = next(
        (
            result
            for result in state.get("tool_results", [])
            if isinstance(result, dict) and result.get("tool") == "kpi_tool" and "error" not in result
        ),
        None,
    )
    if not kpi_result:
        return None

    payload = kpi_result.get("payload", {}) if isinstance(kpi_result, dict) else {}
    if not isinstance(payload, dict):
        return None

    taux_resiliation = _as_float(payload.get("taux_resiliation_pct"), 0.0)
    year_from = payload.get("year_from")
    year_to = payload.get("year_to")
    if isinstance(year_from, int) and isinstance(year_to, int):
        period_label = str(year_from) if year_from == year_to else f"{year_from}-{year_to}"
    else:
        period_label = "la periode demandee"

    branch = str(payload.get("branch", "ALL")).upper()
    branch_label = "toutes les branches" if branch == "ALL" else f"la branche {branch}"
    source = str(payload.get("source", "postgres"))

    return (
        f"Le taux de resiliation pour {period_label} sur {branch_label} est de {taux_resiliation:.2f}%.\n"
        f"Source: {source}."
    )


def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "35"))

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    content = (payload.get("message") or {}).get("content", "")
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
        "\n**Contexte**: (Resume la demande et la source des donnees, ex: 'Basé sur les requêtes SQL et la documentation interne...')\n"
        "\n**Analyse**: (Analyse les donnees recuperees depuis les outils et le brouillon. Explique les tendances et les chiffres.)\n"
        "\n**Decision**: (Propose des recommandations metier basees sur l'analyse, ex: 'Il est recommande de cibler telle branche...')\n"
        "\n**Graphs/Tableaux**: (Si tu as recu des donnees formattes ou de serie temporelle, presente un resume tabulaire ou une suggestion claire du graphique a tracer.)\n"
        "Respecte strictement cette structure à 4 sections."
        "Si un outil rag_tool a retourné des documents, tu DOIS citer leur contenu en priorité "
        "pour répondre aux questions sur les règles, définitions ou politiques. "
        "Ne réponds PAS avec des chiffres KPI si la question est une question de définition.\n"
    )

    user_prompt = (
        f"Question utilisateur: {question}\n"
        f"Intent detecte: {state.get('intent')} (confiance={state.get('intent_confidence')})\n"
        f"Contexte: {_compact_json(context_payload, max_chars=900)}\n"
        f"Rapports agents specialises: {_compact_json(specialist_context, max_chars=4000)}\n"
        f"Sorties outils: {_compact_json(tool_context, max_chars=7000)}\n"
        f"Brouillon deterministe fiable: {_compact_json(grounded_draft, max_chars=3000)}\n"
        "Instruction: produire une synthese plus lisible que le brouillon, sans perdre les faits."
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

    intent, confidence, tool_families, required_families, skip_llm = classify_question(state["question"])
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
    state: AgentState = {
        "question": question,
        "context": context or {},
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
            "model": os.getenv("OLLAMA_CHAT_MODEL", "llama3"),
            "host": os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
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