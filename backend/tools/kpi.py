from __future__ import annotations

from typing import Any

from db import query_dataframe as _query_dataframe
from utils import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    normalize_text as _normalize_text,
    format_metric_value as _format_metric_value,
)
from config import DATA_YEAR_FROM, DATA_YEAR_TO
from tools._shared import (
    _normalize_branch,
    _resolve_period_context,
    _to_markdown_table,
    _build_chart_payload,
)


def _fetch_kpi_context_postgres(context: dict[str, Any]) -> dict[str, Any]:
    branch = _normalize_branch(context.get("branch"))
    year_from, year_to = _resolve_period_context(context)
    params = {
        "branch": branch,
        "year_from": year_from,
        "year_to": year_to,
    }

    production_sql = """
        SELECT
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(mt_commission), 0) AS total_commission,
            COUNT(*) AS nb_quittances
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
    """

    impaye_sql = """
        SELECT
            COALESCE(SUM(mt_acp), 0) AS total_mt_acp,
            COUNT(*) AS nb_impayes
        FROM dwh_fact_impaye
        WHERE (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
    """

    # NOTE: intentionally queries dim_police WITHOUT year filter to match the dashboard
    # (kpi_router.py churn_sql). The taux_resiliation is a portfolio-state KPI, not a
    # flow KPI — it reflects the current situation of all policies regardless of period.
    churn_sql = """
        SELECT
            COUNT(*) AS total_polices,
            SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_actives,
            SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
            ROUND(
                100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                2
            ) AS taux_resiliation
        FROM dim_police
        WHERE (:branch IS NULL OR branche = :branch)
    """

    top_branch_sql = """
        SELECT
            branche,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND (:branch IS NULL OR branche = :branch)
          AND annee_echeance BETWEEN :year_from AND :year_to
        GROUP BY branche
        ORDER BY total_pnet DESC
        LIMIT 5
    """

    top_resiliation_branch_sql = """
        SELECT
            branche,
            COUNT(*) AS total_polices,
            SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
            ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS taux_resiliation_pct
        FROM dim_police
        WHERE branche IN ('AUTO', 'IRDS', 'SANTE')
          AND (:branch IS NULL OR branche = :branch)
        GROUP BY branche
        ORDER BY taux_resiliation_pct DESC
        LIMIT 5
    """

    production = _query_dataframe(production_sql, params).iloc[0]
    impaye = _query_dataframe(impaye_sql, params).iloc[0]
    churn = _query_dataframe(churn_sql, params).iloc[0]
    top_branches = _query_dataframe(top_branch_sql, params).to_dict(orient="records")
    top_resiliation_branches = _query_dataframe(top_resiliation_branch_sql, params).to_dict(orient="records")

    total_pnet = _safe_float(production["total_pnet"])
    total_commission = _safe_float(production["total_commission"])
    total_mt_acp = _safe_float(impaye["total_mt_acp"])

    # ── Optional dimension KPIs ────────────────────────────────────────────
    vehicule_kpis = {}
    try:
        v_sql = """
            SELECT COUNT(*) AS total_vehicules,
                   COUNT(DISTINCT TRIM(UPPER(v.marque))) AS nb_marques
            FROM dim_vehicule v JOIN dim_police p ON p.id_police = v.id_police
            WHERE (:branch IS NULL OR p.branche = :branch)
        """
        row = _query_dataframe(v_sql, params).iloc[0]
        vehicule_kpis = {
            "vehicule_total": _safe_int(row["total_vehicules"]),
            "vehicule_nb_marques": _safe_int(row["nb_marques"]),
        }
    except Exception:
        pass

    sinistre_kpis = {}
    try:
        s_sql = """
            SELECT COUNT(*) AS total_sinistres,
                   COALESCE(SUM(mt_paye), 0) AS total_mt_paye,
                   COALESCE(SUM(mt_evaluation), 0) AS total_provisions
            FROM dwh_fact_sinistre
            WHERE (:branch IS NULL OR branche = :branch)
              AND annee_survenance BETWEEN :year_from AND :year_to
        """
        row = _query_dataframe(s_sql, params).iloc[0]
        sinistre_kpis = {
            "nb_sinistres": _safe_int(row["total_sinistres"]),
            "total_mt_paye_sinistres": _safe_float(row["total_mt_paye"]),
            "total_provisions": _safe_float(row["total_provisions"]),
        }
    except Exception:
        pass

    client_kpis = {}
    try:
        c_sql = """
            SELECT COUNT(*) AS total_clients,
                   COUNT(DISTINCT ville) FILTER (WHERE ville IS NOT NULL AND TRIM(ville) != '') AS nb_villes
            FROM dim_client
        """
        row = _query_dataframe(c_sql, {}).iloc[0]
        client_kpis = {
            "client_total": _safe_int(row["total_clients"]),
            "client_nb_villes": _safe_int(row["nb_villes"]),
        }
    except Exception:
        pass

    police_kpis = {}
    try:
        pol_sql = """
            SELECT COUNT(*) AS total_polices,
                   SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_en_vigueur,
                   SUM(CASE WHEN type_police = 'flotte' THEN 1 ELSE 0 END) AS polices_flotte
            FROM dim_police WHERE (:branch IS NULL OR branche = :branch)
        """
        row = _query_dataframe(pol_sql, params).iloc[0]
        police_kpis = {
            "police_total": _safe_int(row["total_polices"]),
            "police_en_vigueur": _safe_int(row["polices_en_vigueur"]),
            "police_flotte": _safe_int(row["polices_flotte"]),
        }
    except Exception:
        pass

    agent_kpis = {}
    try:
        a_sql = """
            SELECT COUNT(*) AS total_agents,
                   SUM(CASE WHEN etat_agent = 'A' THEN 1 ELSE 0 END) AS agents_actifs
            FROM dim_agent
        """
        row = _query_dataframe(a_sql, {}).iloc[0]
        agent_kpis = {
            "agent_total": _safe_int(row["total_agents"]),
            "agent_actifs": _safe_int(row["agents_actifs"]),
        }
    except Exception:
        pass

    produit_kpis = {}
    try:
        prod_sql = """
            SELECT COUNT(DISTINCT code_produit) AS nb_produits,
                   COUNT(DISTINCT famille_risque) AS nb_familles
            FROM dim_produit WHERE (:branch IS NULL OR branche = :branch)
        """
        row = _query_dataframe(prod_sql, params).iloc[0]
        produit_kpis = {
            "produit_total": _safe_int(row["nb_produits"]),
            "produit_nb_familles": _safe_int(row["nb_familles"]),
        }
    except Exception:
        pass

    # ── Ratio Combiné RÉEL = (sinistres_payés + commission) / prime_nette ──
    # Formule identique à kpi_router.py (dashboard)
    total_mt_paye_sinistres = _safe_float(sinistre_kpis.get("total_mt_paye_sinistres", 0.0))
    ratio_combine_pct = round(
        100.0 * (total_mt_paye_sinistres + total_commission) / total_pnet, 2
    ) if total_pnet > 0 else 0.0
    # S/P pur (sinistres payés seulement / prime nette)
    sp_ratio_reel_pct = round(
        (100.0 * total_mt_paye_sinistres / total_pnet), 2
    ) if total_pnet > 0 else 0.0
    # Proxy impayé (gardé pour rétrocompat frontend)
    sp_ratio_proxy_pct = round(
        (100.0 * total_mt_acp / total_pnet), 2
    ) if total_pnet > 0 else 0.0

    return {
        "source": "postgres",
        "branch": branch or "ALL",
        "year_from": year_from,
        "year_to": year_to,
        "total_pnet": total_pnet,
        "total_commission": total_commission,
        "nb_quittances": _safe_int(production["nb_quittances"]),
        "total_mt_acp": total_mt_acp,
        "nb_impayes": _safe_int(impaye["nb_impayes"]),
        # Ratio Combiné = (sinistres payés + commission) / prime nette
        "ratio_combine_pct": ratio_combine_pct,
        # S/P pur sinistres
        "sp_ratio_reel_pct": sp_ratio_reel_pct,
        # Proxy impayé (rétrocompat)
        "sp_ratio_proxy_pct": sp_ratio_proxy_pct,
        "taux_resiliation_pct": _safe_float(churn["taux_resiliation"]),
        "total_polices_churn": _safe_int(churn["total_polices"]),
        "polices_resiliees": _safe_int(churn["polices_resiliees"]),
        "top_branches": top_branches,
        "top_resiliation_branches": top_resiliation_branches,
        **vehicule_kpis,
        **sinistre_kpis,
        **client_kpis,
        **police_kpis,
        **agent_kpis,
        **produit_kpis,
    }


def _detect_kpi_focus(question: str) -> str:
    """Detect which single KPI the user is asking about. Returns focus key."""
    q = _normalize_text(question)
    if any(t in q for t in ["ratio combine", "ratio combin", "combined ratio", "combined"]):
        return "ratio_combine"
    if any(t in q for t in ["resiliation", "churn", "resilie"]):
        return "resiliation"
    if any(t in q for t in ["impaye", "impayes", "recouvrement"]):
        return "impaye"
    if any(t in q for t in ["sinistre", "sinistres", "s/p", "sp ratio", "sp reel"]):
        return "sinistre"
    if any(t in q for t in ["commission"]):
        return "commission"
    if any(t in q for t in ["prime", "pnet", "production", "quittance"]):
        return "prime"
    return "overview"  # full overview — no specific focus


def kpi_tool(question: str, context: dict[str, Any]) -> dict[str, Any]:
    payload = _fetch_kpi_context_postgres(context)
    payload["source"] = "postgres"

    focus = _detect_kpi_focus(question)
    ratio_combine = _safe_float(payload.get("ratio_combine_pct"), 0.0)
    sp_reel = _safe_float(payload.get("sp_ratio_reel_pct"), 0.0)
    taux_resiliation = _safe_float(payload.get("taux_resiliation_pct"), 0.0)
    total_pnet = _safe_float(payload.get("total_pnet"), 0.0)
    total_commission = _safe_float(payload.get("total_commission"), 0.0)

    # ── Build focused KPIs based on what was asked ─────────────────────────
    if focus == "ratio_combine":
        focused_kpis = [
            {"key": "ratio_combine_pct",      "label": "Ratio Combiné",          "value": ratio_combine,    "unit": "%"},
            {"key": "sp_ratio_reel_pct",       "label": "S/P pur (sinistres/pnet)","value": sp_reel,         "unit": "%"},
            {"key": "total_mt_paye_sinistres", "label": "Sinistres payés",         "value": _safe_float(payload.get("total_mt_paye_sinistres"), 0.0), "unit": "TND"},
            {"key": "total_commission",        "label": "Commission",              "value": total_commission,  "unit": "TND"},
            {"key": "total_pnet",              "label": "Prime nette",             "value": total_pnet,        "unit": "TND"},
        ]
        summary = (
            f"Ratio Combiné {payload['branch']} {payload['year_from']}-{payload['year_to']}: "
            f"{ratio_combine:.2f}% "
            f"(S/P pur {sp_reel:.2f}% + expense {100.0*total_commission/total_pnet:.2f}% si pnet>0)."
        )
        charts = []  # no chart needed for a single ratio
        tables = [{
            "title": "Ratio Combiné",
            "columns": ["kpi", "valeur"],
            "rows": [
                {"kpi": "Ratio Combiné",            "valeur": f"{ratio_combine:.2f}%"},
                {"kpi": "S/P pur",                  "valeur": f"{sp_reel:.2f}%"},
                {"kpi": "Expense ratio",            "valeur": f"{100.0*total_commission/total_pnet:.2f}%" if total_pnet > 0 else "N/A"},
                {"kpi": "Sinistres payés",          "valeur": f"{_safe_float(payload.get('total_mt_paye_sinistres'),0):,.0f} TND"},
                {"kpi": "Commission",               "valeur": f"{total_commission:,.0f} TND"},
                {"kpi": "Prime nette",              "valeur": f"{total_pnet:,.0f} TND"},
            ],
            "markdown": "",
        }]

    elif focus == "resiliation":
        focused_kpis = [
            {"key": "taux_resiliation_pct", "label": "Taux de résiliation", "value": taux_resiliation, "unit": "%"},
            {"key": "polices_resiliees",    "label": "Polices résiliées",   "value": _safe_float(payload.get("polices_resiliees"), 0.0), "unit": "count"},
            {"key": "total_polices_churn",  "label": "Polices totales",     "value": _safe_float(payload.get("total_polices_churn"), 0.0), "unit": "count"},
        ]
        summary = f"Taux résiliation {payload['branch']} {payload['year_from']}-{payload['year_to']}: {taux_resiliation:.2f}%."
        charts = [_build_chart_payload("bar", "Résiliation par branche", "branche", "polices_resiliees",
                                        list(payload.get("top_resiliation_branches", [])))]
        tables = []

    elif focus == "sinistre":
        nb_sin = _safe_float(payload.get("nb_sinistres"), 0.0)
        mt_paye = _safe_float(payload.get("total_mt_paye_sinistres"), 0.0)
        focused_kpis = [
            {"key": "sp_ratio_reel_pct",       "label": "S/P réel",          "value": sp_reel,   "unit": "%"},
            {"key": "nb_sinistres",             "label": "Nb sinistres",       "value": nb_sin,    "unit": "count"},
            {"key": "total_mt_paye_sinistres",  "label": "Montant payé",       "value": mt_paye,   "unit": "TND"},
        ]
        summary = f"Sinistres {payload['branch']} {payload['year_from']}-{payload['year_to']}: {int(nb_sin):,} sinistres, S/P réel {sp_reel:.2f}%."
        charts = []
        tables = []

    elif focus == "impaye":
        nb_imp = _safe_float(payload.get("nb_impayes"), 0.0)
        mt_acp = _safe_float(payload.get("total_mt_acp"), 0.0)
        sp_proxy = _safe_float(payload.get("sp_ratio_proxy_pct"), 0.0)
        focused_kpis = [
            {"key": "nb_impayes",      "label": "Nb impayes",    "value": nb_imp,  "unit": "count"},
            {"key": "total_mt_acp",    "label": "Montant ACP",   "value": mt_acp,  "unit": "TND"},
            {"key": "taux_impaye_pnet", "label": "Taux impayé/prime nette", "value": sp_proxy, "unit": "%"},
        ]
        summary = f"Impayes {payload['branch']} {payload['year_from']}-{payload['year_to']}: {int(nb_imp):,} impayes, ACP {mt_acp:,.0f} TND."
        charts = []
        tables = []

    elif focus == "prime":
        nb_quitt = _safe_float(payload.get("nb_quittances"), 0.0)
        focused_kpis = [
            {"key": "total_pnet",    "label": "Prime nette",      "value": total_pnet,    "unit": "TND"},
            {"key": "total_commission","label": "Commission",     "value": total_commission,"unit": "TND"},
            {"key": "nb_quittances", "label": "Nb quittances",    "value": nb_quitt,       "unit": "count"},
        ]
        summary = f"Production {payload['branch']} {payload['year_from']}-{payload['year_to']}: prime nette {total_pnet:,.0f} TND, {int(nb_quitt):,} quittances."
        charts = [_build_chart_payload("bar", "Top branches par prime nette", "branche", "total_pnet",
                                        list(payload.get("top_branches", [])))]
        tables = []

    else:
        # Full overview — show everything
        focused_kpis = [
            {"key": "total_pnet",           "label": "Prime nette",     "value": total_pnet,      "unit": "TND"},
            {"key": "ratio_combine_pct",    "label": "Ratio Combiné",   "value": ratio_combine,   "unit": "%"},
            {"key": "taux_resiliation_pct", "label": "Taux résiliation","value": taux_resiliation,"unit": "%"},
            {"key": "nb_sinistres",         "label": "Sinistres",       "value": _safe_float(payload.get("nb_sinistres"),0.0), "unit": "count"},
            {"key": "nb_impayes",           "label": "Impayes",         "value": _safe_float(payload.get("nb_impayes"),0.0),  "unit": "count"},
        ]
        parts = [
            f"prime nette {total_pnet:,.0f} TND",
            f"ratio combine {ratio_combine:.2f}%",
            f"resiliation {taux_resiliation:.2f}%",
        ]
        summary = f"Vue globale KPI {payload['branch']} {payload['year_from']}-{payload['year_to']}: {', '.join(parts)}."
        charts = [_build_chart_payload("bar", "Top branches par prime nette", "branche", "total_pnet",
                                        list(payload.get("top_branches", [])))]
        overview_row = {
            "branch":               payload["branch"],
            "periode":              f"{payload['year_from']}-{payload['year_to']}",
            "prime_nette":          round(total_pnet, 0),
            "ratio_combine_pct":    ratio_combine,
            "taux_resiliation_pct": round(taux_resiliation, 2),
            "nb_quittances":        payload.get("nb_quittances", 0),
            "nb_sinistres":         payload.get("nb_sinistres", 0),
            "nb_impayes":           payload.get("nb_impayes", 0),
        }
        tables = [{
            "title": "Synthese KPI",
            "columns": list(overview_row.keys()),
            "rows": [overview_row],
            "markdown": _to_markdown_table(list(overview_row.keys()), [overview_row]),
        }]

    # Attach focused_kpis to payload so _compose_precise_metric_answer can read them
    payload["kpis"] = focused_kpis
    payload["focus"] = focus

    return {
        "tool": "kpi_tool",
        "summary": summary,
        "payload": payload,
        "charts": charts,
        "tables": tables,
    }
