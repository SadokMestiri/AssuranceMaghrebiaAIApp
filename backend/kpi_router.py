from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from db import get_db

router = APIRouter(prefix="/kpis", tags=["kpis"])

VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}
YEAR_MIN = 2019
YEAR_MAX = 2026
SITUATION_LABELS = {
    "V": "Valide",
    "R": "Resiliee",
    "T": "Terminee",
    "S": "Suspendue",
    "A": "Annulee",
}
ETAT_QUITTANCE_LABELS = {
    "E": "Emise",
    "P": "Payee",
    "A": "Annulee",
    "I": "Impayee",
}


def _normalize_branch(branch: str | None) -> str | None:
    if branch is None:
        return None
    normalized = branch.strip().upper()
    if normalized not in VALID_BRANCHES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid branch '{branch}'. Use one of {sorted(VALID_BRANCHES)}",
        )
    return normalized


def _to_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _to_int(value: Any) -> int:
    return int(value) if value is not None else 0


def _validate_year_range(year_from: int | None, year_to: int | None) -> None:
    if year_from is not None and not YEAR_MIN <= year_from <= YEAR_MAX:
        raise HTTPException(status_code=400, detail=f"year_from must be between {YEAR_MIN} and {YEAR_MAX}")
    if year_to is not None and not YEAR_MIN <= year_to <= YEAR_MAX:
        raise HTTPException(status_code=400, detail=f"year_to must be between {YEAR_MIN} and {YEAR_MAX}")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise HTTPException(status_code=400, detail="year_from must be <= year_to")


@router.get("/overview")
def get_overview(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    params = {
        "branch": normalized_branch,
        "year_from": year_from,
        "year_to": year_to,
    }

    emission_sql = text(
        """
        SELECT
            COUNT(*) FILTER (WHERE etat_quit IN ('E','P','A')) AS nb_quittances,
            COALESCE(SUM(mt_pnet) FILTER (WHERE etat_quit IN ('E','P','A')), 0) AS total_pnet,
            COALESCE(SUM(mt_ptt) FILTER (WHERE etat_quit IN ('E','P','A')), 0) AS total_ptt,
            COALESCE(SUM(mt_commission) FILTER (WHERE etat_quit IN ('E','P','A')), 0) AS total_commission,
            COALESCE(AVG(mt_pnet) FILTER (WHERE etat_quit IN ('E','P','A') AND mt_pnet > 0), 0) AS avg_pnet
        FROM dwh_fact_emission
        WHERE (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        """
    )

    churn_sql = text(
        """
        SELECT
            COUNT(*) AS total_polices,
            SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) AS polices_actives,
            SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) AS polices_resiliees,
            ROUND(100.0 * SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS taux_churn_pct
        FROM dim_police
        WHERE (:branch IS NULL OR branche = :branch)
        """
    )

    impaye_sql = text(
        """
        SELECT
            COUNT(*) AS nb_impayes,
            COALESCE(SUM(mt_acp), 0) AS total_mt_acp,
            COALESCE(AVG(mt_acp), 0) AS avg_mt_acp
        FROM dwh_fact_impaye
        WHERE (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        """
    )

    emission = db.execute(emission_sql, params).mappings().first()
    churn = db.execute(churn_sql, params).mappings().first()
    impaye = db.execute(impaye_sql, params).mappings().first()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "production": {
            "nb_quittances": _to_int(emission["nb_quittances"]),
            "total_pnet": _to_float(emission["total_pnet"]),
            "total_ptt": _to_float(emission["total_ptt"]),
            "total_commission": _to_float(emission["total_commission"]),
            "avg_pnet": _to_float(emission["avg_pnet"]),
        },
        "portefeuille": {
            "total_polices": _to_int(churn["total_polices"]),
            "polices_actives": _to_int(churn["polices_actives"]),
            "polices_resiliees": _to_int(churn["polices_resiliees"]),
            "taux_churn_pct": _to_float(churn["taux_churn_pct"]),
        },
        "impayes": {
            "nb_impayes": _to_int(impaye["nb_impayes"]),
            "total_mt_acp": _to_float(impaye["total_mt_acp"]),
            "avg_mt_acp": _to_float(impaye["avg_mt_acp"]),
        },
    }


@router.get("/primes/by-branch")
def get_primes_by_branch(
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            branche,
            COUNT(*) AS nb_quittances,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(mt_ptt), 0) AS total_ptt,
            COALESCE(SUM(mt_commission), 0) AS total_commission
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND mt_pnet >= 0
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY branche
        ORDER BY total_pnet DESC
        """
    )

    rows = db.execute(sql, {"year_from": year_from, "year_to": year_to}).mappings().all()

    return {
        "filters": {"year_from": year_from, "year_to": year_to},
        "items": [
            {
                "branche": row["branche"],
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_ptt": _to_float(row["total_ptt"]),
                "total_commission": _to_float(row["total_commission"]),
            }
            for row in rows
        ],
    }


@router.get("/top-agents")
def get_top_agents(
    limit: int = Query(default=10, ge=1, le=100),
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            e.id_agent,
            COALESCE(a.code_agent, 'N/A') AS code_agent,
            COALESCE(a.nom_agent, 'N/A') AS nom_agent,
            COALESCE(a.groupe_agent, 'N/A') AS groupe_agent,
            COUNT(*) AS nb_quittances,
            COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(e.mt_commission), 0) AS total_commission
        FROM dwh_fact_emission e
        LEFT JOIN dim_agent a ON a.id_agent = e.id_agent
        WHERE e.etat_quit IN ('E','P','A')
          AND e.mt_pnet >= 0
          AND (:branch IS NULL OR e.branche = :branch)
          AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
          AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
        GROUP BY e.id_agent, a.code_agent, a.nom_agent, a.groupe_agent
        ORDER BY total_pnet DESC
        LIMIT :limit
        """
    )

    rows = db.execute(
        sql,
        {
            "limit": limit,
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "limit": limit,
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "id_agent": _to_int(row["id_agent"]),
                "code_agent": row["code_agent"],
                "nom_agent": row["nom_agent"],
                "groupe_agent": row["groupe_agent"],
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_commission": _to_float(row["total_commission"]),
            }
            for row in rows
        ],
    }


@router.get("/impayes/by-branch")
def get_impayes_by_branch(
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            branche,
            COUNT(*) AS nb_impayes,
            COALESCE(SUM(mt_acp), 0) AS total_mt_acp,
            COALESCE(AVG(mt_acp), 0) AS avg_mt_acp
        FROM dwh_fact_impaye
        WHERE (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY branche
        ORDER BY total_mt_acp DESC
        """
    )

    rows = db.execute(sql, {"year_from": year_from, "year_to": year_to}).mappings().all()

    return {
        "filters": {"year_from": year_from, "year_to": year_to},
        "items": [
            {
                "branche": row["branche"],
                "nb_impayes": _to_int(row["nb_impayes"]),
                "total_mt_acp": _to_float(row["total_mt_acp"]),
                "avg_mt_acp": _to_float(row["avg_mt_acp"]),
            }
            for row in rows
        ],
    }


@router.get("/production/monthly-trend")
def get_production_monthly_trend(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            annee_echeance,
            mois_echeance,
            branche,
            COUNT(*) AS nb_quittances,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(mt_ptt), 0) AS total_ptt
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND mt_pnet >= 0
          AND mois_echeance BETWEEN 1 AND 12
          AND (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY annee_echeance, mois_echeance, branche
        ORDER BY annee_echeance, mois_echeance, branche
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    items = [
        {
            "annee": _to_int(row["annee_echeance"]),
            "mois": _to_int(row["mois_echeance"]),
            "periode": f"{_to_int(row['annee_echeance']):04d}-{_to_int(row['mois_echeance']):02d}",
            "branche": row["branche"],
            "nb_quittances": _to_int(row["nb_quittances"]),
            "total_pnet": _to_float(row["total_pnet"]),
            "total_ptt": _to_float(row["total_ptt"]),
        }
        for row in rows
    ]

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": items,
    }


@router.get("/portfolio/situation")
def get_portfolio_situation(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)

    sql = text(
        """
        SELECT
            situation,
            COUNT(*) AS nb_polices
        FROM dim_police
        WHERE (:branch IS NULL OR branche = :branch)
        GROUP BY situation
        ORDER BY nb_polices DESC
        """
    )

    rows = db.execute(sql, {"branch": normalized_branch}).mappings().all()

    return {
        "filters": {"branch": normalized_branch},
        "items": [
            {
                "situation": row["situation"],
                "label": SITUATION_LABELS.get(str(row["situation"]), str(row["situation"])),
                "nb_polices": _to_int(row["nb_polices"]),
            }
            for row in rows
        ],
    }


@router.get("/churn/by-branch")
def get_churn_by_branch(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)

    sql = text(
        """
        SELECT
            branche,
            COUNT(*) AS total_polices,
            SUM((situation = 'V')::int) AS polices_actives,
            SUM((situation = 'R')::int) AS polices_resiliees,
            ROUND(100.0 * SUM((situation = 'R')::int) / NULLIF(COUNT(*), 0), 2) AS taux_churn_pct
        FROM dim_police
        WHERE branche IN ('AUTO', 'IRDS', 'SANTE')
          AND (:branch IS NULL OR branche = :branch)
        GROUP BY branche
        ORDER BY branche
        """
    )

    rows = db.execute(sql, {"branch": normalized_branch}).mappings().all()
    return {
        "filters": {"branch": normalized_branch},
        "items": [
            {
                "branche": row["branche"],
                "total_polices": _to_int(row["total_polices"]),
                "polices_actives": _to_int(row["polices_actives"]),
                "polices_resiliees": _to_int(row["polices_resiliees"]),
                "taux_churn_pct": _to_float(row["taux_churn_pct"]),
            }
            for row in rows
        ]
    }


@router.get("/commissions/by-branch")
def get_commissions_by_branch(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            branche,
            COALESCE(SUM(mt_commission), 0) AS total_commission,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            ROUND(100.0 * COALESCE(SUM(mt_commission), 0) / NULLIF(COALESCE(SUM(mt_pnet), 0), 0), 2) AS taux_commission_pct
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND mt_pnet > 0
          AND (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY branche
        ORDER BY total_commission DESC
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "branche": row["branche"],
                "total_commission": _to_float(row["total_commission"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "taux_commission_pct": _to_float(row["taux_commission_pct"]),
            }
            for row in rows
        ],
    }


@router.get("/agents/by-group")
def get_agents_by_group(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            COALESCE(a.groupe_agent, 'N/A') AS groupe_agent,
            COUNT(DISTINCT a.id_agent) AS nb_agents,
            COUNT(e.num_quittance) AS nb_quittances,
            COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(e.mt_commission), 0) AS total_commission
        FROM dim_agent a
        LEFT JOIN dwh_fact_emission e
          ON e.id_agent = a.id_agent
         AND e.etat_quit IN ('E','P','A')
         AND e.mt_pnet >= 0
         AND (:branch IS NULL OR e.branche = :branch)
         AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
         AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
        GROUP BY COALESCE(a.groupe_agent, 'N/A')
        ORDER BY total_pnet DESC, nb_agents DESC
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "groupe_agent": row["groupe_agent"],
                "nb_agents": _to_int(row["nb_agents"]),
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_commission": _to_float(row["total_commission"]),
            }
            for row in rows
        ],
    }


@router.get("/production/by-state")
def get_production_by_state(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            etat_quit,
            COUNT(*) AS nb_quittances,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(mt_ptt), 0) AS total_ptt
        FROM dwh_fact_emission
        WHERE (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY etat_quit
        ORDER BY nb_quittances DESC
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "etat_quit": row["etat_quit"],
                "label": ETAT_QUITTANCE_LABELS.get(str(row["etat_quit"]), str(row["etat_quit"])),
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_ptt": _to_float(row["total_ptt"]),
            }
            for row in rows
        ],
    }


@router.get("/production/yearly")
def get_production_yearly(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            annee_echeance,
            COUNT(*) AS nb_quittances,
            COALESCE(SUM(mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(mt_ptt), 0) AS total_ptt,
            COALESCE(SUM(mt_commission), 0) AS total_commission
        FROM dwh_fact_emission
        WHERE etat_quit IN ('E','P','A')
          AND mt_pnet >= 0
          AND (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY annee_echeance
        ORDER BY annee_echeance
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "annee": _to_int(row["annee_echeance"]),
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_ptt": _to_float(row["total_ptt"]),
                "total_commission": _to_float(row["total_commission"]),
            }
            for row in rows
        ],
    }


@router.get("/production/yoy")
def get_production_yoy(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        WITH yearly AS (
            SELECT
                annee_echeance,
                COUNT(*) AS nb_quittances,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet,
                COALESCE(SUM(mt_ptt), 0) AS total_ptt
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND mt_pnet >= 0
              AND (:branch IS NULL OR branche = :branch)
              AND (:year_from IS NULL OR annee_echeance >= :year_from)
              AND (:year_to IS NULL OR annee_echeance <= :year_to)
            GROUP BY annee_echeance
        )
        SELECT
            annee_echeance,
            nb_quittances,
            total_pnet,
            total_ptt,
            LAG(nb_quittances) OVER (ORDER BY annee_echeance) AS prev_nb_quittances,
            LAG(total_pnet) OVER (ORDER BY annee_echeance) AS prev_total_pnet,
            ROUND(
                100.0
                * (nb_quittances - LAG(nb_quittances) OVER (ORDER BY annee_echeance))
                / NULLIF(LAG(nb_quittances) OVER (ORDER BY annee_echeance), 0),
                2
            ) AS croissance_quittances_pct,
            ROUND(
                100.0
                * (total_pnet - LAG(total_pnet) OVER (ORDER BY annee_echeance))
                / NULLIF(LAG(total_pnet) OVER (ORDER BY annee_echeance), 0),
                2
            ) AS croissance_pnet_pct
        FROM yearly
        ORDER BY annee_echeance
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "annee": _to_int(row["annee_echeance"]),
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_ptt": _to_float(row["total_ptt"]),
                "prev_nb_quittances": _to_int(row["prev_nb_quittances"]),
                "prev_total_pnet": _to_float(row["prev_total_pnet"]),
                "croissance_quittances_pct": _to_float(row["croissance_quittances_pct"]),
                "croissance_pnet_pct": _to_float(row["croissance_pnet_pct"]),
            }
            for row in rows
        ],
    }


@router.get("/production/branch-share")
def get_production_branch_share(
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        WITH totals AS (
            SELECT
                branche,
                COUNT(*) AS nb_quittances,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND mt_pnet >= 0
              AND (:year_from IS NULL OR annee_echeance >= :year_from)
              AND (:year_to IS NULL OR annee_echeance <= :year_to)
            GROUP BY branche
        )
        SELECT
            branche,
            nb_quittances,
            total_pnet,
            ROUND(100.0 * total_pnet / NULLIF(SUM(total_pnet) OVER (), 0), 2) AS part_pnet_pct
        FROM totals
        ORDER BY total_pnet DESC
        """
    )

    rows = db.execute(sql, {"year_from": year_from, "year_to": year_to}).mappings().all()

    return {
        "filters": {"year_from": year_from, "year_to": year_to},
        "items": [
            {
                "branche": row["branche"],
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "part_pnet_pct": _to_float(row["part_pnet_pct"]),
            }
            for row in rows
        ],
    }


@router.get("/portfolio/type-police")
def get_portfolio_type_police(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)

    sql = text(
        """
        SELECT
            COALESCE(type_police, 'N/A') AS type_police,
            COUNT(*) AS nb_polices
        FROM dim_police
        WHERE (:branch IS NULL OR branche = :branch)
        GROUP BY COALESCE(type_police, 'N/A')
        ORDER BY nb_polices DESC
        """
    )

    rows = db.execute(sql, {"branch": normalized_branch}).mappings().all()

    return {
        "filters": {"branch": normalized_branch},
        "items": [
            {
                "type_police": row["type_police"],
                "nb_polices": _to_int(row["nb_polices"]),
            }
            for row in rows
        ],
    }


@router.get("/portfolio/bonus-malus")
def get_portfolio_bonus_malus(
    branch: str | None = Query(default="AUTO", description="AUTO | IRDS | SANTE"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)

    sql = text(
        """
        WITH base AS (
            SELECT
                CASE
                    WHEN bonus_malus < 0.8 THEN 'Bonus fort (<0.8)'
                    WHEN bonus_malus < 1.0 THEN 'Bonus (0.8-0.99)'
                    WHEN bonus_malus = 1.0 THEN 'Neutre (1.0)'
                    WHEN bonus_malus <= 1.25 THEN 'Malus (1.01-1.25)'
                    ELSE 'Malus fort (>1.25)'
                END AS bucket,
                CASE
                    WHEN bonus_malus < 0.8 THEN 1
                    WHEN bonus_malus < 1.0 THEN 2
                    WHEN bonus_malus = 1.0 THEN 3
                    WHEN bonus_malus <= 1.25 THEN 4
                    ELSE 5
                END AS bucket_order,
                bonus_malus
            FROM dim_police
            WHERE bonus_malus IS NOT NULL
              AND (:branch IS NULL OR branche = :branch)
        )
        SELECT
            bucket,
            COUNT(*) AS nb_polices,
            COALESCE(AVG(bonus_malus), 0) AS avg_bonus_malus
        FROM base
        GROUP BY bucket, bucket_order
        ORDER BY bucket_order
        """
    )

    rows = db.execute(sql, {"branch": normalized_branch}).mappings().all()

    return {
        "filters": {"branch": normalized_branch},
        "items": [
            {
                "bucket": row["bucket"],
                "nb_polices": _to_int(row["nb_polices"]),
                "avg_bonus_malus": _to_float(row["avg_bonus_malus"]),
            }
            for row in rows
        ],
    }


@router.get("/agents/by-locality")
def get_agents_by_locality(
    limit: int = Query(default=20, ge=1, le=200),
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            COALESCE(a.localite_agent, 'N/A') AS localite_agent,
            COUNT(DISTINCT a.id_agent) AS nb_agents,
            COUNT(e.num_quittance) AS nb_quittances,
            COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(e.mt_commission), 0) AS total_commission
        FROM dim_agent a
        LEFT JOIN dwh_fact_emission e
          ON e.id_agent = a.id_agent
         AND e.etat_quit IN ('E','P','A')
         AND e.mt_pnet >= 0
         AND (:branch IS NULL OR e.branche = :branch)
         AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
         AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
        GROUP BY COALESCE(a.localite_agent, 'N/A')
        ORDER BY total_pnet DESC, nb_agents DESC
        LIMIT :limit
        """
    )

    rows = db.execute(
        sql,
        {
            "limit": limit,
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "limit": limit,
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "localite_agent": row["localite_agent"],
                "nb_agents": _to_int(row["nb_agents"]),
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_commission": _to_float(row["total_commission"]),
            }
            for row in rows
        ],
    }


@router.get("/impayes/monthly-trend")
def get_impayes_monthly_trend(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            annee_echeance,
            mois_echeance,
            branche,
            COUNT(*) AS nb_impayes,
            COALESCE(SUM(mt_acp), 0) AS total_mt_acp,
            COALESCE(SUM(mt_ptt), 0) AS total_mt_ptt
        FROM dwh_fact_impaye
        WHERE mois_echeance BETWEEN 1 AND 12
          AND (:branch IS NULL OR branche = :branch)
          AND (:year_from IS NULL OR annee_echeance >= :year_from)
          AND (:year_to IS NULL OR annee_echeance <= :year_to)
        GROUP BY annee_echeance, mois_echeance, branche
        ORDER BY annee_echeance, mois_echeance, branche
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "annee": _to_int(row["annee_echeance"]),
                "mois": _to_int(row["mois_echeance"]),
                "periode": f"{_to_int(row['annee_echeance']):04d}-{_to_int(row['mois_echeance']):02d}",
                "branche": row["branche"],
                "nb_impayes": _to_int(row["nb_impayes"]),
                "total_mt_acp": _to_float(row["total_mt_acp"]),
                "total_mt_ptt": _to_float(row["total_mt_ptt"]),
            }
            for row in rows
        ],
    }


@router.get("/impayes/rate-by-branch")
def get_impayes_rate_by_branch(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        WITH emission AS (
            SELECT
                branche,
                COALESCE(SUM(mt_pnet), 0) AS total_pnet
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P','A')
              AND mt_pnet >= 0
              AND (:year_from IS NULL OR annee_echeance >= :year_from)
              AND (:year_to IS NULL OR annee_echeance <= :year_to)
            GROUP BY branche
        ),
        impayes AS (
            SELECT
                branche,
                COUNT(*) AS nb_impayes,
                COALESCE(SUM(mt_acp), 0) AS total_mt_acp
            FROM dwh_fact_impaye
            WHERE (:year_from IS NULL OR annee_echeance >= :year_from)
              AND (:year_to IS NULL OR annee_echeance <= :year_to)
            GROUP BY branche
        )
        SELECT
            COALESCE(e.branche, i.branche) AS branche,
            COALESCE(i.nb_impayes, 0) AS nb_impayes,
            COALESCE(i.total_mt_acp, 0) AS total_mt_acp,
            COALESCE(e.total_pnet, 0) AS total_pnet,
            ROUND(
                100.0 * COALESCE(i.total_mt_acp, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0),
                2
            ) AS taux_impaye_sur_pnet_pct
        FROM emission e
        FULL OUTER JOIN impayes i ON i.branche = e.branche
        WHERE (:branch IS NULL OR COALESCE(e.branche, i.branche) = :branch)
        ORDER BY taux_impaye_sur_pnet_pct DESC NULLS LAST
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "branche": row["branche"],
                "nb_impayes": _to_int(row["nb_impayes"]),
                "total_mt_acp": _to_float(row["total_mt_acp"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "taux_impaye_sur_pnet_pct": _to_float(row["taux_impaye_sur_pnet_pct"]),
            }
            for row in rows
        ],
    }


@router.get("/commissions/top-agents")
def get_commissions_top_agents(
    limit: int = Query(default=10, ge=1, le=100),
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=None, ge=2019, le=2026),
    year_to: int | None = Query(default=None, ge=2019, le=2026),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            e.id_agent,
            COALESCE(a.code_agent, 'N/A') AS code_agent,
            COALESCE(a.nom_agent, 'N/A') AS nom_agent,
            COALESCE(a.groupe_agent, 'N/A') AS groupe_agent,
            COUNT(*) AS nb_quittances,
            COALESCE(SUM(e.mt_pnet), 0) AS total_pnet,
            COALESCE(SUM(e.mt_commission), 0) AS total_commission,
            ROUND(
                100.0 * COALESCE(SUM(e.mt_commission), 0) / NULLIF(COALESCE(SUM(e.mt_pnet), 0), 0),
                2
            ) AS taux_commission_pct
        FROM dwh_fact_emission e
        LEFT JOIN dim_agent a ON a.id_agent = e.id_agent
        WHERE e.etat_quit IN ('E','P','A')
          AND e.mt_pnet >= 0
          AND (:branch IS NULL OR e.branche = :branch)
          AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
          AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
        GROUP BY e.id_agent, a.code_agent, a.nom_agent, a.groupe_agent
        ORDER BY total_commission DESC
        LIMIT :limit
        """
    )

    rows = db.execute(
        sql,
        {
            "limit": limit,
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
    ).mappings().all()

    return {
        "filters": {
            "limit": limit,
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "items": [
            {
                "id_agent": _to_int(row["id_agent"]),
                "code_agent": row["code_agent"],
                "nom_agent": row["nom_agent"],
                "groupe_agent": row["groupe_agent"],
                "nb_quittances": _to_int(row["nb_quittances"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "total_commission": _to_float(row["total_commission"]),
                "taux_commission_pct": _to_float(row["taux_commission_pct"]),
            }
            for row in rows
        ],
    }


@router.get("/dashboard/ceo")
def get_ceo_dashboard_bundle(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    top_agents_limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    overview = get_overview(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    primes = get_primes_by_branch(year_from=year_from, year_to=year_to, db=db)
    monthly = get_production_monthly_trend(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    production_yoy = get_production_yoy(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    production_share = get_production_branch_share(
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    churn = get_churn_by_branch(branch=normalized_branch, db=db)
    portfolio = get_portfolio_situation(branch=normalized_branch, db=db)
    portfolio_type = get_portfolio_type_police(branch=normalized_branch, db=db)
    portfolio_bonus_malus = get_portfolio_bonus_malus(branch=normalized_branch, db=db)
    commissions = get_commissions_by_branch(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    commissions_top_agents = get_commissions_top_agents(
        limit=top_agents_limit,
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    agents = get_agents_by_group(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    agents_locality = get_agents_by_locality(
        limit=20,
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    impayes_monthly = get_impayes_monthly_trend(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    impayes_rate = get_impayes_rate_by_branch(
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )
    top_agents = get_top_agents(
        limit=top_agents_limit,
        branch=normalized_branch,
        year_from=year_from,
        year_to=year_to,
        db=db,
    )

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
        },
        "overview": overview,
        "primes_by_branch": primes["items"],
        "production_monthly_trend": monthly["items"],
        "production_yoy": production_yoy["items"],
        "production_branch_share": production_share["items"],
        "churn_by_branch": churn["items"],
        "portfolio_situation": portfolio["items"],
        "portfolio_type_police": portfolio_type["items"],
        "portfolio_bonus_malus": portfolio_bonus_malus["items"],
        "commissions_by_branch": commissions["items"],
        "commissions_top_agents": commissions_top_agents["items"],
        "agents_by_group": agents["items"],
        "agents_by_locality": agents_locality["items"],
        "impayes_monthly_trend": impayes_monthly["items"],
        "impayes_rate_by_branch": impayes_rate["items"],
        "top_agents": top_agents["items"],
    }

@router.get("/ml/churn-risk")
def get_ml_churn_risk_kpi(
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sql = text(
        """
        SELECT 
            COUNT(*) as total_polices,
            COALESCE(AVG(taux_sinistralite), 0) as avg_sinistralite,
            COALESCE(AVG(ratio_impaye), 0) as avg_impaye,
            COALESCE(AVG(taux_commission), 0) as avg_commission,
            SUM(target_churn) as total_resiliations,
            COALESCE(AVG(target_churn) * 100, 0) as taux_churn_pct
        FROM ml_features_churn
        """
    )
    row = db.execute(sql).mappings().first()

    return {
        "metrics": {
            "total_polices": _to_int(row["total_polices"]),
            "avg_sinistralite_pct": round(_to_float(row["avg_sinistralite"]) * 100, 2),
            "avg_impaye_pct": round(_to_float(row["avg_impaye"]) * 100, 2),
            "avg_commission_pct": round(_to_float(row["avg_commission"]) * 100, 2),
            "total_resiliations": _to_int(row["total_resiliations"]),
            "taux_churn_pct": round(_to_float(row["taux_churn_pct"]), 2),
        }
    }


@router.get("/ml/client-ltv")
def get_ml_client_ltv_kpi(
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    sql = text(
        """
        SELECT
            COUNT(*) as total_clients,
            COALESCE(AVG(ltv_estimee), 0) as avg_ltv,
            COALESCE(AVG(anciennete_client_annees), 0) as avg_anciennete,
            COALESCE(SUM(total_prime_facturee), 0) as total_ca,
            COALESCE(SUM(cout_total_sinistres), 0) as total_sinistres
        FROM ml_features_client
        """
    )
    row = db.execute(sql).mappings().first()

    return {
        "metrics": {
            "total_clients": _to_int(row["total_clients"]),
            "avg_ltv": round(_to_float(row["avg_ltv"]), 2),
            "avg_anciennete_annees": round(_to_float(row["avg_anciennete"]), 2),
            "total_ca": round(_to_float(row["total_ca"]), 2),
            "total_sinistres": round(_to_float(row["total_sinistres"]), 2),
        }
    }
