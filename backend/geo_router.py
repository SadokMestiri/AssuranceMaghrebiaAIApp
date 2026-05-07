from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from db import get_db

router = APIRouter(prefix="/geo", tags=["geo"])

VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}
YEAR_MIN = 2019
YEAR_MAX = 2026


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


def _validate_year_range(year_from: int | None, year_to: int | None) -> None:
    if year_from is not None and not YEAR_MIN <= year_from <= YEAR_MAX:
        raise HTTPException(status_code=400, detail=f"year_from must be between {YEAR_MIN} and {YEAR_MAX}")
    if year_to is not None and not YEAR_MIN <= year_to <= YEAR_MAX:
        raise HTTPException(status_code=400, detail=f"year_to must be between {YEAR_MIN} and {YEAR_MAX}")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise HTTPException(status_code=400, detail="year_from must be <= year_to")


def _to_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _to_int(value: Any) -> int:
    return int(value) if value is not None else 0


@router.get("/heatmap-polices")
def get_heatmap_polices(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    limit: int = Query(default=250, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        WITH emission AS (
            SELECT
                e.id_agent,
                COUNT(DISTINCT e.id_police) AS nb_polices,
                COALESCE(SUM(e.mt_pnet), 0) AS total_pnet
            FROM dwh_fact_emission e
            WHERE e.etat_quit IN ('E','P','A')
              AND e.mt_pnet >= 0
              AND (:branch IS NULL OR e.branche = :branch)
              AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
              AND (:year_to IS NULL OR e.annee_echeance <= :year_to)
            GROUP BY e.id_agent
        ),
        impaye AS (
            SELECT
                i.id_agent,
                COUNT(*) AS nb_impayes,
                COALESCE(SUM(i.mt_acp), 0) AS total_mt_acp
            FROM dwh_fact_impaye i
            WHERE (:branch IS NULL OR i.branche = :branch)
              AND (:year_from IS NULL OR i.annee_echeance >= :year_from)
              AND (:year_to IS NULL OR i.annee_echeance <= :year_to)
            GROUP BY i.id_agent
        )
        SELECT
            a.id_agent,
            COALESCE(a.localite_agent, 'N/A') AS gouvernorat,
            a.latitude_agent,
            a.longitude_agent,
            COALESCE(e.nb_polices, 0) AS nb_polices,
            COALESCE(e.total_pnet, 0) AS total_pnet,
            COALESCE(i.nb_impayes, 0) AS nb_impayes,
            COALESCE(i.total_mt_acp, 0) AS total_mt_acp,
            ROUND(100.0 * COALESCE(i.nb_impayes, 0) / NULLIF(COALESCE(e.nb_polices, 0), 0), 2) AS taux_impaye_pct
        FROM dim_agent a
        LEFT JOIN emission e ON e.id_agent = a.id_agent
        LEFT JOIN impaye i ON i.id_agent = a.id_agent
        WHERE a.latitude_agent IS NOT NULL
          AND a.longitude_agent IS NOT NULL
          AND COALESCE(e.nb_polices, 0) > 0
        ORDER BY COALESCE(e.nb_polices, 0) DESC
        LIMIT :limit
        """
    )

    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
            "limit": limit,
        },
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
            "limit": limit,
        },
        "signal_source": "impayes_proxy_for_geo_risk",
        "items": [
            {
                "id_agent": _to_int(row["id_agent"]),
                "gouvernorat": row["gouvernorat"],
                "latitude": _to_float(row["latitude_agent"]),
                "longitude": _to_float(row["longitude_agent"]),
                "nb_polices": _to_int(row["nb_polices"]),
                "total_pnet": _to_float(row["total_pnet"]),
                "nb_sinistres_proxy": _to_int(row["nb_impayes"]),
                "total_sinistres_proxy": _to_float(row["total_mt_acp"]),
                "taux_sinistres_proxy_pct": _to_float(row["taux_impaye_pct"]),
            }
            for row in rows
        ],
    }


@router.get("/sinistres/by-gouvernorat")
def get_sinistres_by_gouvernorat(
    branch: str | None = Query(default=None),
    year_from: int | None = Query(default=None, ge=2019, le=2025),
    year_to: int | None = Query(default=None, ge=2019, le=2025),
    gouvernorat: str | None = Query(default=None),   # ← ajouter
    db: Session = Depends(get_db),
) -> dict:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        SELECT
            TRIM(UPPER(c.ville))              AS gouvernorat,
            COUNT(*)                           AS nb_sinistres,
            COALESCE(SUM(s.mt_paye), 0)       AS total_mt_paye,
            COALESCE(SUM(s.mt_evaluation), 0) AS total_mt_evaluation
        FROM dwh_fact_sinistre s
        JOIN dim_client c ON c.id_client = s.id_client
        WHERE c.ville IS NOT NULL
          AND TRIM(c.ville) != ''
          AND (:branch IS NULL OR s.branche = :branch)
          AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
          AND (:year_to   IS NULL OR s.annee_survenance <= :year_to)
          AND (:gouvernorat IS NULL OR TRIM(UPPER(c.ville)) = UPPER(TRIM(:gouvernorat)))  -- ← ajouter
        GROUP BY TRIM(UPPER(c.ville))
        ORDER BY nb_sinistres DESC
        """
    )
    rows = db.execute(
        sql,
        {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
            "gouvernorat": gouvernorat,
        }
    ).mappings().all()

    return {
        "filters": {
            "branch": normalized_branch,
            "year_from": year_from,
            "year_to": year_to,
            "gouvernorat": gouvernorat,
        },
        "signal_source": "impayes_proxy_for_sinistres",
        "items": [
            {
                "gouvernorat": row["gouvernorat"],
                "nb_sinistres_proxy": _to_int(row.get("nb_sinistres_proxy", row.get("nb_sinistres"))),
                "total_sinistres_proxy": _to_float(row.get("total_sinistres_proxy", row.get("total_mt_paye"))),
                "taux_sinistres_proxy_sur_pnet_pct": _to_float(
                    row.get("taux_sinistres_proxy_sur_pnet_pct", row.get("taux_sinistres_sur_pnet_pct"))
                ),
                "nb_sinistres": _to_int(row.get("nb_sinistres", row.get("nb_sinistres_proxy"))),
                "total_mt_paye": _to_float(row.get("total_mt_paye", row.get("total_sinistres_proxy"))),
                "total_mt_evaluation": _to_float(row.get("total_mt_evaluation")),
            }
            for row in rows
        ]
    }


@router.get("/top-zones-risque")
def get_top_zones_risque(
    branch: str | None = Query(default=None, description="AUTO | IRDS | SANTE"),
    year_from: int | None = Query(default=2019, ge=2019, le=2026),
    year_to: int | None = Query(default=2026, ge=2019, le=2026),
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        WITH emission AS (
            SELECT
                TRIM(UPPER(c.ville))          AS gouvernorat,
                COALESCE(SUM(e.mt_pnet), 0)   AS total_pnet
            FROM dwh_fact_emission e
            JOIN dim_police p ON p.id_police = e.id_police        
            JOIN dim_client c ON c.id_client = p.id_client        
            WHERE e.etat_quit IN ('E','P','A')
              AND e.mt_pnet >= 0
              AND c.ville IS NOT NULL AND TRIM(c.ville) != ''
              AND (:branch IS NULL OR e.branche = :branch)
              AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
              AND (:year_to   IS NULL OR e.annee_echeance <= :year_to)
            GROUP BY TRIM(UPPER(c.ville))
        ),
        sinistres AS (
            SELECT
                TRIM(UPPER(c.ville))              AS gouvernorat,
                COUNT(*)                           AS nb_sinistres,
                COALESCE(SUM(s.mt_paye), 0)       AS total_mt_paye
            FROM dwh_fact_sinistre s
            JOIN dim_client c ON c.id_client = s.id_client
            WHERE c.ville IS NOT NULL AND TRIM(c.ville) != ''
              AND (:branch IS NULL OR s.branche = :branch)
              AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
              AND (:year_to   IS NULL OR s.annee_survenance <= :year_to)
            GROUP BY TRIM(UPPER(c.ville))
        ),
        merged AS (
            SELECT
                COALESCE(e.gouvernorat, s.gouvernorat) AS gouvernorat,
                COALESCE(e.total_pnet, 0)              AS total_pnet,
                COALESCE(s.nb_sinistres, 0)            AS nb_sinistres,
                COALESCE(s.total_mt_paye, 0)           AS total_mt_paye,
                ROUND(
                    100.0 * COALESCE(s.total_mt_paye, 0) / NULLIF(COALESCE(e.total_pnet, 0), 0),
                    2
                ) AS taux_sinistres_sur_pnet_pct
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
        LIMIT :limit
        """
    )

    rows = db.execute(
        sql,
        {"branch": normalized_branch, "year_from": year_from, "year_to": year_to, "limit": limit},
    ).mappings().all()

    return {
        "filters": {"branch": normalized_branch, "year_from": year_from, "year_to": year_to, "limit": limit},
        "signal_source": "impayes_proxy_for_geo_risk",
        "items": [
            {
                "rang": idx + 1,
                "gouvernorat": row["gouvernorat"],
                "total_pnet": _to_float(row["total_pnet"]),
                "nb_sinistres_proxy": _to_int(row.get("nb_sinistres_proxy", row.get("nb_sinistres"))),
                "total_sinistres_proxy": _to_float(row.get("total_sinistres_proxy", row.get("total_mt_paye"))),
                "taux_sinistres_proxy_sur_pnet_pct": _to_float(
                    row.get("taux_sinistres_proxy_sur_pnet_pct", row.get("taux_sinistres_sur_pnet_pct"))
                ),
                "nb_sinistres": _to_int(row.get("nb_sinistres", row.get("nb_sinistres_proxy"))),
                "total_mt_paye": _to_float(row.get("total_mt_paye", row.get("total_sinistres_proxy"))),
                "taux_sinistres_sur_pnet_pct": _to_float(
                    row.get("taux_sinistres_sur_pnet_pct", row.get("taux_sinistres_proxy_sur_pnet_pct"))
                ),
                "score_risque": _to_float(row["score_risque"]),
            }
            for idx, row in enumerate(rows)
        ],
    }

