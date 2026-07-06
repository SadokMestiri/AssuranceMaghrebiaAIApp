from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from db import get_db

router = APIRouter(prefix="/geo", tags=["geo"])

VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}
YEAR_MIN = 2019
YEAR_MAX = 2026

# WGS-84 centroids for Tunisian cities/villes (one bubble per ville)
_VILLE_CENTROIDS: dict[str, tuple[float, float]] = {
    # Grand Tunis
    "TUNIS":                  (36.8190, 10.1658),
    "ARIANA":                 (36.8665, 10.1647),
    "LA MARSA":               (36.8879, 10.3233),
    "MARSA":                  (36.8879, 10.3233),
    "SIDI THABET":            (36.9214, 10.0289),
    "LA SOUKRA":              (36.8950, 10.2300),
    "RAOUED":                 (36.9017, 10.1958),
    "MNIHLA":                 (36.8833, 10.1333),
    "CARTHAGE":               (36.8525, 10.3239),
    "EL MENZAH":              (36.8597, 10.1772),
    "BARDO":                  (36.8100, 10.1347),
    "BEN AROUS":              (36.7533, 10.2282),
    "HAMMAM LIF":             (36.7286, 10.3347),
    "HAMMAM CHOTT":           (36.7200, 10.3167),
    "RADES":                  (36.7692, 10.2764),
    "MEGRINE":                (36.7567, 10.2400),
    "EZZAHRA":                (36.7567, 10.2700),
    "MOUROUJ":                (36.7000, 10.2167),
    "FOUCHANA":               (36.7000, 10.1700),
    "MORNAG":                 (36.6711, 10.2847),
    "BOU MHEL EL BASSATINE":  (36.7167, 10.2333),
    "MANOUBA":                (36.8100, 10.0986),
    "DEN DEN":                (36.8200, 10.1500),
    "TEBOURBA":               (36.8330,  9.8419),
    "OUED ELLIL":             (36.8097, 10.0200),
    "BORJ EL AMRI":           (36.6981, 10.0558),
    "DJEDEIDA":               (36.8364,  9.9108),
    # Nabeul
    "NABEUL":                 (36.4561, 10.7376),
    "HAMMAMET":               (36.4000, 10.6000),
    "KELIBIA":                (36.8472, 11.0946),
    "KORBA":                  (36.5735, 10.8698),
    "MENZEL TEMIME":          (36.7806, 10.9792),
    "DAR CHAABANE":           (36.5028, 10.7531),
    "EL HAOUARIA":            (37.0545, 11.0047),
    "GROMBALIA":              (36.5994, 10.5014),
    "BENI KHIAR":             (36.4766, 10.7284),
    "BENI KHALLED":           (36.4983, 10.5758),
    "SOLIMAN":                (36.7000, 10.4833),
    "MENZEL BOU ZELFA":       (36.6736, 10.6058),
    # Zaghouan
    "ZAGHOUAN":               (36.4020, 10.1420),
    "EL FAHS":                (36.3722, 10.0983),
    # Bizerte
    "BIZERTE":                (37.2744,  9.8739),
    "MENZEL BOURGUIBA":       (37.1614,  9.7978),
    "MATEUR":                 (37.0431,  9.6639),
    "SEJNANE":                (37.0555,  8.9828),
    "ZARZOUNA":               (37.2950,  9.8728),
    "MENZEL JEMIL":           (37.2236, 10.1117),
    "RAS JEBEL":              (37.2153, 10.1250),
    "EL ALIA":                (37.1834, 10.0628),
    # Beja
    "BEJA":                   (36.7258,  9.1817),
    "MEDJEZ EL BAB":          (36.6494,  9.6119),
    "TESTOUR":                (36.5530,  9.4441),
    "NEFZA":                  (37.0278,  9.0156),
    "AMDOUN":                 (36.7500,  8.7500),
    # Jendouba
    "JENDOUBA":               (36.5011,  8.7803),
    "TABARKA":                (36.9547,  8.7567),
    "AIN DRAHAM":             (36.7758,  8.6925),
    "GHARDIMAOU":             (36.4500,  8.4372),
    "FERNANA":                (36.6267,  8.7244),
    "BOUSALEM":               (36.6336,  8.9728),
    # Kef
    "KEF":                    (36.1826,  8.7149),
    "DAHMANI":                (35.9490,  8.8386),
    "TAJEROUINE":             (35.8930,  8.5481),
    "NEBEUR":                 (36.4577,  8.8013),
    "KALAAT SENAN":           (36.3167,  8.6667),
    # Siliana
    "SILIANA":                (36.0849,  9.3708),
    "BOU ARADA":              (36.3595,  9.6031),
    "GAAFOUR":                (36.3202,  9.3234),
    "MAKTHAR":                (35.8571,  9.2044),
    # Sousse
    "SOUSSE":                 (35.8256, 10.6369),
    "HAMMAM SOUSSE":          (35.8617, 10.5942),
    "AKOUDA":                 (35.8702, 10.5642),
    "KALAA SEGHIRA":          (35.8531, 10.6136),
    "KALAA KEBIRA":           (35.9190, 10.5303),
    "MSAKEN":                 (35.7326, 10.5786),
    "M'SAKEN":                (35.7326, 10.5786),
    "CHOTT MERIEM":           (35.9181, 10.5503),
    "ENFIDHA":                (36.1381, 10.3789),
    "KONDAR":                 (35.7000, 10.4000),
    # Monastir
    "MONASTIR":               (35.7643, 10.8113),
    "MOKNINE":                (35.6406, 10.9053),
    "KSAR HELLAL":            (35.6486, 10.8892),
    "TEBOULBA":               (35.6773, 10.9781),
    "KSIBET EL MEDIOUNI":     (35.7145, 10.8617),
    "SAYADA":                 (35.6573, 10.7381),
    "BEKALTA":                (35.6037, 11.0153),
    "LAMTA":                  (35.7383, 10.9208),
    "SAHLINE":                (35.7636, 10.7561),
    # Mahdia
    "MAHDIA":                 (35.5047, 11.0622),
    "EL JEM":                 (35.2956, 10.7101),
    "KSOUR ESSEF":            (35.0699, 11.0581),
    "CHEBBA":                 (35.2376, 11.1135),
    "SIDI ALOUANE":           (35.3833, 10.9333),
    # Sfax
    "SFAX":                   (34.7398, 10.7600),
    "SAKIET EZZIT":           (34.7667, 10.8333),
    "SAKIET EDDAIER":         (34.7833, 10.7500),
    "AGAREB":                 (34.7564, 10.5333),
    "MAHRES":                 (34.5328, 10.4998),
    "MAHRÈS":                 (34.5328, 10.4998),
    "JEBENIANA":              (34.9667, 10.5333),
    "EL HENCHA":              (34.6167, 10.3333),
    "MENZEL CHAKER":          (34.9833, 10.4333),
    # Kairouan
    "KAIROUAN":               (35.6781, 10.0963),
    "SBIKHA":                 (35.9161, 10.0267),
    "HAFFOUZ":                (35.6322,  9.6714),
    "BOUHAJLA":               (35.7500,  9.8667),
    "NASRALLAH":              (35.5333,  9.9167),
    "OUESLATIA":              (35.9667,  9.4167),
    # Kasserine
    "KASSERINE":              (35.1721,  8.8307),
    "SBEITLA":                (35.2318,  9.1252),
    "THALA":                  (35.5747,  8.6697),
    "FERIANA":                (34.9483,  8.5583),
    "FOUSSANA":               (35.2050,  8.7972),
    "HIDRA":                  (35.4167,  8.5167),
    # Sidi Bouzid
    "SIDI BOUZID":            (35.0382,  9.4849),
    "JILMA":                  (35.2819,  9.5736),
    "REGUEB":                 (34.8333,  9.7667),
    "MEZZOUNA":               (34.6000,  9.9833),
    "BIR EL HAFEY":           (34.7167,  9.1500),
    # Gabes
    "GABES":                  (33.8814, 10.0982),
    "EL HAMMA":               (33.8894,  9.7958),
    "MARETH":                 (33.6383, 10.1994),
    "METOUIA":                (33.9673, 10.0046),
    "MATMATA":                (33.5438,  9.9714),
    "NOUVELLE MATMATA":       (33.5167,  9.9833),
    # Medenine
    "MEDENINE":               (33.3540, 10.5053),
    "ZARZIS":                 (33.5021, 11.1115),
    "BEN GARDANE":            (33.1384, 11.2192),
    "HOUMT SOUK":             (33.8757, 10.8550),
    "DJERBA":                 (33.8178, 10.9200),
    "MIDOUN":                 (33.8178, 10.9961),
    "AJIM":                   (33.7263, 10.7450),
    "BENI KHEDACHE":          (33.0833, 10.0167),
    # Tataouine
    "TATAOUINE":              (32.9211, 10.4518),
    "REMADA":                 (32.3167, 10.4000),
    "BIR LAHMAR":             (32.9311, 10.1236),
    "GHOMRASSEN":             (33.0500, 10.4500),
    "DEHIBA":                 (31.9989, 10.8069),
    # Gafsa
    "GAFSA":                  (34.4251,  8.7842),
    "OM EL ARAIES":           (34.0667,  9.0000),
    "REDEYEF":                (34.3825,  8.1547),
    "METLAOUI":               (34.3281,  8.4039),
    "EL KSAR":                (34.4500,  8.6500),
    "MOULARES":               (34.4667,  8.2833),
    "SNED":                   (34.7333,  8.5667),
    # Tozeur
    "TOZEUR":                 (33.9197,  8.1335),
    "NEFTA":                  (33.8728,  7.8767),
    "DEGACHE":                (33.9761,  8.2197),
    "TAMEGHZA":               (34.3806,  7.9736),
    "HAZOUA":                 (33.8983,  8.3539),
    # Kebili
    "KEBILI":                 (33.7054,  8.9700),
    "SOUK LAHAD":             (33.7583,  9.1686),
    "DOUZ":                   (33.4551,  9.0155),
    "FAOUAR":                 (33.5706,  9.2600),
    "EL GOLAA":               (33.9667,  8.8167),
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
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized_branch = _normalize_branch(branch)
    _validate_year_range(year_from, year_to)

    sql = text(
        """
        WITH polices AS (
            SELECT
                TRIM(UPPER(c.ville))            AS gouvernorat,
                COUNT(DISTINCT e.id_police)      AS nb_polices,
                COALESCE(SUM(e.mt_pnet), 0)      AS total_pnet
            FROM dwh_fact_emission e
            JOIN dim_police p  ON p.id_police  = e.id_police
            JOIN dim_client c  ON c.id_client  = p.id_client
            WHERE e.etat_quit IN ('E','P','A')
              AND e.mt_pnet >= 0
              AND c.ville IS NOT NULL
              AND TRIM(c.ville) != ''
              AND (:branch IS NULL OR e.branche = :branch)
              AND (:year_from IS NULL OR e.annee_echeance >= :year_from)
              AND (:year_to   IS NULL OR e.annee_echeance <= :year_to)
            GROUP BY TRIM(UPPER(c.ville))
        ),
        sinistres AS (
            SELECT
                TRIM(UPPER(c.ville))              AS gouvernorat,
                COUNT(*)                           AS nb_sinistres,
                COALESCE(SUM(s.mt_paye), 0)        AS total_mt_paye
            FROM dwh_fact_sinistre s
            JOIN dim_client c ON c.id_client = s.id_client
            WHERE c.ville IS NOT NULL
              AND TRIM(c.ville) != ''
              AND (:branch IS NULL OR s.branche = :branch)
              AND (:year_from IS NULL OR s.annee_survenance >= :year_from)
              AND (:year_to   IS NULL OR s.annee_survenance <= :year_to)
            GROUP BY TRIM(UPPER(c.ville))
        )
        SELECT
            p.gouvernorat,
            p.nb_polices,
            p.total_pnet,
            COALESCE(s.nb_sinistres, 0)  AS nb_sinistres,
            COALESCE(s.total_mt_paye, 0) AS total_mt_paye,
            ROUND(
                100.0 * COALESCE(s.total_mt_paye, 0) / NULLIF(p.total_pnet, 0),
                2
            ) AS taux_sinistres_pct,
            ROUND(
                0.70 * ROUND(100.0 * COALESCE(s.total_mt_paye, 0) / NULLIF(p.total_pnet, 0), 2)
                + 30.0 * COALESCE(s.nb_sinistres, 0)
                        / NULLIF(MAX(COALESCE(s.nb_sinistres, 0)) OVER (), 0),
                2
            ) AS score_risque
        FROM polices p
        LEFT JOIN sinistres s ON s.gouvernorat = p.gouvernorat
        WHERE p.nb_polices > 0
        ORDER BY p.nb_polices DESC
        """
    )

    rows = db.execute(
        sql,
        {"branch": normalized_branch, "year_from": year_from, "year_to": year_to},
    ).mappings().all()

    items = []
    for row in rows:
        ville = (row["gouvernorat"] or "").strip().upper()
        coords = _VILLE_CENTROIDS.get(ville)
        if coords is None:
            continue
        lat, lon = coords
        pnet = _to_float(row["total_pnet"])
        mt_paye = _to_float(row["total_mt_paye"])
        taux = round(100.0 * mt_paye / pnet, 2) if pnet > 0 else 0.0
        items.append({
            "ville":              ville,
            "latitude":           lat,
            "longitude":          lon,
            "nb_polices":         _to_int(row["nb_polices"]),
            "total_pnet":         round(pnet, 2),
            "nb_sinistres":       _to_int(row["nb_sinistres"]),
            "total_mt_paye":      round(mt_paye, 2),
            "taux_sinistres_pct": taux,
            "score_risque":       _to_float(row["score_risque"]),
        })

    return {
        "filters": {"branch": normalized_branch, "year_from": year_from, "year_to": year_to},
        "items": items,
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

