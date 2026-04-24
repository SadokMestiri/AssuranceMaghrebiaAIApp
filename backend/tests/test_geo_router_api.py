from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

import geo_router


class _FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeResult":
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _FakeDB:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.params: dict[str, object] | None = None

    def execute(self, _sql: object, params: dict[str, object]) -> _FakeResult:
        self.params = params
        return _FakeResult(self.rows)


def test_get_heatmap_polices_returns_items() -> None:
    db = _FakeDB(
        [
            {
                "id_agent": 10,
                "gouvernorat": "Tunis",
                "latitude_agent": 36.8,
                "longitude_agent": 10.2,
                "nb_polices": 120,
                "total_pnet": 400000.0,
                "nb_impayes": 12,
                "total_mt_acp": 30000.0,
                "taux_impaye_pct": 10.0,
            }
        ]
    )

    payload = geo_router.get_heatmap_polices(
        branch="auto",
        year_from=2020,
        year_to=2024,
        limit=25,
        db=db,
    )

    assert payload["filters"]["branch"] == "AUTO"
    assert payload["signal_source"] == "impayes_proxy_for_geo_risk"
    assert payload["items"][0]["gouvernorat"] == "Tunis"
    assert payload["items"][0]["nb_polices"] == 120


def test_get_sinistres_by_gouvernorat_returns_items() -> None:
    db = _FakeDB(
        [
            {
                "gouvernorat": "Sousse",
                "nb_polices": 80,
                "total_pnet": 250000.0,
                "nb_sinistres_proxy": 9,
                "total_sinistres_proxy": 18000.0,
                "taux_sinistres_proxy_sur_pnet_pct": 7.2,
            }
        ]
    )

    payload = geo_router.get_sinistres_by_gouvernorat(
        branch="IRDS",
        year_from=2021,
        year_to=2024,
        db=db,
    )

    assert payload["filters"]["branch"] == "IRDS"
    assert payload["signal_source"] == "impayes_proxy_for_sinistres"
    assert payload["items"][0]["gouvernorat"] == "Sousse"
    assert payload["items"][0]["nb_sinistres_proxy"] == 9


def test_get_top_zones_risque_assigns_ranking() -> None:
    db = _FakeDB(
        [
            {
                "gouvernorat": "Ariana",
                "total_pnet": 130000.0,
                "nb_sinistres_proxy": 11,
                "total_sinistres_proxy": 25000.0,
                "taux_sinistres_proxy_sur_pnet_pct": 19.2,
                "score_risque": 42.5,
            },
            {
                "gouvernorat": "Monastir",
                "total_pnet": 98000.0,
                "nb_sinistres_proxy": 7,
                "total_sinistres_proxy": 12000.0,
                "taux_sinistres_proxy_sur_pnet_pct": 12.3,
                "score_risque": 28.7,
            },
        ]
    )

    payload = geo_router.get_top_zones_risque(
        branch="SANTE",
        year_from=2022,
        year_to=2024,
        limit=2,
        db=db,
    )

    assert payload["filters"]["branch"] == "SANTE"
    assert payload["items"][0]["rang"] == 1
    assert payload["items"][1]["rang"] == 2


def test_geo_endpoints_reject_invalid_branch() -> None:
    db = _FakeDB([])

    with pytest.raises(HTTPException) as heatmap_error:
        geo_router.get_heatmap_polices(branch="bad", db=db)

    assert heatmap_error.value.status_code == 400

    with pytest.raises(HTTPException) as sinistres_error:
        geo_router.get_sinistres_by_gouvernorat(branch="bad", db=db)

    assert sinistres_error.value.status_code == 400

    with pytest.raises(HTTPException) as zones_error:
        geo_router.get_top_zones_risque(branch="bad", db=db)

    assert zones_error.value.status_code == 400
