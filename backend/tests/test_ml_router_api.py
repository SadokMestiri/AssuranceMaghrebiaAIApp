from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

import ml_router


def test_get_current_promotion_policy_returns_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ml_router,
        "get_promotion_policy",
        lambda: {
            "policy_env": "development",
            "promotion_metrics": ["avg_precision", "f1", "recall", "roc_auc"],
            "promotion_min_delta": 0.0,
            "env_specific_key": "PROMOTION_MIN_DELTA_DEVELOPMENT",
            "global_key": "PROMOTION_MIN_DELTA",
        },
    )

    response = ml_router.get_current_promotion_policy()

    assert response["status"] == "ready"
    assert response["promotion_policy"]["policy_env"] == "development"


def test_get_model_governance_passes_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ml_router,
        "get_model_governance_summary",
        lambda limit: {"history_count": limit, "history": []},
    )

    response = ml_router.get_model_governance(limit=7)

    assert response["status"] == "ready"
    assert response["governance"]["history_count"] == 7


def test_get_operations_readiness_returns_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ml_router,
        "get_impaye_operations_readiness",
        lambda months: {
            "tdsp": {"phase": "Phase 4 - Deploiement ML et Monitoring"},
            "window_months": months,
            "readiness": {"status": "green", "score": 100.0},
            "recommendations": [],
        },
    )

    response = ml_router.get_operations_readiness(months=9)

    assert response["status"] == "ready"
    assert response["operations_readiness"]["window_months"] == 9
    assert response["operations_readiness"]["readiness"]["status"] == "green"


def test_get_operations_readiness_maps_not_found_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_not_found(months: int = 6) -> dict[str, object]:
        raise FileNotFoundError("missing champion")

    monkeypatch.setattr(ml_router, "get_impaye_operations_readiness", _raise_not_found)

    with pytest.raises(HTTPException) as exc_info:
        ml_router.get_operations_readiness(months=6)

    assert exc_info.value.status_code == 404


def test_get_operations_readiness_metrics_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ml_router,
        "get_impaye_operations_readiness",
        lambda months: {
            "window_months": months,
            "readiness": {
                "status": "amber",
                "score": 82.5,
                "pass_count": 4,
                "warn_count": 1,
                "fail_count": 1,
            },
            "model": {
                "age_days": 12,
                "metrics": {
                    "avg_precision": 0.02,
                    "recall": 0.31,
                    "avg_precision_lift": 1.6,
                    "predicted_positive_rate": 0.22,
                },
            },
            "recent_business_signal": {
                "observed_impaye_rate": 0.18,
                "emissions_count": 2000,
                "impaye_count": 360,
            },
            "recommendations": ["review threshold"],
            "checks": [
                {"name": "model_freshness", "status": "pass"},
                {"name": "predicted_vs_observed_rate_gap", "status": "warn"},
            ],
        },
    )

    response = ml_router.get_operations_readiness_metrics(months=6)
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "maghrebia_ml_ops_readiness_up 1" in body
    assert "maghrebia_ml_ops_readiness_score 82.5" in body
    assert 'maghrebia_ml_ops_readiness_status_code{status="amber"} 1' in body
    assert "maghrebia_ml_ops_recommendations_count 1.0" in body
    assert "maghrebia_ml_ops_readiness_error 0" in body


def test_get_operations_readiness_metrics_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_error(months: int = 6) -> dict[str, object]:
        raise RuntimeError("readiness unavailable")

    monkeypatch.setattr(ml_router, "get_impaye_operations_readiness", _raise_error)

    response = ml_router.get_operations_readiness_metrics(months=6)
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "maghrebia_ml_ops_readiness_up 0" in body
    assert 'maghrebia_ml_ops_readiness_status_code{status="unavailable"} -1' in body
    assert "maghrebia_ml_ops_readiness_error 1" in body


def test_train_model_rejects_invalid_split_strategy() -> None:
    request = ml_router.TrainRequest(split_strategy="bad")

    with pytest.raises(HTTPException) as exc_info:
        ml_router.train_model(request)

    assert exc_info.value.status_code == 400
    assert "split_strategy" in str(exc_info.value.detail)


def test_train_model_returns_trained_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ml_router,
        "train_and_persist_model",
        lambda **_: {
            "run_id": "run_123",
            "promoted_to_champion": False,
            "promotion_reason": "promotion_disabled",
        },
    )

    request = ml_router.TrainRequest(split_strategy="temporal", promote_to_champion=False)
    response = ml_router.train_model(request)

    assert response["status"] == "trained"
    assert response["run_id"] == "run_123"
    assert response["promoted_to_champion"] is False


def test_get_model_info_maps_not_found_to_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_not_found() -> dict[str, object]:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(ml_router, "load_model_metadata", _raise_not_found)

    with pytest.raises(HTTPException) as exc_info:
        ml_router.get_model_info()

    assert exc_info.value.status_code == 404


def test_promote_latest_maps_file_not_found_to_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_not_found(force: bool = False, reason: str | None = None) -> dict[str, object]:
        raise FileNotFoundError("latest missing")

    monkeypatch.setattr(ml_router, "promote_latest_to_champion", _raise_not_found)

    with pytest.raises(HTTPException) as exc_info:
        ml_router.promote_latest(ml_router.PromoteRequest(force=False, reason="check"))

    assert exc_info.value.status_code == 404


def test_predict_impaye_normalizes_payload_and_returns_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_predict(features: dict[str, object], threshold: float | None = None) -> dict[str, object]:
        captured["features"] = features
        captured["threshold"] = threshold
        return {
            "probability_impaye": 0.42,
            "predicted_label": 1,
            "threshold": 0.3,
            "threshold_source": "request",
            "model_role": "champion",
            "selected_model": "rf_balanced",
        }

    monkeypatch.setattr(ml_router, "predict_impaye_probability", _fake_predict)

    request = ml_router.PredictRequest(
        branche="auto",
        periodicite="a",
        police_situation="v",
        annee_echeance=2025,
        mois_echeance=4,
        mt_pnet=1000,
        mt_rc=500,
        mt_fga=20,
        mt_timbre=5,
        mt_taxe=10,
        mt_ptt=0,
        mt_commission=120,
        bonus_malus=1,
        threshold=0.3,
    )

    response = ml_router.predict_impaye(request)

    assert response["status"] == "ok"
    assert response["input"]["branche"] == "AUTO"
    assert response["input"]["periodicite"] == "A"
    assert response["input"]["police_situation"] == "V"
    assert captured["threshold"] == 0.3


def test_predict_impaye_rejects_invalid_branche() -> None:
    request = ml_router.PredictRequest(
        branche="invalid",
        periodicite="A",
        police_situation="V",
        annee_echeance=2025,
        mois_echeance=4,
        mt_pnet=1000,
        mt_rc=500,
        mt_fga=20,
        mt_timbre=5,
        mt_taxe=10,
        mt_ptt=0,
        mt_commission=120,
        bonus_malus=1,
    )

    with pytest.raises(HTTPException) as exc_info:
        ml_router.predict_impaye(request)

    assert exc_info.value.status_code == 400
    assert "Invalid branche" in str(exc_info.value.detail)
