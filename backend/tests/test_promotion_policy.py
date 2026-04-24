from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[reportMissingImports]

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_pipeline import _evaluate_promotion, get_promotion_policy


@pytest.fixture
def clean_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "PROMOTION_POLICY_ENV",
        "APP_ENV",
        "ENVIRONMENT",
        "PROMOTION_MIN_DELTA",
        "PROMOTION_MIN_DELTA_DEVELOPMENT",
        "PROMOTION_MIN_DELTA_STAGING",
        "PROMOTION_MIN_DELTA_PRODUCTION",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_policy_defaults_to_development(clean_policy_env: None) -> None:
    policy = get_promotion_policy()

    assert policy["policy_env"] == "development"
    assert policy["promotion_min_delta"] == 0.0


def test_policy_alias_stage_maps_to_staging(clean_policy_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMOTION_POLICY_ENV", "stage")

    policy = get_promotion_policy()

    assert policy["policy_env"] == "staging"
    assert policy["promotion_min_delta"] == 0.0005


def test_policy_uses_env_specific_before_global(
    clean_policy_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROMOTION_POLICY_ENV", "staging")
    monkeypatch.setenv("PROMOTION_MIN_DELTA_STAGING", "0.002")
    monkeypatch.setenv("PROMOTION_MIN_DELTA", "0.01")

    policy = get_promotion_policy()

    assert policy["policy_env"] == "staging"
    assert policy["promotion_min_delta"] == 0.002


def test_policy_uses_global_when_env_specific_missing(
    clean_policy_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PROMOTION_MIN_DELTA", "0.004")

    policy = get_promotion_policy()

    assert policy["policy_env"] == "production"
    assert policy["promotion_min_delta"] == 0.004


def test_policy_invalid_env_or_delta_falls_back_defaults(
    clean_policy_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "qa")
    monkeypatch.setenv("PROMOTION_MIN_DELTA_DEVELOPMENT", "bad-value")

    policy = get_promotion_policy()

    assert policy["policy_env"] == "development"
    assert policy["promotion_min_delta"] == 0.0


def test_evaluate_promotion_applies_policy_min_delta() -> None:
    challenger_metrics = {
        "avg_precision": 0.0200,
        "f1": 0.10,
        "recall": 0.20,
        "roc_auc": 0.30,
    }
    champion_metrics = {
        "avg_precision": 0.0195,
        "f1": 0.10,
        "recall": 0.20,
        "roc_auc": 0.30,
    }

    decision, reason, comparison = _evaluate_promotion(
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        champion_exists=True,
        promote_to_champion=True,
        promotion_policy={"policy_env": "staging", "promotion_min_delta": 0.001},
    )

    assert decision is False
    assert reason == "below_min_delta"
    assert comparison["promotion_policy_env"] == "staging"
    assert comparison["primary_metric_delta"] == pytest.approx(0.0005)


def test_evaluate_promotion_promotes_when_threshold_is_met() -> None:
    challenger_metrics = {
        "avg_precision": 0.0200,
        "f1": 0.11,
        "recall": 0.21,
        "roc_auc": 0.31,
    }
    champion_metrics = {
        "avg_precision": 0.0195,
        "f1": 0.10,
        "recall": 0.20,
        "roc_auc": 0.30,
    }

    decision, reason, comparison = _evaluate_promotion(
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        champion_exists=True,
        promote_to_champion=True,
        promotion_policy={"policy_env": "staging", "promotion_min_delta": 0.0004},
    )

    assert decision is True
    assert reason == "better_metrics"
    assert comparison["promotion_policy_env"] == "staging"
