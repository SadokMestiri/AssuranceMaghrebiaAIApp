from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from ml_pipeline import (
    get_impaye_operations_readiness,
    get_promotion_policy,
    get_model_governance_summary,
    load_latest_model_metadata,
    load_model_metadata,
    promote_latest_to_champion,
    predict_impaye_probability,
    train_and_persist_model,
)
from ml_services.churn_service        import get_churn_summary, predict_churn
from ml_services.fraud_service        import get_fraud_summary
from ml_services.segmentation_service import get_segmentation_summary
from ml_services.risk_service         import score_risk, get_risk_table
from ml_services.claim_service        import predict_claim_severity
from ml_services.prophet_service      import get_forecast, get_all_forecasts
from ml_services.anomaly_service      import detect_anomalies
from ml_services.drift_service        import detect_drift


router = APIRouter(prefix="/ml", tags=["ml"])

VALID_BRANCHES          = {"AUTO", "IRDS", "SANTE"}
VALID_PERIODICITES      = {"A", "S", "T", "C"}
VALID_POLICE_SITUATIONS = {"V", "R", "T", "S", "A"}
YEAR_MIN = 2019
YEAR_MAX = 2026


# ── Pydantic schemas ───────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    year_from:           int | None = Field(default=None, ge=YEAR_MIN, le=YEAR_MAX)
    year_to:             int | None = Field(default=None, ge=YEAR_MIN, le=YEAR_MAX)
    test_size:           float      = Field(default=0.2, gt=0.05, lt=0.5)
    random_state:        int        = 42
    split_strategy:      str        = "temporal"
    promote_to_champion: bool       = True


class PredictRequest(BaseModel):
    branche:          str
    periodicite:      str
    police_situation: str
    annee_echeance:   int   = Field(ge=YEAR_MIN, le=YEAR_MAX)
    mois_echeance:    int   = Field(ge=1, le=12)
    mt_pnet:          float = Field(ge=0)
    mt_rc:            float = Field(ge=0)
    mt_fga:           float | None = Field(default=None, ge=0)
    mt_timbre:        float | None = Field(default=None, ge=0)
    mt_taxe:          float | None = Field(default=None, ge=0)
    mt_ptt:           float | None = Field(default=None, ge=0)
    mt_commission:    float = Field(ge=0)
    bonus_malus:      float = Field(ge=0)
    threshold:        float | None = Field(default=None, ge=0, le=1)


class PromoteRequest(BaseModel):
    force:  bool        = False
    reason: str | None  = None


class ChurnPredictRequest(BaseModel):
    branche:       str   = "AUTO"
    bonus_malus:   float = Field(default=1.0, ge=0)
    nb_quittances: int   = Field(default=4,   ge=0)
    mt_pnet:       float = Field(default=1200, ge=0)
    taux_impaye:   float = Field(default=0.0, ge=0, le=1)
    nb_sinistres:  int   = Field(default=0,   ge=0)


class RiskScoreRequest(BaseModel):
    branche:           str   = "AUTO"
    bonus_malus:       float = Field(default=1.0, ge=0)
    puissance:         int   = Field(default=6,   ge=1)
    age_vehicule:      int   = Field(default=5,   ge=0)
    age_client:        int   = Field(default=40,  ge=18)
    nb_sinistres_hist: int   = Field(default=0,   ge=0)
    mt_pnet:           float = Field(default=1200, ge=0)


class ClaimPredictRequest(BaseModel):
    branche:          str   = "AUTO"
    nature_sinistre:  str   = "MATERIEL"
    mt_evaluation:    float = Field(default=5000, ge=0)
    age_client:       int   = Field(default=40,   ge=18)
    bonus_malus:      float = Field(default=1.0,  ge=0)


# ── Prometheus helpers ─────────────────────────────────────────────────────

PROMETHEUS_STATUS_CODE = {"green": 2, "amber": 1, "red": 0, "unavailable": -1}


def _metric_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(default) if value is None else float(value)
    except (TypeError, ValueError):
        return float(default)


def _escape_prometheus_label(value: Any) -> str:
    raw = str(value)
    return raw.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _safe_metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _render_operations_readiness_prometheus(
    readiness: dict[str, Any] | None,
    months: int,
    up: int,
) -> str:
    readiness_payload    = readiness or {}
    readiness_block      = readiness_payload.get("readiness", {})
    model_block          = readiness_payload.get("model", {})
    model_metrics        = model_block.get("metrics", {})
    recent_signal        = readiness_payload.get("recent_business_signal", {})
    checks               = readiness_payload.get("checks", [])

    status      = str(readiness_block.get("status", "unavailable")).lower()
    status_code = PROMETHEUS_STATUS_CODE.get(status, -1)

    score                  = _metric_float(readiness_block.get("score"), 0.0)
    fail_count             = _metric_float(readiness_block.get("fail_count"), 0)
    warn_count             = _metric_float(readiness_block.get("warn_count"), 0)
    pass_count             = _metric_float(readiness_block.get("pass_count"), 0)
    recommendations_count  = _metric_float(len(readiness_payload.get("recommendations", [])), 0)
    model_age_days         = _metric_float(model_block.get("age_days"), 0.0)
    avg_precision          = _metric_float(model_metrics.get("avg_precision"), 0.0)
    recall                 = _metric_float(model_metrics.get("recall"), 0.0)
    avg_precision_lift     = _metric_float(model_metrics.get("avg_precision_lift"), 0.0)
    predicted_positive_rate= _metric_float(model_metrics.get("predicted_positive_rate"), 0.0)
    observed_impaye_rate   = _metric_float(recent_signal.get("observed_impaye_rate"), 0.0)
    emissions_count        = _metric_float(recent_signal.get("emissions_count"), 0)
    impaye_count           = _metric_float(recent_signal.get("impaye_count"), 0)
    rate_gap               = abs(predicted_positive_rate - observed_impaye_rate)

    lines: list[str] = [
        "# HELP maghrebia_ml_ops_readiness_up 1 when operations readiness is computed successfully.",
        "# TYPE maghrebia_ml_ops_readiness_up gauge",
        f"maghrebia_ml_ops_readiness_up {up}",
        "# HELP maghrebia_ml_ops_readiness_score Operational readiness score (0-100).",
        "# TYPE maghrebia_ml_ops_readiness_score gauge",
        f"maghrebia_ml_ops_readiness_score {score}",
        "# HELP maghrebia_ml_ops_readiness_status_code Readiness status code (green=2, amber=1, red=0, unavailable=-1).",
        "# TYPE maghrebia_ml_ops_readiness_status_code gauge",
        f'maghrebia_ml_ops_readiness_status_code{{status="{_escape_prometheus_label(status)}"}} {status_code}',
        "# HELP maghrebia_ml_ops_readiness_fail_count Number of failing readiness checks.",
        "# TYPE maghrebia_ml_ops_readiness_fail_count gauge",
        f"maghrebia_ml_ops_readiness_fail_count {fail_count}",
        "# HELP maghrebia_ml_ops_readiness_warn_count Number of warning readiness checks.",
        "# TYPE maghrebia_ml_ops_readiness_warn_count gauge",
        f"maghrebia_ml_ops_readiness_warn_count {warn_count}",
        "# HELP maghrebia_ml_ops_readiness_pass_count Number of passing readiness checks.",
        "# TYPE maghrebia_ml_ops_readiness_pass_count gauge",
        f"maghrebia_ml_ops_readiness_pass_count {pass_count}",
        "# HELP maghrebia_ml_ops_recommendations_count Number of operational recommendations.",
        "# TYPE maghrebia_ml_ops_recommendations_count gauge",
        f"maghrebia_ml_ops_recommendations_count {recommendations_count}",
        "# HELP maghrebia_ml_ops_window_months Observation window in months.",
        "# TYPE maghrebia_ml_ops_window_months gauge",
        f"maghrebia_ml_ops_window_months {months}",
        "# HELP maghrebia_ml_ops_model_age_days Champion model age in days.",
        "# TYPE maghrebia_ml_ops_model_age_days gauge",
        f"maghrebia_ml_ops_model_age_days {model_age_days}",
        "# HELP maghrebia_ml_ops_avg_precision Champion avg_precision metric.",
        "# TYPE maghrebia_ml_ops_avg_precision gauge",
        f"maghrebia_ml_ops_avg_precision {avg_precision}",
        "# HELP maghrebia_ml_ops_recall Champion recall metric.",
        "# TYPE maghrebia_ml_ops_recall gauge",
        f"maghrebia_ml_ops_recall {recall}",
        "# HELP maghrebia_ml_ops_avg_precision_lift Champion avg_precision lift versus prevalence.",
        "# TYPE maghrebia_ml_ops_avg_precision_lift gauge",
        f"maghrebia_ml_ops_avg_precision_lift {avg_precision_lift}",
        "# HELP maghrebia_ml_ops_predicted_positive_rate Predicted positive rate from champion metrics.",
        "# TYPE maghrebia_ml_ops_predicted_positive_rate gauge",
        f"maghrebia_ml_ops_predicted_positive_rate {predicted_positive_rate}",
        "# HELP maghrebia_ml_ops_observed_impaye_rate Observed impaye rate over recent window.",
        "# TYPE maghrebia_ml_ops_observed_impaye_rate gauge",
        f"maghrebia_ml_ops_observed_impaye_rate {observed_impaye_rate}",
        "# HELP maghrebia_ml_ops_rate_gap Absolute gap between predicted positive rate and observed impaye rate.",
        "# TYPE maghrebia_ml_ops_rate_gap gauge",
        f"maghrebia_ml_ops_rate_gap {rate_gap}",
        "# HELP maghrebia_ml_ops_recent_emissions_count Number of recent emissions used for readiness signal.",
        "# TYPE maghrebia_ml_ops_recent_emissions_count gauge",
        f"maghrebia_ml_ops_recent_emissions_count {emissions_count}",
        "# HELP maghrebia_ml_ops_recent_impaye_count Number of recent impaye records used for readiness signal.",
        "# TYPE maghrebia_ml_ops_recent_impaye_count gauge",
        f"maghrebia_ml_ops_recent_impaye_count {impaye_count}",
        "# HELP maghrebia_ml_ops_tdsp_phase_constant TDSP phase constant metric for monitoring alignment.",
        "# TYPE maghrebia_ml_ops_tdsp_phase_constant gauge",
        'maghrebia_ml_ops_tdsp_phase_constant{phase="phase4_deployment_monitoring"} 1',
    ]

    check_status_map = {"pass": 2, "warn": 1, "fail": 0}
    lines.append("# HELP maghrebia_ml_ops_check_status_code Per-check status code (pass=2, warn=1, fail=0).")
    lines.append("# TYPE maghrebia_ml_ops_check_status_code gauge")
    for check in checks:
        check_name   = _safe_metric_name(str(check.get("name", "unknown_check"))).lower()
        check_status = str(check.get("status", "fail")).lower()
        check_code   = check_status_map.get(check_status, 0)
        lines.append(
            "maghrebia_ml_ops_check_status_code"
            f'{{check="{_escape_prometheus_label(check_name)}",'
            f'status="{_escape_prometheus_label(check_status)}"}} {check_code}'
        )

    lines.append("# HELP maghrebia_ml_ops_readiness_error 1 when readiness computation failed.")
    lines.append("# TYPE maghrebia_ml_ops_readiness_error gauge")
    lines.append(f"maghrebia_ml_ops_readiness_error {0 if up else 1}")
    lines.append("")
    return "\n".join(lines)


def _normalize_prediction_payload(payload: PredictRequest) -> dict[str, Any]:
    branche          = payload.branche.strip().upper()
    periodicite      = payload.periodicite.strip().upper()
    police_situation = payload.police_situation.strip().upper()

    if branche not in VALID_BRANCHES:
        raise HTTPException(status_code=400, detail=f"Invalid branche '{payload.branche}'")
    if periodicite not in VALID_PERIODICITES:
        raise HTTPException(status_code=400, detail=f"Invalid periodicite '{payload.periodicite}'")
    if police_situation not in VALID_POLICE_SITUATIONS:
        raise HTTPException(status_code=400, detail=f"Invalid police_situation '{payload.police_situation}'")

    return {
        "branche":          branche,
        "periodicite":      periodicite,
        "police_situation": police_situation,
        "annee_echeance":   payload.annee_echeance,
        "mois_echeance":    payload.mois_echeance,
        "mt_pnet":          payload.mt_pnet,
        "mt_rc":            payload.mt_rc,
        "mt_fga":           payload.mt_fga,
        "mt_timbre":        payload.mt_timbre,
        "mt_taxe":          payload.mt_taxe,
        "mt_ptt":           payload.mt_ptt,
        "mt_commission":    payload.mt_commission,
        "bonus_malus":      payload.bonus_malus,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING IMPAYÉ MODEL ENDPOINTS (ml_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/model-info")
def get_model_info() -> dict[str, Any]:
    try:
        metadata = load_model_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model found. Train one first via POST /api/v1/ml/train.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read model metadata: {exc}")
    return {"status": "ready", "model": metadata}


@router.get("/latest-model-info")
def get_latest_model_info() -> dict[str, Any]:
    try:
        metadata = load_latest_model_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No latest challenger model found. Train one first via POST /api/v1/ml/train.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read latest model metadata: {exc}")
    return {"status": "ready", "model": metadata}


@router.get("/governance")
def get_model_governance(
    limit: int = Query(default=20, ge=1, le=200, description="Number of latest governance events"),
) -> dict[str, Any]:
    try:
        summary = get_model_governance_summary(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read governance summary: {exc}")
    return {"status": "ready", "governance": summary}


@router.get("/promotion-policy")
def get_current_promotion_policy() -> dict[str, Any]:
    try:
        policy = get_promotion_policy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read promotion policy: {exc}")
    return {"status": "ready", "promotion_policy": policy}


@router.get("/operations-readiness")
def get_operations_readiness(
    months: int = Query(default=6, ge=1, le=24, description="Observation window in months"),
) -> dict[str, Any]:
    try:
        readiness = get_impaye_operations_readiness(months=months)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No champion model found. Train and promote a model before operations readiness checks.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot compute operations readiness: {exc}")
    return {"status": "ready", "operations_readiness": readiness}


@router.get("/metrics/operations-readiness")
def get_operations_readiness_metrics(
    months: int = Query(default=6, ge=1, le=24, description="Observation window in months"),
) -> Response:
    try:
        readiness = get_impaye_operations_readiness(months=months)
        up = 1
    except Exception:
        readiness = {}
        up = 0
    payload = _render_operations_readiness_prometheus(readiness=readiness, months=months, up=up)
    return Response(content=payload, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.post("/train")
def train_model(request: TrainRequest) -> dict[str, Any]:
    if request.year_from and request.year_to and request.year_from > request.year_to:
        raise HTTPException(status_code=400, detail="year_from must be <= year_to")
    split_strategy = request.split_strategy.strip().lower()
    if split_strategy not in {"temporal", "random"}:
        raise HTTPException(status_code=400, detail="split_strategy must be one of ['temporal', 'random']")
    try:
        result = train_and_persist_model(
            year_from=request.year_from,
            year_to=request.year_to,
            test_size=request.test_size,
            random_state=request.random_state,
            split_strategy=split_strategy,
            promote_to_champion=request.promote_to_champion,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")
    return {"status": "trained", **result}


@router.post("/promote-latest")
def promote_latest(request: PromoteRequest) -> dict[str, Any]:
    try:
        result = promote_latest_to_champion(force=request.force, reason=request.reason)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {exc}")
    return {"status": "ok", **result}


@router.post("/predict")
def predict_impaye(request: PredictRequest) -> dict[str, Any]:
    payload = _normalize_prediction_payload(request)
    try:
        result = predict_impaye_probability(payload, threshold=request.threshold)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model found. Train one first via POST /api/v1/ml/train.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return {"status": "ok", "input": payload, **result}


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST  (forecast_model.ipynb — Prophet / SARIMA / XGBoost / LSTM)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/forecast")
def api_get_forecast(
    departement: str = Query("AUTO",                  description="Branche"),
    indicateur:  str = Query("primes_acquises_tnd",   description="Indicateur"),
    nb_mois:     int = Query(6, ge=1, le=24,          description="Horizon (mois)"),
    model:       str = Query("prophet",               description="Modèle : prophet | sarima | xgboost | lstm"),
) -> dict[str, Any]:
    try:
        return get_forecast(departement, indicateur, nb_mois)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/forecast/all")
def api_get_all_forecasts(
    nb_mois: int = Query(6, ge=1, le=24),
) -> dict[str, Any]:
    try:
        return get_all_forecasts(nb_mois)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION  (anomaly_detection.ipynb — IF + LOF + AE + DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/anomalies")
def api_detect_anomalies(
    departement:    str | None = Query(None,  description="Branche (Optionnel)"),
    contamination:  float      = Query(0.05,  ge=0.01, le=0.5, description="Taux d'anomalies attendu"),
    nb_mois_recent: int        = Query(24,    ge=6,    le=60,   description="Fenêtre historique (mois)"),
) -> dict[str, Any]:
    try:
        return detect_anomalies(departement, contamination, nb_mois_recent)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# DATA DRIFT  (data_drift_evidently.ipynb — Evidently AI + PSI)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/drift")
def api_detect_drift(
    departement:       str | None = Query(None, description="Branche (Optionnel)"),
    nb_mois_reference: int        = Query(12,   ge=6,  le=36, description="Fenêtre de référence (mois)"),
    nb_mois_courant:   int        = Query(6,    ge=3,  le=24, description="Fenêtre courante (mois)"),
) -> dict[str, Any]:
    try:
        return detect_drift(departement, nb_mois_reference, nb_mois_courant)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# CHURN PREDICTION  (churn_prediction_v3.ipynb — LightGBM calibré + SMOTE)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/churn/summary")
def api_churn_summary() -> dict[str, Any]:
    """
    Résumé portefeuille : taux_churn_pct, nb_churn, nb_polices,
    by_branche[], top_features[], risk_segments[].
    Modèle : churn_prediction_v3.ipynb — LightGBM calibré.
    """
    try:
        return get_churn_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/churn/predict")
def api_churn_predict(request: ChurnPredictRequest) -> dict[str, Any]:
    """
    Prédit la probabilité de résiliation d'une police.
    Retourne : churn_probability, churn_predicted, threshold, action.
    """
    try:
        return predict_churn(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD DETECTION  (fraud_detection.ipynb — IF 40% + AE 40% + LOF 20%)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/fraud/summary")
def api_fraud_summary(
    top_n: int = Query(default=10, ge=1, le=50, description="Nombre de cas suspects à retourner"),
) -> dict[str, Any]:
    """
    Scoring fraude ensemble.
    Niveaux : Normal < p90 | Modéré p90–p95 | Élevé p95–p99 | Critique > p99.
    """
    try:
        return get_fraud_summary(top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER SEGMENTATION  (customer_segmentation.ipynb — K-Means k=4 + RFM)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/segmentation/summary")
def api_segmentation_summary() -> dict[str, Any]:
    """
    Segmentation client K-Means + RFM.
    Retourne : nb_clients, nb_clusters, segments[] avec profil, radar et action.
    """
    try:
        return get_segmentation_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# RISK SCORING & PRICING  (risk_scoring_pricing.ipynb — GBM fréq. + sév.)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/risk/score")
def api_risk_score(request: RiskScoreRequest) -> dict[str, Any]:
    """
    Score de risque [0–1000] et prime technique (fréquence × sévérité × chargements).
    Retourne : risk_score, risk_label, prime_technique, loading_factor, components.
    """
    try:
        return score_risk(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/risk/table")
def api_risk_table() -> dict[str, Any]:
    """
    Table de risque agrégée Branche × Genre Véhicule.
    Colonnes : freq_sin_moy, sev_moy, sp_ratio_moy, prime_technique.
    """
    try:
        return get_risk_table()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM SEVERITY  (claim_severity.ipynb — XGBoost + LightGBM ensemble)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/claim/predict")
def api_claim_predict(request: ClaimPredictRequest) -> dict[str, Any]:
    """
    Prédit le coût final de règlement d'un sinistre.
    Retourne : predicted_severity, ci_low, ci_high, reserve_recommandee, severity_class.
    """
    try:
        return predict_claim_severity(request.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))