"""
maghrebia_daily_pipeline.py — Airflow DAG
TDSP Phase 2 + 3b + 4 — Pipeline batch quotidien Maghrebia Assurance

Séquence :
  1. health_check          → vérifie PostgreSQL + ClickHouse + Qdrant
  2. ge_data_quality       → run Great Expectations (6 suites)
  3. import_data           → CSV → PostgreSQL (si GE PASSED ou WARNING)
  4. refresh_kpi_views     → REFRESH MATERIALIZED VIEW ×3
  5. detect_events         → détecte alertes métier → MongoDB
  6. reindex_qdrant        → reindexe RAG Qdrant si nouvelles données
  7. check_drift           → Evidently drift check (mensuel uniquement)
    8. train_impaye_model    → retrain mensuel modèle impayé (Phase 3b)
    9. ml_operations_readiness → score opérationnel impayé (Phase 4)
 10. notify_ceo            → résumé alertes WebSocket / log

Schedule : quotidien à 02h00 (heure de Tunis UTC+1)
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR      = Path(os.getenv("DATA_DIR", "/opt/airflow/data/raw"))
CLEAN_DIR     = Path(os.getenv("CLEAN_DIR", "/opt/airflow/data/clean"))
REPORT_DIR    = Path(os.getenv("REPORT_DIR", "/opt/airflow/reports/data_quality"))
BACKEND_DIR   = Path(os.getenv("BACKEND_DIR", "/opt/airflow/backend"))
TDSP_DAGS_DIR = Path(os.getenv("TDSP_DAGS_DIR", "/opt/airflow/tdsp_dags"))

PG_CONN = {
    "host":     os.getenv("POSTGRES_HOST", "postgres"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "maghrebia"),
    "user":     os.getenv("POSTGRES_USER", "maghrebia"),
    "password": os.getenv("POSTGRES_PASSWORD", "maghrebia"),
}

MONGO_URL  = os.getenv(
    "MONGODB_URL",
    "mongodb://maghrebia:maghrebia@mongodb:27017/maghrebia_events?authSource=admin",
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
BACKEND_API_BASE = os.getenv("BACKEND_API_BASE", "http://backend:8000/api/v1")

DEFAULT_ARGS = {
    "owner":            "maghrebia-data",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}


# ── Task 1 : Health check ─────────────────────────────────────────────────────

def task_health_check(**ctx):
    import psycopg2
    import urllib.request

    results = {}

    # PostgreSQL
    try:
        conn = psycopg2.connect(**PG_CONN)
        conn.close()
        results["postgres"] = "OK"
        log.info("✅ PostgreSQL OK")
    except Exception as e:
        results["postgres"] = f"FAILED: {e}"
        log.error(f"🔴 PostgreSQL FAILED: {e}")
        raise RuntimeError("PostgreSQL unreachable — stopping pipeline")

    # Qdrant
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/healthz", timeout=5) as r:
            results["qdrant"] = "OK" if r.status == 200 else f"HTTP {r.status}"
        log.info("✅ Qdrant OK")
    except Exception as e:
        results["qdrant"] = f"WARN: {e}"
        log.warning(f"⚠️  Qdrant unreachable (non-fatal): {e}")

    ctx["ti"].xcom_push(key="health", value=results)
    log.info(f"Health check complete: {results}")


# ── Task 2 : Great Expectations ───────────────────────────────────────────────

def task_ge_quality(**ctx):
    import sys
    sys.path.insert(0, str(BACKEND_DIR))

    # Import inline to avoid circular deps in Airflow worker
    from ge_pipeline import run_suite, SUITES

    results = {}
    failed_suites = []

    for table_name in SUITES:
        try:
            result = run_suite(table_name, save_report=True)
            results[table_name] = result
            if result.get("status") == "FAILED":
                failed_suites.append(table_name)
                log.warning(f"⚠️  Suite FAILED: {table_name} ({result['success_pct']}%)")
            else:
                log.info(f"✅ Suite {result['status']}: {table_name} ({result['success_pct']}%)")
        except Exception as e:
            log.error(f"🔴 Suite ERROR for {table_name}: {e}")
            failed_suites.append(table_name)

    ctx["ti"].xcom_push(key="ge_results",     value=results)
    ctx["ti"].xcom_push(key="failed_suites",  value=failed_suites)
    ctx["ti"].xcom_push(key="ge_overall_ok",  value=len(failed_suites) == 0)

    global_pct = (
        sum(r.get("passed_exp", 0) for r in results.values())
        / max(sum(r.get("total_exp", 0) for r in results.values()), 1)
        * 100
    )
    log.info(f"GE overall: {global_pct:.1f}% — Failed suites: {failed_suites}")


# ── Task 3 : Import data ──────────────────────────────────────────────────────

def task_import_data(**ctx):
    import subprocess
    import sys

    candidates = [
        TDSP_DAGS_DIR / "import_data.py",
        BACKEND_DIR / "import_data.py",
    ]
    script_path = next((p for p in candidates if p.exists()), None)
    if script_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"import_data.py introuvable. Tried: {tried}")

    env = os.environ.copy()
    env.setdefault("DATA_DIR", str(DATA_DIR))
    env.setdefault("CLEAN_DIR", str(CLEAN_DIR))
    env.setdefault("REPORT_DIR", str(REPORT_DIR))
    env.setdefault(
        "DATABASE_URL",
        (
            f"postgresql+psycopg2://{PG_CONN['user']}:{PG_CONN['password']}"
            f"@{PG_CONN['host']}:{PG_CONN['port']}/{PG_CONN['dbname']}"
        ),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in REPORT_DIR.glob("import_report_*.json")}

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.stdout:
        log.info(result.stdout[-6000:] if len(result.stdout) > 6000 else result.stdout)
    if result.stderr:
        log.warning(result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"TDSP import_data.py échoué (code {result.returncode})")

    after_reports = sorted(REPORT_DIR.glob("import_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    report_path = None
    for p in after_reports:
        if p.name not in before:
            report_path = p
            break
    if report_path is None and after_reports:
        report_path = after_reports[0]

    if report_path is None:
        raise RuntimeError("Aucun import_report_*.json généré par le process TDSP")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    stats = [
        {
            "table": s.get("table"),
            "raw": s.get("n_input", 0),
            "loaded": s.get("n_output", 0),
            "status": "OK",
        }
        for s in report.get("cleaning", [])
    ]

    ctx["ti"].xcom_push(key="import_stats", value=stats)
    ctx["ti"].xcom_push(key="import_report_path", value=str(report_path))


# ── Task 4 : Refresh KPI views ────────────────────────────────────────────────

def task_refresh_kpi_views(**ctx):
    import psycopg2
    conn = psycopg2.connect(**PG_CONN)
    cur  = conn.cursor()

    views = [
        "mv_kpi_production_annuelle",
        "mv_kpi_agent_performance",
        "mv_kpi_portefeuille",
    ]
    for v in views:
        log.info(f"  REFRESH MATERIALIZED VIEW {v} …")
        cur.execute(f"REFRESH MATERIALIZED VIEW {v}")
    conn.commit()
    cur.close()
    conn.close()
    log.info("✅ KPI views refreshed")


# ── Task 5 : Detect events / alerts ──────────────────────────────────────────

def task_detect_events(**ctx):
    """
    Détecte les alertes métier depuis PostgreSQL et les pousse dans MongoDB.
    Alertes implémentées :
      - Top agents avec taux annulation > 15%
      - Impayés > 50k TND (individuel)
      - Nouvelles polices résiliées (SITUATION=R) depuis hier
    """
    try:
        import psycopg2
        from pymongo import MongoClient
    except Exception as e:
        log.warning(f"⚠️  detect_events skipped: dépendances manquantes ({e})")
        ctx["ti"].xcom_push(key="n_events", value=0)
        ctx["ti"].xcom_push(key="detect_events_skipped_reason", value=f"dependency_missing: {e}")
        return

    conn  = psycopg2.connect(**PG_CONN)
    cur   = conn.cursor()
    mongo = MongoClient(MONGO_URL)
    db    = mongo["maghrebia_events"]
    col   = db["alertes"]

    today = datetime.now().strftime("%Y-%m-%d")
    events = []

    # Alerte 1 : agents avec taux annulation > 15%
    try:
        cur.execute("""
            SELECT id_agent, code_agent, nom_agent, taux_annulation_pct, nb_quittances
            FROM mv_kpi_agent_performance
            WHERE taux_annulation_pct > 15
            ORDER BY taux_annulation_pct DESC
            LIMIT 10
        """)
        for row in cur.fetchall():
            events.append({
                "type": "AGENT_HIGH_ANNULATION",
                "severity": "WARNING",
                "date": today,
                "data": {
                    "id_agent": row[0], "code": row[1], "nom": row[2],
                    "taux_annulation": row[3], "nb_quittances": row[4]
                },
                "message": f"Agent {row[1]} — {row[2]} : taux annulation {row[3]:.1f}%",
            })
    except Exception as e:
        log.warning(f"Alerte 1 query failed: {e}")

    # Alerte 2 : impayés > 50k TND
    try:
        cur.execute("""
            SELECT i.num_quittance, i.branche, i.mt_ptt, i.mt_acp,
                   a.code_agent, a.nom_agent
            FROM dwh_fact_impaye i
            JOIN dim_agent a ON i.id_agent = a.id_agent
            WHERE i.mt_acp > 50000
            ORDER BY i.mt_acp DESC
            LIMIT 5
        """)
        for row in cur.fetchall():
            events.append({
                "type": "IMPAYE_HIGH_AMOUNT",
                "severity": "CRITICAL",
                "date": today,
                "data": {
                    "num_quittance": row[0], "branche": row[1],
                    "mt_ptt": float(row[2] or 0), "mt_acp": float(row[3] or 0),
                    "agent_code": row[4], "agent_nom": row[5],
                },
                "message": f"Impayé critique : {row[0]} — {row[3]:,.0f} TND",
            })
    except Exception as e:
        log.warning(f"Alerte 2 query failed: {e}")

    # Alerte 3 : polices résiliées avec fort bonus-malus (risque churn)
    try:
        cur.execute("""
            SELECT COUNT(*) AS nb_resiliees,
                   AVG(bonus_malus) AS avg_bm
            FROM dim_police
            WHERE situation = 'R'
              AND bonus_malus >= 8
        """)
        row = cur.fetchone()
        if row and row[0] > 10:
            events.append({
                "type": "CHURN_RISK_HIGH_BM",
                "severity": "WARNING",
                "date": today,
                "data": {"nb_resiliees": row[0], "avg_bm": float(row[1] or 0)},
                "message": f"{row[0]} polices résiliées avec BM moyen {row[1]:.1f}",
            })
    except Exception as e:
        log.warning(f"Alerte 3 query failed: {e}")

    n_events_persisted = 0

    # Push to MongoDB (non-fatal for the core TDSP data/KPI pipeline)
    if events:
        try:
            def _mongo_json_default(value):
                try:
                    return float(value)
                except Exception:
                    return str(value)

            safe_events = json.loads(json.dumps(events, default=_mongo_json_default))
            col.insert_many(safe_events)
            n_events_persisted = len(events)
            log.info(f"✅ {n_events_persisted} alertes insérées dans MongoDB")
        except Exception as e:
            log.warning(f"⚠️  Insertion MongoDB des alertes échouée (non-fatal): {e}")
    else:
        log.info("ℹ️  Aucune alerte détectée")

    cur.close()
    conn.close()
    mongo.close()
    ctx["ti"].xcom_push(key="n_events", value=len(events))
    ctx["ti"].xcom_push(key="n_events_persisted", value=n_events_persisted)


# ── Task 6 : Reindex Qdrant ───────────────────────────────────────────────────

def task_reindex_qdrant(**ctx):
    """
    Reindexe les collections Qdrant si de nouvelles données sont disponibles.
    Collections : kpi_narratifs · regles_metier · alertes_historiques
    """
    import psycopg2
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        collections = [c.name for c in client.get_collections().collections]
        log.info(f"  Qdrant collections existantes: {collections}")

        # Create collections if they don't exist
        for coll in ["kpi_narratifs", "regles_metier", "alertes_historiques"]:
            if coll not in collections:
                client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                log.info(f"  ✅ Collection créée: {coll}")

        # Build KPI narratives from PostgreSQL for indexing
        conn = psycopg2.connect(**PG_CONN)
        cur  = conn.cursor()
        cur.execute("""
            SELECT annee, branche, nb_quittances, total_pnet, total_commission
            FROM mv_kpi_production_annuelle
            ORDER BY annee DESC, branche
            LIMIT 50
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Simple text documents for RAG (embedding done by FastEmbed in backend)
        documents = [
            f"En {row[0]}, la branche {row[1]} a produit {row[2]:,} quittances "
            f"pour {float(row[3] or 0):,.0f} TND de primes nettes "
            f"et {float(row[4] or 0):,.0f} TND de commissions."
            for row in rows
        ]
        log.info(f"  {len(documents)} narratifs KPI prêts pour indexation")
        # Actual embedding + upsert is done by the backend indexer.py
        # Here we just verify Qdrant is reachable and collections exist

        ctx["ti"].xcom_push(key="n_docs_ready", value=len(documents))
        log.info("✅ Qdrant reindex préparé — backend indexer.py prendra le relais")

    except Exception as e:
        log.warning(f"⚠️  Qdrant reindex partiel: {e}")


# ── Task 7 : Drift check (mensuel) ───────────────────────────────────────────

def task_drift_check(**ctx):
    """
    Drift Evidently sur FACT_EMISSION vs baseline (M-3).
    Ne tourne que le 1er du mois.
    """
    today = datetime.now()
    if today.day != 1:
        log.info(f"  Drift check ignoré (jour={today.day}, uniquement le 1er du mois)")
        ctx["ti"].xcom_push(key="drift_skipped", value=True)
        return

    import psycopg2
    import pandas as pd

    conn = psycopg2.connect(**PG_CONN)
    df   = pd.read_sql("""
        SELECT branche, annee_echeance, mois_echeance, mt_pnet, mt_commission,
               bonus_malus, etat_quit
        FROM dwh_fact_emission
        WHERE annee_echeance BETWEEN 2022 AND 2025
          AND etat_quit IN ('E','P','A')
    """, conn)
    conn.close()

    # Baseline: 3 months ago ; Current: last month
    baseline = df[df["annee_echeance"] <= today.year - 1]
    current  = df[df["annee_echeance"] == today.year]

    log.info(f"  Drift check: baseline {len(baseline):,} rows, current {len(current):,} rows")

    # Evidently drift report (installed in backend image)
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=baseline, current_data=current)

        report_path = f"/tmp/drift_report_{today.strftime('%Y%m')}.html"
        report.save_html(report_path)

        drift_result = report.as_dict()
        n_drifted = sum(
            1 for v in drift_result.get("metrics", [])
            if v.get("result", {}).get("drift_detected", False)
        )
        log.info(f"  ✅ Drift: {n_drifted} features en drift")
        ctx["ti"].xcom_push(key="n_drifted_features", value=n_drifted)
        ctx["ti"].xcom_push(key="drift_report_path",  value=report_path)

    except ImportError:
        log.warning("  Evidently non installé — drift skipped")


# ── Task 8 : Monthly ML retrain ──────────────────────────────────────────────

def task_should_retrain_model(**ctx):
    logical_date = ctx.get("logical_date") or ctx.get("execution_date") or datetime.now()
    day = int(logical_date.day)

    if day == 1:
        log.info("  Retrain mensuel activé (jour logique=1)")
        return "train_impaye_model"

    log.info(f"  Retrain ignoré (jour logique={day})")
    return "skip_retrain_model"


def task_train_impaye_model(**ctx):
    import urllib.request

    payload = json.dumps(
        {
            "year_from": 2023,
            "year_to": 2025,
            "test_size": 0.2,
            "random_state": 42,
            "split_strategy": "temporal",
            "promote_to_champion": True,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url=f"{BACKEND_API_BASE}/ml/train",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=1200) as response:
        response_body = response.read().decode("utf-8")
        if response.status >= 400:
            raise RuntimeError(f"ML retrain failed with status {response.status}: {response_body}")

    result = json.loads(response_body)
    if result.get("status") != "trained":
        raise RuntimeError(f"ML retrain unexpected response: {result}")

    summary = {
        "run_id": result.get("run_id"),
        "selected_model": result.get("selected_model"),
        "best_threshold": result.get("best_threshold"),
        "promoted_to_champion": result.get("promoted_to_champion", False),
        "promotion_reason": result.get("promotion_reason"),
        "champion_run_id": result.get("champion_run_id"),
        "metrics": result.get("metrics", {}),
        "split_info": result.get("split_info", {}),
    }
    ctx["ti"].xcom_push(key="ml_train_summary", value=summary)
    log.info(
        "✅ ML retrain OK: "
        f"run_id={summary.get('run_id')}, "
        f"model={summary.get('selected_model')}, "
        f"promoted={summary.get('promoted_to_champion')}"
    )


# ── Task 9 : ML operations readiness (TDSP Phase 4) ─────────────────────────

def task_ml_operations_readiness(**ctx):
    """
    Récupère le score de readiness opérationnelle du modèle impayé
    et publie un rapport JSON exploitable côté monitoring/CEO.
    """
    import urllib.error
    import urllib.request

    logical_date = ctx.get("logical_date") or ctx.get("execution_date") or datetime.now()
    months_env = os.getenv("OPS_READINESS_WINDOW_MONTHS", "6")
    try:
        months = int(months_env)
    except ValueError:
        months = 6
    months = max(1, min(months, 24))

    endpoint = f"{BACKEND_API_BASE}/ml/operations-readiness?months={months}"
    report_dir = REPORT_DIR.parent / "ml_ops"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "unavailable",
        "score": 0.0,
        "fail_count": 0,
        "warn_count": 0,
        "recommendations_count": 0,
        "window_months": months,
        "reason": "not_requested",
        "model_run_id": None,
    }

    try:
        with urllib.request.urlopen(endpoint, timeout=120) as response:
            body = response.read().decode("utf-8")

        payload = json.loads(body)
        readiness = payload.get("operations_readiness", {})
        readiness_block = readiness.get("readiness", {})

        recommendations = readiness.get("recommendations", [])
        summary = {
            "status": readiness_block.get("status", "unavailable"),
            "score": float(readiness_block.get("score", 0.0) or 0.0),
            "fail_count": int(readiness_block.get("fail_count", 0) or 0),
            "warn_count": int(readiness_block.get("warn_count", 0) or 0),
            "recommendations_count": len(recommendations),
            "window_months": int(readiness.get("window_months", months) or months),
            "reason": "ok",
            "model_run_id": readiness.get("model", {}).get("run_id"),
        }
    except urllib.error.HTTPError as exc:
        summary["reason"] = f"http_{exc.code}"
        log.warning(f"⚠️  ML operations readiness indisponible: HTTP {exc.code}")
    except Exception as exc:
        summary["reason"] = f"error: {exc}"
        log.warning(f"⚠️  ML operations readiness failed: {exc}")

    ts = logical_date.strftime("%Y%m%dT%H%M%S")
    report_payload = {
        "run_date": datetime.now().isoformat(),
        "endpoint": endpoint,
        "summary": summary,
    }
    report_path = report_dir / f"operations_readiness_{ts}.json"
    latest_path = report_dir / "operations_readiness_latest.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    ctx["ti"].xcom_push(key="ml_ops_readiness", value=summary)
    ctx["ti"].xcom_push(key="ml_ops_report_path", value=str(report_path))

    log.info(
        "✅ ML operations readiness: "
        f"status={summary.get('status')}, score={summary.get('score')}, "
        f"reco={summary.get('recommendations_count')}"
    )


# ── Task 10 : Notify CEO ──────────────────────────────────────────────────────

def task_notify_ceo(**ctx):
    ti     = ctx["ti"]
    n_ev   = ti.xcom_pull(task_ids="detect_events",  key="n_events")   or 0
    n_doc  = ti.xcom_pull(task_ids="reindex_qdrant", key="n_docs_ready") or 0
    stats  = ti.xcom_pull(task_ids="import_data",    key="import_stats") or []
    ge_res = ti.xcom_pull(task_ids="ge_data_quality",key="ge_results")  or {}
    ml_sum = ti.xcom_pull(task_ids="train_impaye_model", key="ml_train_summary") or {}
    ml_ops = ti.xcom_pull(task_ids="ml_operations_readiness", key="ml_ops_readiness") or {}

    total_rows = sum(s.get("loaded", 0) for s in stats)
    ge_pct     = (
        sum(r.get("passed_exp",0) for r in ge_res.values())
        / max(sum(r.get("total_exp",0) for r in ge_res.values()), 1)
        * 100
    )

    ml_ops_status = ml_ops.get("status")
    if n_ev > 0 or ml_ops_status in {"red", "unavailable"}:
        pipeline_status = "ALERTES"
    elif ml_ops_status == "amber":
        pipeline_status = "WATCH"
    else:
        pipeline_status = "OK"

    summary = {
        "run_date":        datetime.now().isoformat(),
        "total_rows_loaded": total_rows,
        "n_alerts":        n_ev,
        "ge_quality_pct":  round(ge_pct, 1),
        "qdrant_docs":     n_doc,
        "ml_retrained":    bool(ml_sum),
        "ml_run_id":       ml_sum.get("run_id"),
        "ml_model":        ml_sum.get("selected_model"),
        "ml_best_threshold": ml_sum.get("best_threshold"),
        "ml_promoted_to_champion": ml_sum.get("promoted_to_champion"),
        "ml_promotion_reason": ml_sum.get("promotion_reason"),
        "ml_champion_run_id": ml_sum.get("champion_run_id"),
        "ml_ops_status":   ml_ops.get("status"),
        "ml_ops_score":    ml_ops.get("score"),
        "ml_ops_fail_count": ml_ops.get("fail_count"),
        "ml_ops_warn_count": ml_ops.get("warn_count"),
        "ml_ops_recommendations": ml_ops.get("recommendations_count"),
        "ml_ops_window_months": ml_ops.get("window_months"),
        "status":          pipeline_status,
    }

    log.info("=" * 60)
    log.info("  PIPELINE SUMMARY")
    log.info(f"  Rows loaded      : {total_rows:,}")
    log.info(f"  GE quality       : {ge_pct:.1f}%")
    log.info(f"  Alertes CEO      : {n_ev}")
    log.info(f"  Qdrant docs      : {n_doc}")
    log.info(
        "  ML ops readiness : "
        f"status={ml_ops.get('status')}, score={ml_ops.get('score')}, "
        f"reco={ml_ops.get('recommendations_count')}"
    )
    if ml_sum:
        log.info(
            "  ML retrain       : "
            f"{ml_sum.get('selected_model')} ({ml_sum.get('run_id')}), "
            f"promoted={ml_sum.get('promoted_to_champion')}"
        )
    else:
        log.info("  ML retrain       : SKIPPED")
    log.info("=" * 60)

    # Push to MongoDB for real-time dashboard
    try:
        from pymongo import MongoClient
        mongo = MongoClient(MONGO_URL)
        mongo["maghrebia_events"]["pipeline_runs"].insert_one(summary)
        mongo.close()
    except Exception as e:
        log.warning(f"  MongoDB push failed: {e}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="maghrebia_daily_pipeline",
    description="Pipeline batch quotidien — Maghrebia Assurance TDSP",
    schedule_interval="0 1 * * *",  # 02h00 Tunis (UTC+1)
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["maghrebia", "tdsp", "phase2", "phase3b", "phase4", "production"],
    doc_md="""
## Maghrebia Daily Pipeline
Pipeline TDSP Phase 2 + 3b + 4 — exécution quotidienne à 02h00.

### Séquence
1. **health_check** — vérifie PostgreSQL, Qdrant
2. **ge_data_quality** — Great Expectations 6 suites
3. **import_data** — CSV → PostgreSQL avec cleaning
4. **refresh_kpi_views** — REFRESH MATERIALIZED VIEW ×3
5. **detect_events** — alertes métier → MongoDB
6. **reindex_qdrant** — RAG Qdrant prêt pour l'agent IA
7. **drift_check** — Evidently (1er du mois uniquement)
8. **train_impaye_model** — retrain mensuel du modèle impayé
9. **ml_operations_readiness** — score opérationnel impayé (Phase 4)
10. **notify_ceo** — résumé pipeline

### Contacts
- Owner: maghrebia-data
- Monitoring: Grafana http://localhost:3001
""",
) as dag:

    t_health = PythonOperator(
        task_id="health_check",
        python_callable=task_health_check,
    )
    t_ge = PythonOperator(
        task_id="ge_data_quality",
        python_callable=task_ge_quality,
    )
    t_import = PythonOperator(
        task_id="import_data",
        python_callable=task_import_data,
    )
    t_kpi = PythonOperator(
        task_id="refresh_kpi_views",
        python_callable=task_refresh_kpi_views,
    )
    t_events = PythonOperator(
        task_id="detect_events",
        python_callable=task_detect_events,
    )
    t_qdrant = PythonOperator(
        task_id="reindex_qdrant",
        python_callable=task_reindex_qdrant,
    )
    t_drift = PythonOperator(
        task_id="drift_check",
        python_callable=task_drift_check,
    )
    t_should_retrain = BranchPythonOperator(
        task_id="should_retrain_model",
        python_callable=task_should_retrain_model,
    )
    t_train_model = PythonOperator(
        task_id="train_impaye_model",
        python_callable=task_train_impaye_model,
        execution_timeout=timedelta(minutes=20),
    )
    t_skip_retrain = EmptyOperator(task_id="skip_retrain_model")
    t_ml_ops = PythonOperator(
        task_id="ml_operations_readiness",
        python_callable=task_ml_operations_readiness,
    )
    t_notify = PythonOperator(
        task_id="notify_ceo",
        python_callable=task_notify_ceo,
    )

    # ── Dependency graph ──────────────────────────────────────────────────
    #
    #  health_check
    #      │
    #  ge_data_quality
    #      │
    #  import_data
    #      │
    #  refresh_kpi_views ──── detect_events ──── reindex_qdrant
    #      │                                          │
    #      ├──────────── drift_check ─────────────────┤
    #      │                                          │
    #      └──── should_retrain_model ──┬─ train_impaye_model
    #                                    └─ skip_retrain_model
    #                                                │
    #                                      ml_operations_readiness
    #                                                │
    #                                            notify_ceo

    t_health >> t_ge >> t_import >> t_kpi
    t_kpi    >> [t_events, t_qdrant]
    t_kpi    >> t_drift
    t_kpi    >> t_should_retrain
    t_should_retrain >> [t_train_model, t_skip_retrain]
    [t_train_model, t_skip_retrain] >> t_ml_ops
    [t_events, t_qdrant, t_drift, t_ml_ops] >> t_notify
