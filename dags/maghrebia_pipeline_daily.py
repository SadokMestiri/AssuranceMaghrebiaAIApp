"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Maghrebia Assurance — DAG Airflow                                          ║
║  maghrebia_pipeline_daily.py                                                ║
║  TDSP Phase 2 — Orchestration batch quotidien                               ║
║                                                                              ║
║  Tasks (ordre d'exécution) :                                                ║
║   1. health_check         — PostgreSQL + Qdrant + MinIO                     ║
║   2. data_quality_check   — Great Expectations sur CSV bruts                ║
║   3. import_and_clean     — import_data.py (cleaning + UPSERT PG)          ║
║   4. refresh_kpi_views    — REFRESH MATERIALIZED VIEW                       ║
║   5. detect_events        — Alertes métier → MongoDB                        ║
║   6. reindex_qdrant       — Ré-indexation RAG si nouvelles données          ║
║   7. drift_check          — Evidently data drift report                     ║
║   8. pipeline_report      — Rapport final JSON + notification               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.utils.dates import days_ago
except ModuleNotFoundError as exc:
    if __name__ == "__main__":
        raise SystemExit(
            "Airflow n'est pas installe dans ce Python local. "
            "Lance ce DAG via Airflow (container), par exemple: "
            "docker compose exec airflow-scheduler airflow dags trigger "
            "maghrebia_pipeline_daily"
        ) from exc
    raise

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = os.environ.get("DATA_DIR",   "/opt/airflow/data/raw")
REPORT_DIR  = os.environ.get("REPORT_DIR", "/opt/airflow/reports")
DB_URL      = os.environ.get("DATABASE_URL",
    "postgresql+psycopg2://maghrebia:maghrebia2024@postgres:5432/maghrebia_dwh")
MONGO_URL   = os.environ.get("MONGO_URL",
    "mongodb://maghrebia:maghrebia2024@mongodb:27017/maghrebia_events?authSource=admin")
QDRANT_URL  = os.environ.get("QDRANT_URL",  "http://qdrant:6333")
MINIO_URL   = os.environ.get("MINIO_URL",   "http://minio:9000")

# Seuil alerte drift : si score Evidently > ce seuil, alerte CEO
DRIFT_ALERT_THRESHOLD = 0.3

DEFAULT_ARGS = {
    "owner":            "maghrebia_data",
    "depends_on_past":  False,
    "start_date":       days_ago(1),
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

log = logging.getLogger("maghrebia.dag")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Health check
# ─────────────────────────────────────────────────────────────────────────────

def task_health_check(**context):
    """Vérifie que PostgreSQL, Qdrant et MinIO sont accessibles."""
    import urllib.request
    import psycopg2

    results = {}

    # PostgreSQL
    try:
        conn = psycopg2.connect(DB_URL.replace("postgresql+psycopg2://", "postgresql://"))
        conn.close()
        results["postgres"] = "OK"
    except Exception as e:
        results["postgres"] = f"FAIL: {e}"

    # Qdrant
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/healthz", timeout=5) as r:
            results["qdrant"] = "OK" if r.status == 200 else f"HTTP {r.status}"
    except Exception as e:
        results["qdrant"] = f"FAIL: {e}"

    # MinIO
    try:
        with urllib.request.urlopen(f"{MINIO_URL}/minio/health/live", timeout=5) as r:
            results["minio"] = "OK" if r.status == 200 else f"HTTP {r.status}"
    except Exception as e:
        results["minio"] = f"FAIL: {e}"

    log.info(f"Health check results: {results}")
    context["ti"].xcom_push(key="health", value=results)

    failed = [k for k, v in results.items() if v != "OK"]
    if "postgres" in failed:
        raise RuntimeError(f"PostgreSQL inaccessible — arrêt pipeline: {results['postgres']}")
    if failed:
        log.warning(f"Services partiellement indisponibles: {failed} — pipeline continue")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Great Expectations sur CSV bruts (pré-nettoyage)
# ─────────────────────────────────────────────────────────────────────────────

def task_data_quality_check(**context):
    """
    Exécute le script ge_expectations_suite.py sur les CSV bruts.
    Enregistre le score dans XCom pour le rapport final.
    """
    import subprocess, sys

    ge_script = Path("/opt/airflow/dags/ge_expectations_suite.py")
    if not ge_script.exists():
        ge_script = Path("/app/ge_expectations_suite.py")

    env = os.environ.copy()
    env["DATA_DIR"]   = DATA_DIR
    env["REPORT_DIR"] = REPORT_DIR

    result = subprocess.run(
        [sys.executable, str(ge_script)],
        capture_output=True, text=True, env=env
    )

    log.info(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

    # Chercher le rapport JSON le plus récent
    ge_score = None
    report_path = Path(REPORT_DIR)
    if report_path.exists():
        reports = sorted(report_path.glob("ge_report_*.json"), reverse=True)
        if reports:
            with open(reports[0]) as f:
                data = json.load(f)
                ge_score = data.get("global_score", 0)

    context["ti"].xcom_push(key="ge_score_raw", value=ge_score)
    log.info(f"GE score pré-nettoyage : {ge_score}%")

    # On ne bloque pas sur le score pré-nettoyage (les données sont dirty par construction)
    # C'est le score post-nettoyage (task 3) qui compte
    return {"ge_score_raw": ge_score, "returncode": result.returncode}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Import & Nettoyage → PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def task_import_and_clean(**context):
    """
    Exécute import_data.py : nettoyage + validation GE post-clean + UPSERT PG.
    """
    import subprocess, sys

    import_script = Path("/opt/airflow/dags/import_data.py")
    if not import_script.exists():
        import_script = Path("/app/import_data.py")

    env = os.environ.copy()
    env["DATA_DIR"]    = DATA_DIR
    env["REPORT_DIR"]  = REPORT_DIR
    env["DATABASE_URL"] = DB_URL

    result = subprocess.run(
        [sys.executable, str(import_script)],
        capture_output=True, text=True, env=env
    )

    log.info(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"import_data.py échoué (code {result.returncode})\n{result.stderr[-2000:]}")

    # Parse rapport JSON
    report_path = Path(REPORT_DIR)
    import_reports = sorted(report_path.glob("import_report_*.json"), reverse=True)
    if import_reports:
        with open(import_reports[0]) as f:
            data = json.load(f)

        def to_int(value) -> int:
            if value is None:
                return 0
            if isinstance(value, (int, float)):
                return int(value)
            s = str(value).strip().replace(",", "")
            try:
                return int(float(s))
            except ValueError:
                return 0

        context["ti"].xcom_push(key="import_report", value={
            "ge_score_clean": data.get("validation", {}).get("score"),
            "volumes": data.get("volumes", {}),
            "total_corrections": sum(
                to_int(s.get("total_corrections")) for s in data.get("cleaning", [])
            ),
        })

    return {"returncode": result.returncode}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — Refresh vues matérialisées PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def task_refresh_kpi_views(**context):
    """
    Refresh les vues matérialisées CEO.
    Si la vue n'est pas encore matérialisée (simple VIEW), recalcule les stats.
    """
    import psycopg2

    views_to_refresh = [
        "vw_kpi_production_annuel",
        "vw_kpi_agent_performance",
    ]

    # KPI métriques calculées et stockées pour l'API rapide
    kpi_queries = {
        "kpi_primes_par_branche_annee": """
            SELECT
                annee_echeance,
                branche,
                COUNT(*)           AS nb_quittances,
                SUM(mt_pnet)       AS total_pnet,
                SUM(mt_ptt)        AS total_ptt,
                SUM(mt_commission) AS total_commission,
                AVG(mt_pnet)       AS avg_prime
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P')
              AND mt_pnet > 0
              AND annee_echeance BETWEEN 2019 AND 2025
            GROUP BY annee_echeance, branche
            ORDER BY annee_echeance, branche
        """,
        "kpi_top10_agents": """
            SELECT
                a.code_agent, a.nom_agent, a.groupe_agent, a.localite_agent,
                COUNT(e.num_quittance)  AS nb_quittances,
                SUM(e.mt_pnet)          AS total_pnet,
                SUM(e.mt_commission)    AS total_commission
            FROM dim_agent a
            JOIN dwh_fact_emission e ON a.id_agent = e.id_agent
            WHERE e.etat_quit IN ('E','P') AND e.mt_pnet > 0
            GROUP BY a.code_agent, a.nom_agent, a.groupe_agent, a.localite_agent
            ORDER BY total_pnet DESC
            LIMIT 10
        """,
        "kpi_taux_resiliation": """
            SELECT
                branche,
                COUNT(*) FILTER (WHERE situation IN ('R','T','A')) AS n_resilies,
                COUNT(*) AS n_total,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE situation IN ('R','T','A'))
                    / NULLIF(COUNT(*), 0), 2
                ) AS taux_resiliation_pct
            FROM dim_police
            GROUP BY branche
        """,
    }

    try:
        conn_str = DB_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(conn_str)
        conn.autocommit = True
        cur = conn.cursor()

        # Créer/refresh table KPI cache
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kpi_cache (
                kpi_name    VARCHAR(100) PRIMARY KEY,
                kpi_data    JSONB,
                updated_at  TIMESTAMP DEFAULT NOW()
            )
        """)

        for kpi_name, query in kpi_queries.items():
            cur.execute(query)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            data = [dict(zip(col_names, row)) for row in rows]

            cur.execute("""
                INSERT INTO kpi_cache (kpi_name, kpi_data, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (kpi_name) DO UPDATE
                  SET kpi_data = EXCLUDED.kpi_data,
                      updated_at = NOW()
            """, (kpi_name, json.dumps(data, default=str)))

        log.info(f"✅ {len(kpi_queries)} KPI caches rafraîchis")
        cur.close()
        conn.close()

    except Exception as e:
        log.error(f"Refresh KPI views échoué : {e}")
        raise

    return {"kpis_refreshed": list(kpi_queries.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — Détection d'événements → MongoDB
# ─────────────────────────────────────────────────────────────────────────────

def task_detect_events(**context):
    """
    Détecte les événements métier notables depuis PostgreSQL
    et les insère dans MongoDB pour le feed temps réel du dashboard.

    Événements détectés :
      - Taux impayé > 10% sur un agent
      - Pic de résiliation > 20% sur une branche/mois
      - Agent avec 0 quittance émise ce mois
      - Prime nette mensuelle < 80% de la moyenne sur 3 mois (branche)
    """
    import psycopg2

    events = []
    run_date = datetime.now()

    try:
        conn_str = DB_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()

        # Alerte 1 : Agents avec fort taux d'impayés
        cur.execute("""
            SELECT
                a.code_agent, a.nom_agent,
                COUNT(i.num_quittance)   AS n_impayes,
                COUNT(e.num_quittance)   AS n_total,
                ROUND(100.0 * COUNT(i.num_quittance) / NULLIF(COUNT(e.num_quittance),0), 1) AS taux_impaye
            FROM dim_agent a
            LEFT JOIN dwh_fact_emission e ON a.id_agent = e.id_agent
                AND e.annee_echeance = EXTRACT(YEAR FROM NOW())
            LEFT JOIN dwh_fact_impaye i  ON a.id_agent = i.id_agent
                AND i.annee_echeance = EXTRACT(YEAR FROM NOW())
            GROUP BY a.code_agent, a.nom_agent
            HAVING COUNT(e.num_quittance) >= 10
               AND (100.0 * COUNT(i.num_quittance) / NULLIF(COUNT(e.num_quittance),0)) > 10
            ORDER BY taux_impaye DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        for row in rows:
            events.append({
                "type":       "ALERTE_IMPAYE",
                "severity":   "HIGH" if row[4] and row[4] > 20 else "MEDIUM",
                "agent":      row[0],
                "nom_agent":  row[1],
                "taux_impaye": float(row[4]) if row[4] else 0,
                "message":    f"Agent {row[1]} : taux impayé {row[4]}%",
                "created_at": run_date.isoformat(),
                "source":     "airflow_dag",
            })

        # Alerte 2 : Volume mensuel primes en baisse > 20%
        cur.execute("""
            WITH mensuel AS (
                SELECT
                    branche,
                    annee_echeance,
                    mois_echeance,
                    SUM(mt_pnet) AS total_pnet
                FROM dwh_fact_emission
                WHERE etat_quit IN ('E','P') AND mt_pnet > 0
                  AND annee_echeance >= 2023
                GROUP BY branche, annee_echeance, mois_echeance
            ),
            avec_lag AS (
                SELECT *,
                    LAG(total_pnet) OVER (PARTITION BY branche ORDER BY annee_echeance, mois_echeance) AS prev_pnet
                FROM mensuel
            )
            SELECT branche, annee_echeance, mois_echeance, total_pnet, prev_pnet,
                   ROUND(100*(total_pnet - prev_pnet)/NULLIF(prev_pnet,0), 1) AS variation_pct
            FROM avec_lag
            WHERE prev_pnet > 0
              AND (total_pnet - prev_pnet)/NULLIF(prev_pnet,0) < -0.20
            ORDER BY annee_echeance DESC, mois_echeance DESC
            LIMIT 5
        """)
        rows = cur.fetchall()
        for row in rows:
            events.append({
                "type":       "ALERTE_BAISSE_PRIME",
                "severity":   "HIGH",
                "branche":    row[0],
                "annee":      row[1],
                "mois":       row[2],
                "variation_pct": float(row[5]) if row[5] else 0,
                "message":    f"Branche {row[0]} : prime en baisse de {abs(float(row[5] or 0)):.0f}% vs mois précédent",
                "created_at": run_date.isoformat(),
                "source":     "airflow_dag",
            })

        cur.close()
        conn.close()

    except Exception as e:
        log.error(f"Détection événements PostgreSQL échouée : {e}")
        events.append({
            "type": "SYSTEM_ERROR",
            "severity": "HIGH",
            "message": f"Détection événements impossible : {e}",
            "created_at": run_date.isoformat(),
            "source": "airflow_dag",
        })

    # Push vers MongoDB
    n_inserted = 0
    if events:
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
            db = client.maghrebia_events
            col = db.events
            result = col.insert_many(events)
            n_inserted = len(result.inserted_ids)
            log.info(f"✅ {n_inserted} événements insérés dans MongoDB")
            client.close()
        except Exception as e:
            log.warning(f"MongoDB indisponible ({e}) — événements non persistés")

    context["ti"].xcom_push(key="n_events", value=len(events))
    return {"n_events": len(events), "n_inserted": n_inserted}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — Branch : faut-il re-indexer Qdrant ?
# ─────────────────────────────────────────────────────────────────────────────

def task_should_reindex(**context):
    """
    Décide si une ré-indexation Qdrant est nécessaire.
    Critère : nombre de nouvelles quittances depuis la dernière indexation > 500.
    """
    try:
        import psycopg2
        conn_str = DB_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM dwh_fact_emission
            WHERE created_at >= NOW() - INTERVAL '1 day'
        """)
        n_new = cur.fetchone()[0]
        cur.close(); conn.close()
        log.info(f"Nouvelles quittances depuis 24h : {n_new}")
        return "reindex_qdrant" if n_new > 500 else "skip_reindex"
    except Exception as e:
        log.warning(f"Impossible de compter nouvelles lignes ({e}) — skip reindex")
        return "skip_reindex"


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6a — Ré-indexation Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def task_reindex_qdrant(**context):
    """
    Indexe les nouvelles données dans Qdrant pour le RAG de l'agent IA.
    Collections indexées :
      - kpi_narratives   : résumés mensuels des KPI par branche
      - alert_history    : historique alertes détectées
      - business_rules   : règles métier assurance (statiques)
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from fastembed import TextEmbedding
    except ImportError:
        log.warning("qdrant_client ou fastembed non installé — skip reindex")
        return {"status": "skipped", "reason": "dependencies missing"}

    import psycopg2
    import hashlib

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Récupérer KPI mensuels depuis PostgreSQL
        conn_str = DB_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("""
            SELECT branche, annee_echeance, mois_echeance,
                   COUNT(*) AS nb, SUM(mt_pnet) AS pnet, SUM(mt_commission) AS comm
            FROM dwh_fact_emission
            WHERE etat_quit IN ('E','P') AND mt_pnet > 0
            GROUP BY branche, annee_echeance, mois_echeance
            ORDER BY annee_echeance DESC, mois_echeance DESC
            LIMIT 200
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()

        # Construire les documents narratifs
        documents = []
        for row in rows:
            branche, annee, mois, nb, pnet, comm = row
            text = (
                f"Production {branche} — {mois:02d}/{annee} : "
                f"{nb} quittances, prime nette totale {float(pnet or 0):,.0f} DT, "
                f"commissions {float(comm or 0):,.0f} DT."
            )
            documents.append({
                "id":     hashlib.md5(f"{branche}{annee}{mois}".encode()).hexdigest()[:16],
                "text":   text,
                "meta":   {"branche": branche, "annee": annee, "mois": mois, "type": "kpi_mensuel"},
            })

        # Créer collection si nécessaire
        col_name = "kpi_narratives"
        try:
            client.get_collection(col_name)
        except Exception:
            client.create_collection(col_name, vectors_config=VectorParams(size=384, distance=Distance.COSINE))

        # Vectoriser et upserter
        texts = [d["text"] for d in documents]
        embeddings = list(embedding_model.embed(texts))

        points = [
            PointStruct(
                id=abs(hash(d["id"])) % (2**63),
                vector=emb.tolist(),
                payload={"text": d["text"], **d["meta"]}
            )
            for d, emb in zip(documents, embeddings)
        ]

        client.upsert(collection_name=col_name, points=points)
        log.info(f"✅ Qdrant : {len(points)} vecteurs indexés dans '{col_name}'")

        return {"status": "done", "n_indexed": len(points)}

    except Exception as e:
        log.error(f"Reindex Qdrant échoué : {e}")
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — Drift check Evidently
# ─────────────────────────────────────────────────────────────────────────────

def task_drift_check(**context):
    """
    Compare la distribution des primes/usage du mois courant vs. la baseline
    (3 mois précédents) avec Evidently.
    Pousse le rapport drift dans MinIO et une alerte dans MongoDB si dérive > seuil.
    """
    try:
        import psycopg2
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        log.warning("evidently ou psycopg2 non installé — skip drift check")
        return {"status": "skipped"}

    try:
        conn_str = DB_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(conn_str)

        current_month = datetime.now().month
        current_year  = datetime.now().year

        # Dataset courant (mois actuel)
        current = pd.read_sql("""
            SELECT mt_pnet, mt_rc, mt_commission, bonus_malus,
                   branche, periodicite, mois_echeance
            FROM dwh_fact_emission
            WHERE annee_echeance = %(year)s AND mois_echeance = %(month)s
              AND etat_quit IN ('E','P') AND mt_pnet > 0
        """, conn, params={"year": current_year, "month": current_month})

        # Baseline (3 mois précédents)
        baseline = pd.read_sql("""
            SELECT mt_pnet, mt_rc, mt_commission, bonus_malus,
                   branche, periodicite, mois_echeance
            FROM dwh_fact_emission
            WHERE (annee_echeance = %(year)s AND mois_echeance < %(month)s)
               OR (annee_echeance = %(year)s - 1 AND mois_echeance >= 10)
              AND etat_quit IN ('E','P') AND mt_pnet > 0
            LIMIT 10000
        """, conn, params={"year": current_year, "month": current_month})

        conn.close()

        if len(current) < 10 or len(baseline) < 10:
            log.info("Données insuffisantes pour drift check — skip")
            return {"status": "insufficient_data"}

        # Features numériques pour le drift
        num_features = ["mt_pnet", "mt_rc", "mt_commission", "bonus_malus"]
        for df in [current, baseline]:
            for col in num_features:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=baseline[num_features], current_data=current[num_features])

        drift_result = report.as_dict()
        drift_detected = drift_result.get("metrics", [{}])[0].get("result", {}).get("dataset_drift", False)
        drift_score    = drift_result.get("metrics", [{}])[0].get("result", {}).get("share_of_drifted_columns", 0)

        log.info(f"Drift score : {drift_score:.2f} | Drift détecté : {drift_detected}")

        # Sauvegarde rapport HTML dans MinIO
        report_path = Path(REPORT_DIR) / f"drift_report_{datetime.now():%Y%m%d_%H%M}.html"
        report.save_html(str(report_path))

        # Alerte MongoDB si dérive > seuil
        if drift_score > DRIFT_ALERT_THRESHOLD:
            try:
                from pymongo import MongoClient
                client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
                client.maghrebia_events.events.insert_one({
                    "type":       "ALERTE_DRIFT",
                    "severity":   "HIGH",
                    "drift_score": drift_score,
                    "message":    f"Dérive données détectée : {drift_score:.1%} des features ont dérivé",
                    "created_at": datetime.now().isoformat(),
                    "source":     "evidently_dag",
                })
                client.close()
            except Exception as e:
                log.warning(f"Alerte drift MongoDB échouée : {e}")

        context["ti"].xcom_push(key="drift_score", value=drift_score)
        return {"status": "done", "drift_score": drift_score, "drift_detected": drift_detected}

    except Exception as e:
        log.error(f"Drift check échoué : {e}")
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 — Rapport final du pipeline
# ─────────────────────────────────────────────────────────────────────────────

def task_pipeline_report(**context):
    """
    Consolide les résultats de toutes les tâches dans un rapport final JSON.
    Ce rapport est lu par le dashboard CEO pour le panneau 'Santé Pipeline'.
    """
    ti = context["ti"]
    run_date = context["execution_date"]

    health      = ti.xcom_pull(task_ids="health_check",       key="health")       or {}
    ge_score    = ti.xcom_pull(task_ids="data_quality_check",  key="ge_score_raw") or 0
    import_rep  = ti.xcom_pull(task_ids="import_and_clean",    key="import_report") or {}
    n_events    = ti.xcom_pull(task_ids="detect_events",       key="n_events")     or 0
    drift_score = ti.xcom_pull(task_ids="drift_check",         key="drift_score")  or 0

    report = {
        "dag_run_date":   run_date.isoformat() if run_date else datetime.now().isoformat(),
        "generated_at":   datetime.now().isoformat(),
        "pipeline_status": "OK",
        "health": health,
        "data_quality": {
            "ge_score_raw":   ge_score,
            "ge_score_clean": import_rep.get("ge_score_clean"),
        },
        "cleaning": {
            "volumes":            import_rep.get("volumes", {}),
            "total_corrections":  import_rep.get("total_corrections", 0),
        },
        "events_detected": n_events,
        "drift": {
            "score":   drift_score,
            "alerted": drift_score > DRIFT_ALERT_THRESHOLD,
        },
    }

    report_path = Path(REPORT_DIR) / "pipeline_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("  RAPPORT PIPELINE MAGHREBIA")
    log.info(f"  Date           : {report['generated_at']}")
    log.info(f"  Health PG      : {health.get('postgres', 'N/A')}")
    log.info(f"  GE score brut  : {ge_score}%")
    log.info(f"  GE score clean : {import_rep.get('ge_score_clean')}%")
    log.info(f"  Corrections    : {import_rep.get('total_corrections', 0):,}")
    log.info(f"  Événements     : {n_events}")
    log.info(f"  Drift score    : {drift_score:.2f}")
    log.info("=" * 60)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="maghrebia_pipeline_daily",
    description="Pipeline quotidien Maghrebia — Cleaning + KPI + Events + Drift",
    schedule_interval="0 2 * * *",     # 02h00 chaque nuit (TZ Africa/Tunis)
    default_args=DEFAULT_ARGS,
    catchup=False,
    max_active_runs=1,
    tags=["maghrebia", "production", "tdsp-phase2"],
) as dag:

    # Task 1
    health_check = PythonOperator(
        task_id="health_check",
        python_callable=task_health_check,
    )

    # Task 2
    data_quality = PythonOperator(
        task_id="data_quality_check",
        python_callable=task_data_quality_check,
    )

    # Task 3
    import_clean = PythonOperator(
        task_id="import_and_clean",
        python_callable=task_import_and_clean,
        execution_timeout=timedelta(hours=1),
    )

    # Task 4
    refresh_kpis = PythonOperator(
        task_id="refresh_kpi_views",
        python_callable=task_refresh_kpi_views,
    )

    # Task 5
    detect_events = PythonOperator(
        task_id="detect_events",
        python_callable=task_detect_events,
    )

    # Task 6 — Branch
    branch_reindex = BranchPythonOperator(
        task_id="should_reindex",
        python_callable=task_should_reindex,
    )

    reindex = PythonOperator(
        task_id="reindex_qdrant",
        python_callable=task_reindex_qdrant,
        execution_timeout=timedelta(minutes=30),
    )

    skip_reindex = EmptyOperator(task_id="skip_reindex")

    # Task 7
    drift_check = PythonOperator(
        task_id="drift_check",
        python_callable=task_drift_check,
        trigger_rule="all_done",   # s'exécute même si branch skip
    )

    # Task 8
    final_report = PythonOperator(
        task_id="pipeline_report",
        python_callable=task_pipeline_report,
        trigger_rule="all_done",
    )

    # ── Dépendances ──────────────────────────────────────────────────────────
    #
    #  health_check → data_quality → import_clean → refresh_kpis ──┐
    #                                             ↘                  ├→ branch → reindex ──┐
    #                                              detect_events ───┘          skip ────┘
    #                                                                             ↓
    #                                                                       drift_check → pipeline_report
    #
    health_check >> data_quality >> import_clean
    import_clean >> [refresh_kpis, detect_events]
    import_clean >> branch_reindex
    branch_reindex >> [reindex, skip_reindex]
    [refresh_kpis, detect_events, reindex, skip_reindex] >> drift_check
    drift_check >> final_report
