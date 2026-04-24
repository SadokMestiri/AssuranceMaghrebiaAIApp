-- ══════════════════════════════════════════════════════════════════════════
--  Maghrebia Assurance — PostgreSQL Schema
--  TDSP Phase 2 — Data Warehouse Star Schema
--  Tables : 4 DIM + 4 FACT + 1 airflow DB
-- ══════════════════════════════════════════════════════════════════════════

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Ensure Airflow metadata DB exists on first initialization.
SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

SELECT 'CREATE DATABASE mlflow_tracking'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_tracking')\gexec

-- ── DIMENSIONS ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_agent (
    id_agent          INTEGER PRIMARY KEY,
    code_agent        VARCHAR(20),
    nom_agent         VARCHAR(100),
    prenom_agent      VARCHAR(100),
    tel_agent         VARCHAR(30),
    email_agent       VARCHAR(100),
    groupe_agent      VARCHAR(50),
    localite_agent    VARCHAR(100),
    latitude_agent    NUMERIC(10, 6),
    longitude_agent   NUMERIC(10, 6),
    etat_agent        VARCHAR(5),
    type_agent        VARCHAR(10),
    -- Audit
    loaded_at         TIMESTAMP DEFAULT NOW(),
    updated_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_client (
    id_client         INTEGER PRIMARY KEY,
    code_client       VARCHAR(20),
    cin_mf            VARCHAR(30),
    type_personne     VARCHAR(5),    -- P = Physique, M = Morale
    nom               VARCHAR(200),
    prenom            VARCHAR(200),
    adresse           VARCHAR(300),
    code_postal       INTEGER,
    ville             VARCHAR(100),
    sexe              VARCHAR(5),
    date_naissance    DATE,
    natp              VARCHAR(10),
    email             VARCHAR(200),
    loaded_at         TIMESTAMP DEFAULT NOW(),
    updated_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_police (
    id_police         INTEGER PRIMARY KEY,
    num_police        VARCHAR(50),
    id_branche        INTEGER,
    code_produit      INTEGER,
    lib_produit       VARCHAR(100),
    branche           VARCHAR(20),   -- AUTO / IRDS / SANTE
    id_agent          INTEGER REFERENCES dim_agent(id_agent),
    id_client         INTEGER REFERENCES dim_client(id_client),
    type_police       VARCHAR(20),   -- individuel / flotte
    duree             VARCHAR(20),
    periodicite       VARCHAR(20),
    date_effet        DATE,
    date_echeance     DATE,
    polrp             INTEGER,       -- Police mère (flottes)
    situation         VARCHAR(5),    -- V/R/T/S/A
    bonus_malus       NUMERIC(5, 2),
    loaded_at         TIMESTAMP DEFAULT NOW(),
    updated_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_vehicule (
    id_vehicule       INTEGER PRIMARY KEY,
    id_police         INTEGER REFERENCES dim_police(id_police),
    num_serie         VARCHAR(50),
    marque            VARCHAR(50),
    puissance         INTEGER,
    immatriculation   VARCHAR(30),
    genre_vehicule    VARCHAR(50),
    type_vehicule     VARCHAR(50),
    nb_place          INTEGER,
    date_mec          DATE,          -- Date mise en circulation
    valeur_a_neuf     NUMERIC(15, 2),
    valeur_actuelle   NUMERIC(15, 2),
    charge_utile      NUMERIC(10, 2),
    poids_total       NUMERIC(10, 2),
    code_usage        INTEGER,
    loaded_at         TIMESTAMP DEFAULT NOW()
);

-- ── FACTS ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dwh_fact_emission (
    num_quittance     VARCHAR(50) PRIMARY KEY,
    id_branche        INTEGER,
    annee_echeance    INTEGER,
    mois_echeance     INTEGER,
    id_police         INTEGER REFERENCES dim_police(id_police),
    id_agent          INTEGER REFERENCES dim_agent(id_agent),
    code_produit      INTEGER,
    branche           VARCHAR(20),
    num_avn           INTEGER,
    sit               INTEGER,
    etat_quit         VARCHAR(5),    -- E / P / A
    date_effet        DATE,
    date_echeance     DATE,
    date_emission     DATE,
    periodicite       VARCHAR(20),
    bonus_malus       NUMERIC(5, 2),
    -- Garanties individuelles
    mt_rc             NUMERIC(15, 3),
    mt_dom            NUMERIC(15, 3),
    mt_inc            NUMERIC(15, 3),
    mt_vol            NUMERIC(15, 3),
    mt_bgl            NUMERIC(15, 3),
    mt_domcoll        NUMERIC(15, 3),
    mt_tel            NUMERIC(15, 3),
    mt_cas            NUMERIC(15, 3),
    mt_pta            NUMERIC(15, 3),
    mt_ass            NUMERIC(15, 3),
    mt_immob          NUMERIC(15, 3),
    -- Totaux
    mt_pnet           NUMERIC(15, 3),   -- Prime nette = Σ garanties
    mt_fga            NUMERIC(15, 3),   -- FGA = MT_RC × 0.25
    mt_timbre         NUMERIC(15, 3),
    mt_taxe           NUMERIC(15, 3),
    mt_ptt            NUMERIC(15, 3),   -- MT_PTT = PNET+FGA+TIMBRE+TAXE
    mt_commission     NUMERIC(15, 3),
    -- Data quality flags (Phase 2)
    dq_etat_quit_valid    BOOLEAN,
    dq_ptt_formula_valid  BOOLEAN,
    dq_year_valid         BOOLEAN,
    dq_pnet_positive      BOOLEAN,
    loaded_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dwh_fact_emission_detail (
    id                SERIAL PRIMARY KEY,
    id_branche        INTEGER,
    num_quittance     VARCHAR(50) REFERENCES dwh_fact_emission(num_quittance),
    annee_echeance    INTEGER,
    id_garantie       INTEGER,
    code_garantie     VARCHAR(20),
    id_police         INTEGER REFERENCES dim_police(id_police),
    annee_emission    INTEGER,
    mt_prime          NUMERIC(15, 3),
    mt_frais          NUMERIC(15, 3),
    loaded_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dwh_fact_annulation (
    num_quittance         VARCHAR(50) PRIMARY KEY,
    id_branche            INTEGER,
    branche               VARCHAR(20),
    id_police             INTEGER REFERENCES dim_police(id_police),
    id_agent              INTEGER REFERENCES dim_agent(id_agent),
    annee_emission        INTEGER,
    mois_emission         INTEGER,
    date_emission         DATE,
    date_annulation       DATE,
    mois_annulation       INTEGER,
    annee_annulation      INTEGER,
    nature_annulation     VARCHAR(50),
    situation_annulation  VARCHAR(20),
    mt_rc_ann             NUMERIC(15, 3),
    mt_dom_ann            NUMERIC(15, 3),
    mt_inc_ann            NUMERIC(15, 3),
    mt_vol_ann            NUMERIC(15, 3),
    mt_pnn_ann            NUMERIC(15, 3),
    mt_fga_ann            NUMERIC(15, 3),
    mt_timbre             NUMERIC(15, 3),
    mt_ptt_ann            NUMERIC(15, 3),
    mt_commission_ann     NUMERIC(15, 3),
    loaded_at             TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dwh_fact_impaye (
    id                SERIAL PRIMARY KEY,
    annee_echeance    INTEGER,
    mois_echeance     INTEGER,
    num_quittance     VARCHAR(50),
    id_branche        INTEGER,
    branche           VARCHAR(20),
    id_police         INTEGER REFERENCES dim_police(id_police),
    id_agent          INTEGER REFERENCES dim_agent(id_agent),
    sit               INTEGER,
    num_avn           INTEGER,
    mt_pnn            NUMERIC(15, 3),
    mt_taxe           NUMERIC(15, 3),
    mt_ptt            NUMERIC(15, 3),
    mt_commission     NUMERIC(15, 3),
    mt_acp            NUMERIC(15, 3),
    date_emission     DATE,
    date_situation    DATE,
    loaded_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dwh_fact_sinistre (
    id_sinistre       SERIAL PRIMARY KEY,
    num_sinistre      VARCHAR(50) UNIQUE,
    id_branche        INTEGER,
    branche           VARCHAR(20),
    id_police         INTEGER REFERENCES dim_police(id_police),
    id_agent          INTEGER REFERENCES dim_agent(id_agent),
    id_client         INTEGER REFERENCES dim_client(id_client),
    id_vehicule       INTEGER REFERENCES dim_vehicule(id_vehicule), -- Peut être NULL pour Santé / IRDS
    date_survenance   DATE,
    date_declaration  DATE,
    annee_survenance  INTEGER,
    mois_survenance   INTEGER,
    nature_sinistre   VARCHAR(50),   -- Matériel, Corporel, Vol, Maladie...
    responsabilite    INTEGER,       -- 0, 50, 100 (pourcentage des torts)
    mt_evaluation     NUMERIC(15, 3),-- Coût estimé par l'expert
    mt_paye           NUMERIC(15, 3),-- Ce qui a été réellement remboursé au client
    etat_sinistre     VARCHAR(20),   -- Ouvert, Clos, Refusé
    loaded_at         TIMESTAMP DEFAULT NOW()
);

-- ── MLflow schema ───────────────────────────────────────────────────────────
-- MLflow creates its own tables, but we need the schema to exist
-- (handled by mlflow server --backend-store-uri)

-- ── KPI materialized views ──────────────────────────────────────────────────

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_kpi_production_annuelle AS
SELECT
    annee_echeance                          AS annee,
    branche,
    COUNT(*)                                AS nb_quittances,
    SUM(mt_pnet)                            AS total_pnet,
    SUM(mt_ptt)                             AS total_ptt,
    SUM(mt_commission)                      AS total_commission,
    SUM(mt_fga)                             AS total_fga,
    AVG(mt_pnet)                            AS avg_pnet,
    COUNT(*) FILTER (WHERE etat_quit = 'A') AS nb_annulations,
    COUNT(*) FILTER (WHERE etat_quit = 'E') AS nb_emis,
    COUNT(*) FILTER (WHERE etat_quit = 'P') AS nb_payes
FROM dwh_fact_emission
WHERE annee_echeance BETWEEN 2019 AND 2025
  AND etat_quit IN ('E','P','A')
GROUP BY annee_echeance, branche
ORDER BY annee_echeance, branche;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_kpi_agent_performance AS
SELECT
    e.id_agent,
    a.code_agent,
    a.nom_agent,
    a.groupe_agent,
    a.localite_agent,
    a.latitude_agent,
    a.longitude_agent,
    COUNT(*)           AS nb_quittances,
    SUM(e.mt_pnet)     AS total_pnet,
    SUM(e.mt_commission) AS total_commission,
    AVG(e.mt_pnet)     AS avg_pnet,
    COUNT(*) FILTER (WHERE e.etat_quit = 'A') AS nb_annulations,
    ROUND(100.0 * COUNT(*) FILTER (WHERE e.etat_quit = 'A') / NULLIF(COUNT(*),0), 2) AS taux_annulation_pct
FROM dwh_fact_emission e
JOIN dim_agent a ON e.id_agent = a.id_agent
WHERE e.annee_echeance BETWEEN 2019 AND 2025
  AND e.etat_quit IN ('E','P','A')
GROUP BY e.id_agent, a.code_agent, a.nom_agent, a.groupe_agent,
         a.localite_agent, a.latitude_agent, a.longitude_agent
ORDER BY total_pnet DESC;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_kpi_portefeuille AS
SELECT
    p.branche,
    p.situation,
    COUNT(*)           AS nb_polices,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY p.branche), 2) AS pct_branche
FROM dim_police p
GROUP BY p.branche, p.situation
ORDER BY p.branche, p.situation;

-- ── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_emission_branche     ON dwh_fact_emission(branche);
CREATE INDEX IF NOT EXISTS idx_emission_annee       ON dwh_fact_emission(annee_echeance);
CREATE INDEX IF NOT EXISTS idx_emission_agent       ON dwh_fact_emission(id_agent);
CREATE INDEX IF NOT EXISTS idx_emission_police      ON dwh_fact_emission(id_police);
CREATE INDEX IF NOT EXISTS idx_emission_etat        ON dwh_fact_emission(etat_quit);
CREATE INDEX IF NOT EXISTS idx_police_situation     ON dim_police(situation);
CREATE INDEX IF NOT EXISTS idx_police_branche       ON dim_police(branche);
CREATE INDEX IF NOT EXISTS idx_client_ville         ON dim_client(ville);
CREATE INDEX IF NOT EXISTS idx_agent_groupe         ON dim_agent(groupe_agent);
CREATE INDEX IF NOT EXISTS idx_impaye_branche       ON dwh_fact_impaye(branche);

-- GIN index for text search
CREATE INDEX IF NOT EXISTS idx_client_nom_gin ON dim_client USING gin(nom gin_trgm_ops);

-- ── Data quality log table ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dq_run_log (
    id              SERIAL PRIMARY KEY,
    run_date        TIMESTAMP DEFAULT NOW(),
    table_name      VARCHAR(50),
    suite_name      VARCHAR(100),
    total_rows      INTEGER,
    failed_rows     INTEGER,
    success_pct     NUMERIC(5, 2),
    ge_report_path  VARCHAR(500),
    status          VARCHAR(20)   -- PASSED / FAILED / WARNING
);