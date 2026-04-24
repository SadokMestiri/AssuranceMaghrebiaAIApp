"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Maghrebia Assurance — feature_engineering.py                                ║
║  TDSP Phase 3 — Modélisation des Données & Ingénierie des Variables (ML)     ║
║                                                                              ║
║  Ce script génère les tables de Features (Variables cibles) pour le ML       ║
║  directement depuis les données structurées dans PostgreSQL (Phase 2).       ║
║                                                                              ║
║  Construit 2 tables principales :                                            ║
║  1. ml_features_churn : Niveau Police (Taux sinistralité, Ancienneté, etc.)  ║
║  2. ml_features_client: Niveau Client (LTV, Fréquence impayés, etc.)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Configuration Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configuration DB
_db_url = os.environ.get("DATABASE_URL")
if _db_url:
    DB_URL = _db_url
else:
    DB_URL = (
        "postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'maghrebia')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'maghrebia')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'maghrebia')}"
    )

engine = create_engine(DB_URL)

def generate_features_churn():
    """
    Construit le jeu de données pour le Modèle Churn (Niveau Police / Contrat)
    """
    logging.info("⏳ Étape 1/2 : Construction des features CHURN (Niveau Police)...")
    
    query = """
    WITH stats_emission AS (
        SELECT id_police, 
               COUNT(num_quittance) as nb_quittances,
               SUM(mt_ptt) as total_prime_ptt,
               SUM(mt_commission) as total_commission
        FROM dwh_fact_emission
        GROUP BY id_police
    ),
    stats_sinistre AS (
        SELECT id_police,
               COUNT(id_sinistre) as nb_sinistres,
               SUM(mt_paye) as total_rembourse
        FROM dwh_fact_sinistre
        GROUP BY id_police
    ),
    stats_impaye AS (
        SELECT id_police as id_police,
               COUNT(id) as nb_impayes,
               SUM(mt_acp) as total_impaye_acp
        FROM dwh_fact_impaye
        GROUP BY id_police
    ),
    stats_annulation AS (
        SELECT id_police, 
               SUM(mt_ptt_ann) as total_prime_annulee
        FROM dwh_fact_annulation
        GROUP BY id_police
    )
    SELECT 
        p.id_police,
        p.id_client,
        p.branche,
        p.situation,
        p.bonus_malus,
        p.periodicite,
        p.date_effet,
        p.date_echeance,
        c.date_naissance,
        c.sexe,
        c.code_postal,
        COALESCE(e.nb_quittances, 0) as nb_quittances,
        COALESCE(e.total_prime_ptt, 0) as total_prime_ptt,
        COALESCE(e.total_commission, 0) as total_commission,
        COALESCE(s.nb_sinistres, 0) as nb_sinistres,
        COALESCE(s.total_rembourse, 0) as total_rembourse,
        COALESCE(i.nb_impayes, 0) as nb_impayes,
        COALESCE(i.total_impaye_acp, 0) as total_impaye_acp,
        COALESCE(a.total_prime_annulee, 0) as total_prime_annulee
    FROM dim_police p
    JOIN dim_client c ON p.id_client = c.id_client
    LEFT JOIN stats_emission e ON p.id_police = e.id_police
    LEFT JOIN stats_sinistre s ON p.id_police = s.id_police
    LEFT JOIN stats_impaye i ON p.id_police = i.id_police
    LEFT JOIN stats_annulation a ON p.id_police = a.id_police
    WHERE p.id_police != 0 AND p.id_client != 0
    """
    
    df = pd.read_sql(query, engine)
    
    # === FEATURE ENGINEERING PANDAS ===
    
    # 1. Target Variable (Pour le modèle de prédiction Churn)
    df['target_churn'] = (df['situation'] == 'R').astype(int)
    
    # 2. Gestion Temporelle (Référence : Fin des données, soit 31 Déc 2024 env.)
    ref_date = pd.to_datetime('2024-12-31')
    
    df['date_effet'] = pd.to_datetime(df['date_effet'], errors='coerce')
    df['date_naissance'] = pd.to_datetime(df['date_naissance'], errors='coerce')
    
    # Ancienneté Police en années
    df['anciennete_police_annees'] = (ref_date - df['date_effet']).dt.days / 365.25
    df['anciennete_police_annees'] = df['anciennete_police_annees'].clip(lower=0).fillna(0)
    
    # Age Client au moment de l'analyse
    df['age_client'] = (ref_date - df['date_naissance']).dt.days / 365.25
    df.loc[df['age_client'] > 100, 'age_client'] = np.nan # Filtrer les âges aberrants (ex: nés en 1900)
    df['age_client'] = df['age_client'].fillna(df['age_client'].median())
    
    # 3. KPI & Variables calculées
    # Taux de Sinistralité (S/P - Remboursement / Prime Nette ou PTT)
    df['taux_sinistralite'] = df['total_rembourse'] / df['total_prime_ptt'].replace(0, np.nan)
    df['taux_sinistralite'] = df['taux_sinistralite'].fillna(0).clip(upper=5.0) # Plafonnée à 500% pour éviter les outliers
    
    # Fréquence des sinistres (nb sinistres par an)
    df['frequence_sinistre_annuelle'] = df['nb_sinistres'] / df['anciennete_police_annees'].replace(0, 0.5)
    
    # Ratio d'Impayé (Volume impayé / Volume Facturé)
    df['ratio_impaye'] = df['total_impaye_acp'] / df['total_prime_ptt'].replace(0, np.nan)
    df['ratio_impaye'] = df['ratio_impaye'].fillna(0).clip(upper=1.0)
    
    # Taux de Commission (Commission / Prime TTC)
    df['taux_commission'] = df['total_commission'] / df['total_prime_ptt'].replace(0, np.nan)
    df['taux_commission'] = df['taux_commission'].fillna(0).clip(lower=0, upper=1)
    
    # Nettoyage des colonnes inutiles pour l'entraînement ML
    cols_to_drop = ['situation', 'date_effet', 'date_echeance', 'date_naissance']
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Sauvegarde dans PostgreSQL et affichage Log
    logging.info(f"✅ Features CHURN créées: {len(df):,} lignes x {len(df.columns)} colonnes.")
    
    df.to_sql('ml_features_churn', engine, if_exists='replace', index=False)
    logging.info("💾 Table 'ml_features_churn' sauvegardée avec succès.")


def generate_features_client():
    """
    Construit le jeu de données consolidées niveau Client (Segmentation / LTV)
    """
    logging.info("⏳ Étape 2/2 : Construction des features CLIENT (Niveau Client)...")
    
    query = """
    WITH police_agg AS (
        SELECT 
            id_client,
            COUNT(id_police) as nb_polices_totales,
            SUM(CASE WHEN situation = 'R' THEN 1 ELSE 0 END) as nb_polices_resiliees,
            SUM(CASE WHEN situation = 'V' THEN 1 ELSE 0 END) as nb_polices_en_vigueur,
            MIN(date_effet) as date_premiere_police
        FROM dim_police
        GROUP BY id_client
    ),
    fact_agg AS (
        SELECT
            p.id_client,
            SUM(e.mt_ptt) as total_prime_facturee,
            SUM(e.mt_commission) as total_commission_apportee
        FROM dwh_fact_emission e
        JOIN dim_police p ON e.id_police = p.id_police
        GROUP BY p.id_client
    ),
    impaye_agg AS (
        SELECT
            p.id_client,
            COUNT(i.id) as nb_total_impayes,
            SUM(i.mt_acp) as total_encours_impaye
        FROM dwh_fact_impaye i
        JOIN dim_police p ON i.id_police = p.id_police
        GROUP BY p.id_client
    ),
    sinistre_agg AS (
        SELECT
            p.id_client,
            COUNT(s.id_sinistre) as nb_total_sinistres,
            SUM(s.mt_paye) as cout_total_sinistres
        FROM dwh_fact_sinistre s
        JOIN dim_police p ON s.id_police = p.id_police
        GROUP BY p.id_client
    )
    SELECT 
        c.id_client,
        c.type_personne,
        c.sexe,
        c.age_unknown,
        c.code_postal,
        COALESCE(p.nb_polices_totales, 0) as nb_polices_totales,
        COALESCE(p.nb_polices_resiliees, 0) as nb_polices_resiliees,
        COALESCE(p.nb_polices_en_vigueur, 0) as nb_polices_en_vigueur,
        p.date_premiere_police,
        COALESCE(f.total_prime_facturee, 0) as total_prime_facturee,
        COALESCE(f.total_commission_apportee, 0) as total_commission_apportee,
        COALESCE(i.nb_total_impayes, 0) as nb_total_impayes,
        COALESCE(i.total_encours_impaye, 0) as total_encours_impaye,
        COALESCE(s.nb_total_sinistres, 0) as nb_total_sinistres,
        COALESCE(s.cout_total_sinistres, 0) as cout_total_sinistres
    FROM dim_client c
    LEFT JOIN police_agg p ON c.id_client = p.id_client
    LEFT JOIN fact_agg f ON c.id_client = f.id_client
    LEFT JOIN impaye_agg i ON c.id_client = i.id_client
    LEFT JOIN sinistre_agg s ON c.id_client = s.id_client
    WHERE c.id_client != 0
    """
    
    df = pd.read_sql(query, engine)
    
    # === FEATURE ENGINEERING PANDAS ===
    ref_date = pd.to_datetime('2024-12-31')
    
    # Ancienneté Client Profile
    df['date_premiere_police'] = pd.to_datetime(df['date_premiere_police'], errors='coerce')
    df['anciennete_client_annees'] = (ref_date - df['date_premiere_police']).dt.days / 365.25
    df['anciennete_client_annees'] = df['anciennete_client_annees'].clip(lower=0).fillna(0)
    
    # Valeur Client (Customer Lifetime Value approchée)
    df['ltv_estimee'] = df['total_prime_facturee'] - df['total_commission_apportee'] - df['cout_total_sinistres']
    
    # Profil Risque Client
    df['taux_polices_resiliees'] = df['nb_polices_resiliees'] / df['nb_polices_totales'].replace(0, 1)
    df['frequence_sinistre_globale'] = df['nb_total_sinistres'] / df['nb_polices_totales'].replace(0, 1)
    
    cols_to_drop = ['date_premiere_police']
    df.drop(columns=cols_to_drop, inplace=True)
    
    logging.info(f"✅ Features CLIENT créées: {len(df):,} lignes x {len(df.columns)} colonnes.")
    
    df.to_sql('ml_features_client', engine, if_exists='replace', index=False)
    logging.info("💾 Table 'ml_features_client' sauvegardée avec succès.")


if __name__ == "__main__":
    logging.info("=====================================================")
    logging.info("  MAGHREBIA ASSURANCE — FEATURE ENGINEERING (PHASE 3)")
    logging.info("=====================================================")
    try:
        generate_features_churn()
        generate_features_client()
        logging.info("=====================================================")
        logging.info("🎉 Processus complet : Tables ML prêtes pour l'entraînement !")
    except Exception as e:
        logging.error(f"❌ Erreur lors de la génération des features : {str(e)}")
