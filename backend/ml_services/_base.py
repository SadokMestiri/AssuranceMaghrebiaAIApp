"""
ml_services/_base.py
────────────────────
Shared utilities used by every ML service:
  - DB loading helpers (SQLAlchemy → pandas)
  - Feature helpers (RFM, aggregation)
  - Model artifact cache
  - Lightweight model trainer / loader
"""
from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger("maghrebia.ml_services")

# ── Paths ──────────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODELS_DIR.mkdir(exist_ok=True)

# ── DB helpers ─────────────────────────────────────────────────────────────

def _get_engine():
    """Lazy-import engine to avoid circular at import time."""
    from db import engine as db_engine          # noqa: PLC0415
    return db_engine


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SQL string and return a DataFrame. Returns empty DF on error."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as exc:
        logger.warning("query_df failed: %s", exc)
        return pd.DataFrame()


# ── CSV fallback (dev / notebook mode) ────────────────────────────────────
_CSV_DIR = Path(__file__).resolve().parent.parent  # project root

def _load_csv(name: str) -> pd.DataFrame:
    """Try to load a DWH CSV from the project root for local development."""
    path = _CSV_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def load_table(sql: str, csv_fallback: str | None = None,
               params: dict | None = None) -> pd.DataFrame:
    """
    Try DB first; fall back to CSV if DB returns nothing or raises.
    csv_fallback: stem name e.g. 'DWH_FACT_EMISSION'
    """
    df = query_df(sql, params)
    if df.empty and csv_fallback:
        df = _load_csv(csv_fallback)
    return df


# ── Artifact helpers ────────────────────────────────────────────────────────

def load_artifact(name: str) -> dict[str, Any] | None:
    """Load a pickle artifact from the models directory. Returns None if missing."""
    path = _MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Cannot load artifact %s: %s", name, exc)
        return None


def save_artifact(name: str, obj: Any) -> None:
    path = _MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Artifact saved: %s", path)


# ── Numeric helpers ─────────────────────────────────────────────────────────

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def clip_ratio(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def normalize_0_1000(value: float, p1: float, p99: float) -> float:
    """Map a raw value to [0, 1000] using percentile anchors."""
    denom = p99 - p1
    if denom == 0:
        return 500.0
    return float(np.clip((value - p1) / denom * 1000, 0, 1000))


# ── Common DB queries (reused across services) ─────────────────────────────

_EMISSION_SQL = """
SELECT e.num_quittance, e.id_police, e.id_agent, e.id_branche, e.branche,
       e.annee_echeance, e.mois_echeance, e.etat_quit,
       e.mt_pnet, e.mt_rc, e.mt_commission, e.bonus_malus,
       e.date_emission, e.date_effet, e.date_echeance, e.periodicite
FROM   dwh_fact_emission e
"""

_SINISTRE_SQL = """
SELECT s.id_police, s.id_agent, s.id_client, s.id_branche, s.branche,
       s.num_sinistre, s.date_survenance, s.annee_survenance,
       s.mois_survenance, s.nature_sinistre, s.responsabilite,
       s.mt_evaluation, s.mt_paye, s.etat_sinistre
FROM   dwh_fact_sinistre s
"""

_IMPAYE_SQL = """
SELECT i.id_police, i.id_agent, i.id_branche, i.branche,
       i.num_quittance, i.mt_pnn, i.date_emission
FROM   dwh_fact_impaye i
"""

_POLICE_SQL = """
SELECT p.id_police, p.id_client, p.id_agent, p.id_branche,
       p.branche, p.code_produit, p.lib_produit, p.type_police,
       p.periodicite, p.duree, p.situation,
       p.date_effet, p.date_echeance, p.bonus_malus
FROM   dim_police p
"""

_CLIENT_SQL = """
SELECT c.id_client, c.type_personne, c.sexe,
       c.date_naissance, c.ville, c.natp
FROM   dim_client c
"""

_VEHICULE_SQL = """
SELECT v.id_vehicule, v.puissance, v.genre_vehicule, v.type_vehicule,
       v.marque, v.nb_place, v.charge_utile, v.poids_total,
       v.valeur_vehicule, v.valeur_neuf, v.date_mise_circulation
FROM   dim_vehicule v
"""


def load_emission() -> pd.DataFrame:
    return load_table(_EMISSION_SQL, "DWH_FACT_EMISSION")


def load_sinistre() -> pd.DataFrame:
    return load_table(_SINISTRE_SQL, "DWH_FACT_SINISTRE")


def load_impaye() -> pd.DataFrame:
    return load_table(_IMPAYE_SQL, "DWH_FACT_IMPAYE")


def load_police() -> pd.DataFrame:
    return load_table(_POLICE_SQL, "DIM_POLICE")


def load_client() -> pd.DataFrame:
    return load_table(_CLIENT_SQL, "DIM_CLIENT")


def load_vehicule() -> pd.DataFrame:
    return load_table(_VEHICULE_SQL, "DIM_VEHICULE")