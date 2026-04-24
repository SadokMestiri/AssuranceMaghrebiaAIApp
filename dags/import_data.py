"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Maghrebia Assurance — import_data.py                                       ║
║  TDSP Phase 2 — Data Preprocessing, Cleaning & DWH Loading                 ║
║                                                                              ║
║  Pipeline en 5 étapes :                                                      ║
║   1. EXTRACT  — lecture CSV bruts                                            ║
║   2. CLEAN    — 34 règles de nettoyage documentées                          ║
║   3. VALIDATE — Great Expectations (24 règles métier)                       ║
║   4. LOAD     — chargement PostgreSQL (UPSERT idempotent)                   ║
║   5. REPORT   — rapport qualité JSON + log console                          ║
║                                                                              ║
║  Usage :                                                                     ║
║    python import_data.py                          # mode standard           ║
║    python import_data.py --dry-run                # sans écriture DB        ║
║    python import_data.py --table FACT_EMISSION    # table spécifique        ║
║    python import_data.py --skip-validate          # sans GE                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, logging, argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(os.environ.get("DATA_DIR",    "maghrebia/data/raw"))
REPORT_DIR  = Path(os.environ.get("REPORT_DIR",  "maghrebia/reports/data_quality"))
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
CLEAN_DIR   = Path(os.environ.get("CLEAN_DIR", str(DATA_DIR.parent / "clean")))

UNKNOWN_ID = 0
DEFAULT_BRANCHE = "AUTO"
DEFAULT_YEAR = 2024
VALID_BRANCHES = {"AUTO", "IRDS", "SANTE"}

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
class SafeConsoleFormatter(logging.Formatter):
    """Downgrade unsupported console characters instead of crashing on cp1252."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        return msg.encode(encoding, errors="replace").decode(encoding, errors="replace")


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(SafeConsoleFormatter(_LOG_FORMAT))

_file_handler = logging.FileHandler(
    REPORT_DIR / f"import_{datetime.now():%Y%m%d_%H%M}.log",
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_console_handler, _file_handler],
    force=True,
)
log = logging.getLogger("maghrebia.import")


# ─────────────────────────────────────────────────────────────────────────────
# STATS TRACKER  — compte les corrections appliquées
# ─────────────────────────────────────────────────────────────────────────────

class CleaningStats:
    def __init__(self, table: str):
        self.table = table
        self.n_input = 0
        self.n_output = 0
        self.rules: dict[str, int] = {}

    def record(self, rule: str, n: int):
        if n > 0:
            self.rules[rule] = self.rules.get(rule, 0) + n
            log.info(f"  [{self.table}] {rule}: {n:,} corrections")

    def summary(self) -> dict:
        n_dropped = max(self.n_input - self.n_output, 0)
        n_added = max(self.n_output - self.n_input, 0)
        return {
            "table":      self.table,
            "n_input":    self.n_input,
            "n_output":   self.n_output,
            "n_dropped":  n_dropped,
            "n_added":    n_added,
            "corrections": self.rules,
            "total_corrections": sum(self.rules.values()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(name: str) -> pd.DataFrame:
    """Cherche le CSV par nom partiel dans DATA_DIR."""
    for f in DATA_DIR.iterdir():
        if name.lower() in f.name.lower() and f.suffix == ".csv":
            df = pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
            log.info(f"  Chargé : {f.name} → {len(df):,} lignes × {len(df.columns)} cols")
            return df
    raise FileNotFoundError(f"CSV '{name}' introuvable dans {DATA_DIR}")


def parse_dates_multi_format(series: pd.Series) -> pd.Series:
    """
    Convertit une colonne de dates avec formats mixtes :
      YYYY-MM-DD  (nominal)
      YYYY/MM/DD  (dirty)
      DD-MM-YYYY  (dirty)
    Retourne pd.NaT pour les valeurs non parsables.
    """
    def parse_one(v):
        if pd.isna(v):
            return pd.NaT
        s = str(v).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y%m%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except ValueError:
                continue
        return pd.NaT

    return series.apply(parse_one)


def clean_text(series: pd.Series) -> pd.Series:
    """Strip + upper sur les codes catégoriels."""
    return series.astype(str).str.strip().str.upper().replace("NAN", np.nan)


def winsorize(series: pd.Series, lower_q=0.001, upper_q=0.999) -> pd.Series:
    """Plafonne les outliers extrêmes (< 0.1% et > 99.9% quantiles)."""
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def mode_or_default(series: pd.Series, default):
    s = series.dropna()
    if s.empty:
        return default
    m = s.mode(dropna=True)
    if not m.empty:
        return m.iloc[0]
    return s.iloc[0]


def ensure_unknown_dim_agent(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    if (pd.to_numeric(df["ID_AGENT"], errors="coerce") == UNKNOWN_ID).any():
        return df

    row = {
        "ID_AGENT": UNKNOWN_ID,
        "CODE_AGENT": "AG0000",
        "NOM_AGENT": "INCONNU",
        "PRENOM_AGENT": "INCONNU",
        "TEL_AGENT": np.nan,
        "EMAIL_AGENT": "agent.inconnu@maghrebia.com.tn",
        "GROUPE_AGENT": "Inconnu",
        "LOCALITE_AGENT": "Inconnu",
        "LATITUDE_AGENT": np.nan,
        "LONGITUDE_AGENT": np.nan,
        "ETAT_AGENT": "A",
        "TYPE_AGENT": "AG",
    }
    stats.record("A7 - ligne UNKNOWN agent (ID=0) ajoutée", 1)
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def ensure_unknown_dim_client(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    if (pd.to_numeric(df["ID_CLIENT"], errors="coerce") == UNKNOWN_ID).any():
        return df

    row = {
        "ID_CLIENT": UNKNOWN_ID,
        "CODE_CLIENT": "CL0000",
        "CIN_MF": np.nan,
        "TYPE_PERSONNE": "P",
        "NOM": "INCONNU",
        "PRENOM": "INCONNU",
        "ADRESSE": np.nan,
        "CODE_POSTAL": pd.NA,
        "VILLE": "Inconnue",
        "SEXE": np.nan,
        "DATE_NAISSANCE": pd.Timestamp("1900-01-01"),
        "AGE_UNKNOWN": 1,
        "NATP": np.nan,
        "EMAIL": np.nan,
    }
    stats.record("C9 - ligne UNKNOWN client (ID=0) ajoutée", 1)
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def ensure_unknown_dim_police(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    if (pd.to_numeric(df["ID_POLICE"], errors="coerce") == UNKNOWN_ID).any():
        return df

    default_branche = mode_or_default(
        df.loc[df["BRANCHE"].isin(VALID_BRANCHES), "BRANCHE"],
        DEFAULT_BRANCHE,
    )
    row = {
        "ID_POLICE": UNKNOWN_ID,
        "NUM_POLICE": "UNK0000000",
        "ID_BRANCHE": 0,
        "CODE_PRODUIT": 0,
        "LIB_PRODUIT": "Produit Inconnu",
        "BRANCHE": default_branche,
        "ID_AGENT": UNKNOWN_ID,
        "ID_CLIENT": UNKNOWN_ID,
        "TYPE_POLICE": "individuel",
        "DUREE": "R",
        "PERIODICITE": "A",
        "DATE_EFFET": pd.NaT,
        "DATE_ECHEANCE": pd.NaT,
        "POLRP": pd.NA,
        "SITUATION": "V",
        "BONUS_MALUS": 1.0,
    }
    stats.record("P11 - ligne UNKNOWN police (ID=0) ajoutée", 1)
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def find_schema_sql_file() -> Path | None:
    """Retourne le chemin du SQL d'initialisation si disponible."""
    script_root = Path(__file__).resolve().parents[1]
    env_path = os.environ.get("SCHEMA_SQL_PATH")

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        script_root / "sql" / "init_schema.sql",
        Path("maghrebia/sql/init_schema.sql"),
        Path("sql/init_schema.sql"),
    ])

    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists() and rp.is_file():
            return rp
    return None


def ensure_database_schema(engine) -> None:
    """Crée le schéma DWH si la base ne contient pas encore les tables cibles."""
    with engine.connect() as conn:
        dim_agent_exists = conn.execute(
            text("SELECT to_regclass('public.dim_agent')")
        ).scalar()

    if dim_agent_exists:
        return

    schema_sql = find_schema_sql_file()
    if schema_sql is None:
        raise FileNotFoundError(
            "Fichier init_schema.sql introuvable. Définir SCHEMA_SQL_PATH ou "
            "placer le fichier dans maghrebia/sql/init_schema.sql."
        )

    log.warning(
        f"  [BOOTSTRAP] Table dim_agent absente, initialisation du schéma via {schema_sql}"
    )
    sql_text = schema_sql.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql_text))
    log.info("  [BOOTSTRAP] Schéma PostgreSQL initialisé")


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING FUNCTIONS — une par table, documentées règle par règle
# ─────────────────────────────────────────────────────────────────────────────

def clean_dim_agent(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      A1 — Suppression doublons ID_AGENT
      A2 — CODE_AGENT : strip + upper
      A3 — NOM_AGENT / PRENOM_AGENT : strip + title case
      A4 — EMAIL_AGENT null → placeholder standard
      A5 — ETAT_AGENT valeurs invalides → 'A' (Actif)
    A6 — GPS : lat/lon hors Tunisie → NaN
    A7 — Ajout d'une ligne UNKNOWN (ID_AGENT=0) pour imputation FK
    """
    stats = CleaningStats("DIM_AGENT")
    df = df_raw.copy()
    stats.n_input = len(df)

    # A1
    n = df.duplicated("ID_AGENT").sum()
    stats.record("A1 - doublons ID_AGENT", n)
    df = df.drop_duplicates("ID_AGENT")

    # A2
    df["CODE_AGENT"] = df["CODE_AGENT"].astype(str).str.strip().str.upper()

    # A3
    for col in ["NOM_AGENT", "PRENOM_AGENT", "LOCALITE_AGENT"]:
        df[col] = df[col].astype(str).str.strip().str.title().replace("Nan", np.nan)

    # A4
    n = df["EMAIL_AGENT"].isna().sum()
    stats.record("A4 - email_agent null rempli", n)
    df["EMAIL_AGENT"] = df.apply(
        lambda r: r["EMAIL_AGENT"] if pd.notna(r["EMAIL_AGENT"])
        else f"agent.{str(r['CODE_AGENT']).lower()}@maghrebia.com.tn",
        axis=1
    )

    # A5
    valid_etat = {"A", "S", "R"}
    mask = ~df["ETAT_AGENT"].isin(valid_etat)
    stats.record("A5 - ETAT_AGENT invalide → A", mask.sum())
    df.loc[mask, "ETAT_AGENT"] = "A"

    # A6 — Tunisie : lat [30, 38], lon [7.5, 11.8]
    mask_lat = (df["LATITUDE_AGENT"] < 30) | (df["LATITUDE_AGENT"] > 38)
    mask_lon = (df["LONGITUDE_AGENT"] < 7.5) | (df["LONGITUDE_AGENT"] > 11.8)
    n = (mask_lat | mask_lon).sum()
    stats.record("A6 - GPS hors Tunisie → NaN", n)
    df.loc[mask_lat | mask_lon, ["LATITUDE_AGENT", "LONGITUDE_AGENT"]] = np.nan

    # A7
    df = ensure_unknown_dim_agent(df, stats)

    stats.n_output = len(df)
    return df, stats


def clean_dim_client(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      C1 — Suppression doublons ID_CLIENT
      C2 — TYPE_PERSONNE : strip + upper → {P, M}; invalide → 'P'
      C3 — NOM : strip + title case; null → 'INCONNU'
      C4 — VILLE : strip + title case
      C5 — DATE_NAISSANCE : parse multi-format + bornes [1920, 2005]
      C6 — SEXE : strip + upper → {M, F}; invalide → NaN
      C7 — EMAIL : strip + lower; format invalide → NaN
    C8 — CODE_POSTAL : entier, hors [1000, 9999] → NaN
    C9 — Ajout d'une ligne UNKNOWN (ID_CLIENT=0) pour imputation FK
    """
    stats = CleaningStats("DIM_CLIENT")
    df = df_raw.copy()
    stats.n_input = len(df)

    # C1
    n = df.duplicated("ID_CLIENT").sum()
    stats.record("C1 - doublons ID_CLIENT", n)
    df = df.drop_duplicates("ID_CLIENT")

    # C2
    df["TYPE_PERSONNE"] = clean_text(df["TYPE_PERSONNE"])
    mask = ~df["TYPE_PERSONNE"].isin({"P", "M"})
    stats.record("C2 - TYPE_PERSONNE invalide → P", mask.sum())
    df.loc[mask, "TYPE_PERSONNE"] = "P"

    # C3
    df["NOM"] = df["NOM"].astype(str).str.strip().str.title()
    n = (df["NOM"].isnull() | (df["NOM"] == "Nan")).sum()
    stats.record("C3 - NOM null → INCONNU", n)
    df["NOM"] = df["NOM"].replace("Nan", "INCONNU").fillna("INCONNU")

    # C4
    df["VILLE"] = df["VILLE"].astype(str).str.strip().str.title().replace("Nan", np.nan)

    # C5
    df["DATE_NAISSANCE"] = parse_dates_multi_format(df["DATE_NAISSANCE"])
    borne_min = pd.Timestamp("1920-01-01")
    borne_max = pd.Timestamp("2005-12-31")
    mask = df["DATE_NAISSANCE"].notna() & (
        (df["DATE_NAISSANCE"] < borne_min) | (df["DATE_NAISSANCE"] > borne_max)
    )
    stats.record("C5 - DATE_NAISSANCE hors bornes → NaT", mask.sum())
    df.loc[mask, "DATE_NAISSANCE"] = pd.NaT

    n_null_dob = df["DATE_NAISSANCE"].isna().sum()
    stats.record("C5b - DATE_NAISSANCE null → flag AGE_UNKNOWN & imputation", n_null_dob)
    df["AGE_UNKNOWN"] = df["DATE_NAISSANCE"].isna().astype(int)
    df["DATE_NAISSANCE"] = df["DATE_NAISSANCE"].fillna(pd.Timestamp("1900-01-01"))

    # C6
    df["SEXE"] = clean_text(df["SEXE"])
    mask = ~df["SEXE"].isin({"M", "F"}) & df["SEXE"].notna()
    stats.record("C6 - SEXE invalide → NaN", mask.sum())
    df.loc[mask, "SEXE"] = np.nan

    # C7
    def valid_email(e):
        if pd.isna(e):
            return np.nan
        e = str(e).strip().lower()
        return e if "@" in e and "." in e.split("@")[-1] else np.nan
    n_before = df["EMAIL"].notna().sum()
    df["EMAIL"] = df["EMAIL"].apply(valid_email)
    stats.record("C7 - EMAIL format invalide → NaN", n_before - df["EMAIL"].notna().sum())

    # C8
    df["CODE_POSTAL"] = pd.to_numeric(df["CODE_POSTAL"], errors="coerce")
    mask = df["CODE_POSTAL"].notna() & (
        (df["CODE_POSTAL"] < 1000) | (df["CODE_POSTAL"] > 9999)
    )
    stats.record("C8 - CODE_POSTAL hors [1000,9999] → NaN", mask.sum())
    df.loc[mask, "CODE_POSTAL"] = np.nan
    df["CODE_POSTAL"] = df["CODE_POSTAL"].astype("Int64")

    # C9
    df = ensure_unknown_dim_client(df, stats)

    stats.n_output = len(df)
    return df, stats


def clean_dim_police(df_raw: pd.DataFrame,
                     dim_agent: pd.DataFrame,
                     dim_client: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      P1 — Suppression doublons ID_POLICE
      P2 — NUM_POLICE null → généré automatiquement
      P3 — SITUATION valeurs invalides → 'V'
      P4 — DUREE valeurs invalides → 'R'
      P5 — PERIODICITE valeurs invalides → 'A'
    P6 — BRANCHE valeurs invalides → imputation (mode)
      P7 — DATE_EFFET / DATE_ECHEANCE : parse + cohérence (effet < echéance)
      P8 — BONUS_MALUS : bornes [0.5, 3.5]; null AUTO → 1.0
      P9 — POLRP orphelin (police mère inexistante) → NaN
    P10 — FK orphelines ID_AGENT / ID_CLIENT → imputation ID=0
    P11 — Ajout d'une ligne UNKNOWN (ID_POLICE=0) pour imputation FK
    """
    stats = CleaningStats("DIM_POLICE")
    df = df_raw.copy()
    stats.n_input = len(df)

    # P1
    n = df.duplicated("ID_POLICE").sum()
    stats.record("P1 - doublons ID_POLICE", n)
    df = df.drop_duplicates("ID_POLICE")

    # P2
    mask = df["NUM_POLICE"].isnull() | (df["NUM_POLICE"].astype(str).str.strip() == "")
    stats.record("P2 - NUM_POLICE null → généré", mask.sum())
    df.loc[mask, "NUM_POLICE"] = df.loc[mask, "ID_POLICE"].apply(lambda x: f"GEN{x:07d}")

    # P3
    valid_sit = {"V", "R", "T", "S", "A"}
    df["SITUATION"] = clean_text(df["SITUATION"])
    mask = ~df["SITUATION"].isin(valid_sit)
    stats.record("P3 - SITUATION invalide → V", mask.sum())
    df.loc[mask, "SITUATION"] = "V"

    # P4
    valid_duree = {"R", "S", "F"}
    df["DUREE"] = df["DUREE"].astype(str).str.strip().str.upper()
    mask = ~df["DUREE"].isin(valid_duree)
    stats.record("P4 - DUREE invalide → R", mask.sum())
    df.loc[mask, "DUREE"] = "R"

    # P5
    valid_per = {"A", "S", "T", "C"}
    df["PERIODICITE"] = df["PERIODICITE"].astype(str).str.strip().str.upper()
    mask = ~df["PERIODICITE"].isin(valid_per)
    stats.record("P5 - PERIODICITE invalide → A", mask.sum())
    df.loc[mask, "PERIODICITE"] = "A"

    # P6
    valid_branche = VALID_BRANCHES
    df["BRANCHE"] = df["BRANCHE"].astype(str).str.strip().str.upper()
    default_branche = mode_or_default(
        df.loc[df["BRANCHE"].isin(valid_branche), "BRANCHE"],
        DEFAULT_BRANCHE,
    )
    mask = ~df["BRANCHE"].isin(valid_branche)
    stats.record("P6 - BRANCHE invalide → imputation", mask.sum())
    df.loc[mask, "BRANCHE"] = default_branche

    # P7
    df["DATE_EFFET"]    = parse_dates_multi_format(df["DATE_EFFET"])
    df["DATE_ECHEANCE"] = parse_dates_multi_format(df["DATE_ECHEANCE"])
    mask_incoh = (
        df["DATE_EFFET"].notna() & df["DATE_ECHEANCE"].notna() &
        (df["DATE_EFFET"] >= df["DATE_ECHEANCE"])
    )
    stats.record("P7 - DATE incohérente (effet≥échéance) → corrigé", mask_incoh.sum())
    df.loc[mask_incoh, "DATE_ECHEANCE"] = (
        df.loc[mask_incoh, "DATE_EFFET"] + pd.DateOffset(years=1)
    )

    # P8
    df["BONUS_MALUS"] = pd.to_numeric(df["BONUS_MALUS"], errors="coerce")
    mask_bm = df["BONUS_MALUS"].notna() & (
        (df["BONUS_MALUS"] < 0.5) | (df["BONUS_MALUS"] > 3.5)
    )
    stats.record("P8 - BONUS_MALUS hors [0.5,3.5] → 1.0", mask_bm.sum())
    df.loc[mask_bm, "BONUS_MALUS"] = 1.0
    # AUTO sans BM → 1.0
    mask_null_bm = (df["BRANCHE"] == "AUTO") & df["BONUS_MALUS"].isna()
    stats.record("P8 - BONUS_MALUS AUTO null → 1.0", mask_null_bm.sum())
    df.loc[mask_null_bm, "BONUS_MALUS"] = 1.0

    # P9 — POLRP orphelin
    valid_ids = set(df["ID_POLICE"].astype(str))
    mask_polrp = df["POLRP"].notna() & (~df["POLRP"].astype(str).isin(valid_ids))
    stats.record("P9 - POLRP orphelin → NaN", mask_polrp.sum())
    df.loc[mask_polrp, "POLRP"] = np.nan
    df["POLRP"] = pd.to_numeric(df["POLRP"], errors="coerce").astype("Int64")

    # P10 — FK orphelines
    df["ID_AGENT"] = pd.to_numeric(df["ID_AGENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    df["ID_CLIENT"] = pd.to_numeric(df["ID_CLIENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    valid_agents  = set(dim_agent["ID_AGENT"].astype(str))
    valid_clients = set(dim_client["ID_CLIENT"].astype(str))
    mask_agent_fk = ~df["ID_AGENT"].astype(str).isin(valid_agents)
    mask_client_fk = ~df["ID_CLIENT"].astype(str).isin(valid_clients)
    stats.record("P10a - FK ID_AGENT orpheline → 0", mask_agent_fk.sum())
    stats.record("P10b - FK ID_CLIENT orpheline → 0", mask_client_fk.sum())
    df.loc[mask_agent_fk, "ID_AGENT"] = UNKNOWN_ID
    df.loc[mask_client_fk, "ID_CLIENT"] = UNKNOWN_ID

    # P11
    df = ensure_unknown_dim_police(df, stats)

    # Forcer les types des clés numériques avant chargement SQL
    for col in ["ID_POLICE", "ID_BRANCHE", "CODE_PRODUIT", "ID_AGENT", "ID_CLIENT", "POLRP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ID_POLICE"] = df["ID_POLICE"].fillna(UNKNOWN_ID).astype(int)
    df["ID_AGENT"] = df["ID_AGENT"].fillna(UNKNOWN_ID).astype(int)
    df["ID_CLIENT"] = df["ID_CLIENT"].fillna(UNKNOWN_ID).astype(int)
    df["ID_BRANCHE"] = df["ID_BRANCHE"].astype("Int64")
    df["CODE_PRODUIT"] = df["CODE_PRODUIT"].astype("Int64")
    df["POLRP"] = df["POLRP"].astype("Int64")

    stats.n_output = len(df)
    return df, stats


def clean_dim_vehicule(df_raw: pd.DataFrame,
                       dim_police: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      V1 — Suppression doublons ID_VEHICULE
    V2 — FK ID_POLICE orphelin → imputation ID=0
      V3 — IMMATRICULATION null → 'INCONNU'
      V4 — GENRE_VEHICULE invalide → 'VP'
      V5 — DATE_MEC : parse + bornes [1970, 2025]
      V6 — VALEUR_A_NEUF négative ou > 2M TND → NaN
      V7 — VALEUR_ACTUELLE > VALEUR_A_NEUF → remplacé par 80% NEUF
      V8 — NB_PLACE : bornes [1, 100]
      V9 — PUISSANCE : bornes [1, 500 CV]
    """
    stats = CleaningStats("DIM_VEHICULE")
    df = df_raw.copy()
    stats.n_input = len(df)

    # V1
    n = df.duplicated("ID_VEHICULE").sum()
    stats.record("V1 - doublons ID_VEHICULE", n)
    df = df.drop_duplicates("ID_VEHICULE")

    # V2
    df["ID_POLICE"] = pd.to_numeric(df["ID_POLICE"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    valid_pol = set(dim_police["ID_POLICE"].astype(str))
    mask = ~df["ID_POLICE"].astype(str).isin(valid_pol)
    stats.record("V2 - FK ID_POLICE orphelin → 0", mask.sum())
    df.loc[mask, "ID_POLICE"] = UNKNOWN_ID

    # V3
    mask = df["IMMATRICULATION"].isna() | (df["IMMATRICULATION"].astype(str).str.strip() == "")
    stats.record("V3 - IMMATRICULATION null → INCONNU", mask.sum())
    df.loc[mask, "IMMATRICULATION"] = "INCONNU"

    # V4
    valid_genres = {"VP", "VU", "PL", "TC", "AR"}
    df["GENRE_VEHICULE"] = df["GENRE_VEHICULE"].astype(str).str.strip().str.upper()
    mask = ~df["GENRE_VEHICULE"].isin(valid_genres)
    stats.record("V4 - GENRE_VEHICULE invalide → VP", mask.sum())
    df.loc[mask, "GENRE_VEHICULE"] = "VP"

    # V5
    df["DATE_MEC"] = parse_dates_multi_format(df["DATE_MEC"])
    borne_min = pd.Timestamp("1970-01-01")
    borne_max = pd.Timestamp("2025-12-31")
    mask = df["DATE_MEC"].notna() & (
        (df["DATE_MEC"] < borne_min) | (df["DATE_MEC"] > borne_max)
    )
    stats.record("V5 - DATE_MEC hors bornes → NaT", mask.sum())
    df.loc[mask, "DATE_MEC"] = pd.NaT

    # V6
    df["VALEUR_A_NEUF"] = pd.to_numeric(df["VALEUR_A_NEUF"], errors="coerce")
    mask = df["VALEUR_A_NEUF"].notna() & (
        (df["VALEUR_A_NEUF"] < 0) | (df["VALEUR_A_NEUF"] > 2_000_000)
    )
    stats.record("V6 - VALEUR_A_NEUF invalide → NaN", mask.sum())
    df.loc[mask, "VALEUR_A_NEUF"] = np.nan

    # V7
    df["VALEUR_ACTUELLE"] = pd.to_numeric(df["VALEUR_ACTUELLE"], errors="coerce")
    mask = (
        df["VALEUR_ACTUELLE"].notna() & df["VALEUR_A_NEUF"].notna() &
        (df["VALEUR_ACTUELLE"] > df["VALEUR_A_NEUF"])
    )
    stats.record("V7 - VALEUR_ACTUELLE > NEUF → 80% NEUF", mask.sum())
    df.loc[mask, "VALEUR_ACTUELLE"] = df.loc[mask, "VALEUR_A_NEUF"] * 0.8

    # V8
    df["NB_PLACE"] = pd.to_numeric(df["NB_PLACE"], errors="coerce")
    mask = df["NB_PLACE"].notna() & ((df["NB_PLACE"] < 1) | (df["NB_PLACE"] > 100))
    stats.record("V8 - NB_PLACE hors [1,100] → NaN", mask.sum())
    df.loc[mask, "NB_PLACE"] = np.nan

    # V9
    df["PUISSANCE"] = pd.to_numeric(df["PUISSANCE"], errors="coerce")
    mask = df["PUISSANCE"].notna() & ((df["PUISSANCE"] < 1) | (df["PUISSANCE"] > 500))
    stats.record("V9 - PUISSANCE hors [1,500CV] → NaN", mask.sum())
    df.loc[mask, "PUISSANCE"] = np.nan

    stats.n_output = len(df)
    return df, stats


def clean_fact_emission(df_raw: pd.DataFrame,
                        dim_police: pd.DataFrame,
                        dim_agent: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      E1  — Suppression doublons NUM_QUITTANCE (PK)
      E2  — ETAT_QUIT : strip + upper → {E, P, A}; invalide → 'E'
    E3  — BRANCHE : strip + upper → {AUTO, IRDS, SANTE}; invalide → imputation
      E4  — PERIODICITE : strip + upper → {A, S, T, C}; invalide → 'A'
    E5  — ANNEE_ECHEANCE : bornes [2018, 2025]; invalide → imputation
      E6  — Montants négatifs → ABS()  (dirty: -prime devient prime)
      E7  — Montants NULL → 0.0
      E8  — Outliers montants MT_PNET > 99.9% → winsorize
      E9  — Recompute MT_PNET = Σ garanties si écart > 0.5 DT
      E10 — Recompute MT_FGA = MT_RC × 0.25 si écart > 0.5 DT
      E11 — Recompute MT_PTT = PNET + FGA + TIMBRE + TAXE si écart > 0.5 DT
      E12 — Dates : parse multi-format; DATE_EFFET < DATE_ECHEANCE enforced
      E13 — BONUS_MALUS null AUTO → 1.0
    E14 — FK ID_POLICE orphelin → imputation ID=0
    E15 — FK ID_AGENT  orphelin → imputation ID=0
    """
    stats = CleaningStats("FACT_EMISSION")
    df = df_raw.copy()
    stats.n_input = len(df)

    # E1
    n = df.duplicated("NUM_QUITTANCE").sum()
    stats.record("E1 - doublons NUM_QUITTANCE", n)
    df = df.drop_duplicates("NUM_QUITTANCE")

    # E2
    df["ETAT_QUIT"] = clean_text(df["ETAT_QUIT"])
    mask = ~df["ETAT_QUIT"].isin({"E", "P", "A"})
    stats.record("E2 - ETAT_QUIT invalide → E", mask.sum())
    df.loc[mask, "ETAT_QUIT"] = "E"

    # E3
    df["BRANCHE"] = clean_text(df["BRANCHE"])
    branche_by_police = (
        dim_police[["ID_POLICE", "BRANCHE"]]
        .drop_duplicates("ID_POLICE")
        .set_index("ID_POLICE")["BRANCHE"]
        .to_dict()
    )
    default_branche = mode_or_default(
        dim_police.loc[dim_police["BRANCHE"].isin(VALID_BRANCHES), "BRANCHE"],
        DEFAULT_BRANCHE,
    )
    mask = ~df["BRANCHE"].isin(VALID_BRANCHES)
    stats.record("E3 - BRANCHE invalide → imputation", mask.sum())
    imputed_branche = pd.to_numeric(df.loc[mask, "ID_POLICE"], errors="coerce").map(branche_by_police)
    df.loc[mask, "BRANCHE"] = imputed_branche.where(imputed_branche.isin(VALID_BRANCHES), default_branche)

    # E4
    df["PERIODICITE"] = clean_text(df["PERIODICITE"])
    mask = ~df["PERIODICITE"].isin({"A", "S", "T", "C"})
    stats.record("E4 - PERIODICITE invalide → A", mask.sum())
    df.loc[mask, "PERIODICITE"] = "A"

    # E5 — années aberrantes (0, 1900, 9999)
    for dc in ["DATE_EFFET", "DATE_ECHEANCE", "DATE_EMISSION"]:
        df[dc] = parse_dates_multi_format(df[dc])

    df["ANNEE_ECHEANCE"] = pd.to_numeric(df["ANNEE_ECHEANCE"], errors="coerce")
    mask_invalid_yr = df["ANNEE_ECHEANCE"].isna() | ~df["ANNEE_ECHEANCE"].between(2018, 2025)
    n_invalid_yr = mask_invalid_yr.sum()
    stats.record("E5 - ANNEE_ECHEANCE invalide → Suppression", n_invalid_yr)
    df = df[~mask_invalid_yr].copy()
    df["ANNEE_ECHEANCE"] = df["ANNEE_ECHEANCE"].astype(int)

    # E6 + E7 — montants
    montant_cols = [
        "MT_RC","MT_DOM","MT_INC","MT_VOL","MT_BGL","MT_DOMCOLL",
        "MT_TEL","MT_CAS","MT_PTA","MT_ASS","MT_IMMOB",
        "MT_PNET","MT_FGA","MT_TIMBRE","MT_TAXE","MT_PTT","MT_COMMISSION"
    ]
    
    # E6_bis - MT_PNET négatifs
    df["MT_PNET"] = pd.to_numeric(df["MT_PNET"], errors="coerce")
    n_neg_pnet = (df["MT_PNET"] < 0).sum()
    if n_neg_pnet > 0:
        stats.record("E6 - MT_PNET négatif → Suppression", n_neg_pnet)
        df = df[df["MT_PNET"] >= 0].copy()

    for col in montant_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            stats.record(f"E6 - {col} négatif → ABS", n_neg)
            df[col] = df[col].abs()
        n_null = df[col].isna().sum()
        if n_null > 0:
            stats.record(f"E7 - {col} null → 0.0", n_null)
            df[col] = df[col].fillna(0.0)

    # E8 — winsorize MT_PNET (outliers > 99.9%)
    pnet_q999 = df["MT_PNET"].quantile(0.999)
    mask_out = df["MT_PNET"] > pnet_q999
    stats.record(f"E8 - MT_PNET > {pnet_q999:.0f} DT winsorisé", mask_out.sum())
    df.loc[mask_out, "MT_PNET"] = pnet_q999

    # E9 — recompute PNET
    garantie_cols = ["MT_RC","MT_DOM","MT_INC","MT_VOL","MT_BGL",
                     "MT_DOMCOLL","MT_TEL","MT_CAS","MT_PTA","MT_ASS","MT_IMMOB"]
    df["_PNET_CALC"] = df[garantie_cols].sum(axis=1).round(3)
    mask = (df["_PNET_CALC"] - df["MT_PNET"]).abs() > 0.5
    stats.record("E9 - MT_PNET recalculé (Σ garanties)", mask.sum())
    df.loc[mask, "MT_PNET"] = df.loc[mask, "_PNET_CALC"]
    df.drop(columns=["_PNET_CALC"], inplace=True)

    # E10 — recompute FGA = RC × 0.25
    df["_FGA_CALC"] = (df["MT_RC"] * 0.25).round(3)
    mask = (df["_FGA_CALC"] - df["MT_FGA"]).abs() > 0.5
    stats.record("E10 - MT_FGA recalculé (RC × 0.25)", mask.sum())
    df.loc[mask, "MT_FGA"] = df.loc[mask, "_FGA_CALC"]
    df.drop(columns=["_FGA_CALC"], inplace=True)

    # E11 — recompute PTT
    df["_PTT_CALC"] = (
        df["MT_PNET"] + df["MT_FGA"] + df["MT_TIMBRE"] + df["MT_TAXE"]
    ).round(3)
    mask = (df["_PTT_CALC"] - df["MT_PTT"]).abs() > 0.5
    stats.record("E11 - MT_PTT recalculé (PNET+FGA+TIMBRE+TAXE)", mask.sum())
    df.loc[mask, "MT_PTT"] = df.loc[mask, "_PTT_CALC"]
    df.drop(columns=["_PTT_CALC"], inplace=True)

    # E12 — cohérence dates
    mask_date = (
        df["DATE_EFFET"].notna() & df["DATE_ECHEANCE"].notna() &
        (df["DATE_EFFET"] >= df["DATE_ECHEANCE"])
    )
    stats.record("E12 - DATE_EFFET≥DATE_ECHEANCE corrigé", mask_date.sum())
    df.loc[mask_date, "DATE_ECHEANCE"] = (
        df.loc[mask_date, "DATE_EFFET"] + pd.DateOffset(years=1)
    )

    # E13 — BM
    df["BONUS_MALUS"] = pd.to_numeric(df["BONUS_MALUS"], errors="coerce")
    mask = (df["BRANCHE"] == "AUTO") & df["BONUS_MALUS"].isna()
    stats.record("E13 - BONUS_MALUS AUTO null → 1.0", mask.sum())
    df.loc[mask, "BONUS_MALUS"] = 1.0

    # E14 + E15 — FK
    df["ID_POLICE"] = pd.to_numeric(df["ID_POLICE"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    df["ID_AGENT"] = pd.to_numeric(df["ID_AGENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    valid_pol = set(dim_police["ID_POLICE"].astype(str))
    valid_agt = set(dim_agent["ID_AGENT"].astype(str))
    mask_pol_fk = ~df["ID_POLICE"].astype(str).isin(valid_pol)
    mask_agt_fk = ~df["ID_AGENT"].astype(str).isin(valid_agt)
    stats.record("E14 - FK ID_POLICE orpheline → 0", mask_pol_fk.sum())
    stats.record("E15 - FK ID_AGENT orpheline → 0", mask_agt_fk.sum())
    df.loc[mask_pol_fk, "ID_POLICE"] = UNKNOWN_ID
    df.loc[mask_agt_fk, "ID_AGENT"] = UNKNOWN_ID

    stats.n_output = len(df)
    return df, stats


def clean_fact_annulation(df_raw: pd.DataFrame,
                          dim_police: pd.DataFrame,
                          dim_agent: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      N1 — Suppression doublons NUM_QUITTANCE
      N2 — NATURE_ANNULATION invalide → 'NEANT'
      N3 — SITUATION_ANNULATION invalide → 'En Cours'
      N4 — Montants annulés négatifs → ABS / null → 0.0
      N5 — DATE_ANNULATION < DATE_EMISSION → swap
    N6 — FK ID_POLICE orphelin → imputation ID=0
    N7 — FK ID_AGENT orphelin → imputation ID=0
    """
    stats = CleaningStats("FACT_ANNULATION")
    df = df_raw.copy()
    stats.n_input = len(df)

    # N1
    n = df.duplicated("NUM_QUITTANCE").sum()
    stats.record("N1 - doublons NUM_QUITTANCE", n)
    df = df.drop_duplicates("NUM_QUITTANCE")

    # N2
    valid_nat = {"Ann.Cxp", "Annulation", "Rist.Caisse", "Rist.Agent",
                 "Rist.Cheque", "Rist.Bank", "NEANT"}
    mask = ~df["NATURE_ANNULATION"].isin(valid_nat)
    stats.record("N2 - NATURE_ANNULATION invalide → NEANT", mask.sum())
    df.loc[mask, "NATURE_ANNULATION"] = "NEANT"

    # N3
    mask = ~df["SITUATION_ANNULATION"].isin({"Anterieur", "En Cours"})
    stats.record("N3 - SITUATION_ANNULATION invalide → En Cours", mask.sum())
    df.loc[mask, "SITUATION_ANNULATION"] = "En Cours"

    # N4
    amt_cols = ["MT_RC_ANN","MT_DOM_ANN","MT_INC_ANN","MT_VOL_ANN",
                "MT_PNN_ANN","MT_FGA_ANN","MT_TIMBRE","MT_PTT_ANN","MT_COMMISSION_ANN"]
    for col in amt_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].abs().fillna(0.0)

    # N5
    df["DATE_EMISSION"]   = parse_dates_multi_format(df["DATE_EMISSION"])
    df["DATE_ANNULATION"] = parse_dates_multi_format(df["DATE_ANNULATION"])
    mask = (
        df["DATE_EMISSION"].notna() & df["DATE_ANNULATION"].notna() &
        (df["DATE_ANNULATION"] < df["DATE_EMISSION"])
    )
    stats.record("N5 - DATE_ANNULATION < DATE_EMISSION → swap", mask.sum())
    tmp = df.loc[mask, "DATE_EMISSION"].copy()
    df.loc[mask, "DATE_EMISSION"]   = df.loc[mask, "DATE_ANNULATION"]
    df.loc[mask, "DATE_ANNULATION"] = tmp

    # N6
    df["ID_POLICE"] = pd.to_numeric(df["ID_POLICE"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    valid_pol = set(dim_police["ID_POLICE"].astype(str))
    mask_pol = ~df["ID_POLICE"].astype(str).isin(valid_pol)
    stats.record("N6 - FK ID_POLICE orphelin → 0", mask_pol.sum())
    df.loc[mask_pol, "ID_POLICE"] = UNKNOWN_ID

    if "ID_AGENT" in df.columns:
        df["ID_AGENT"] = pd.to_numeric(df["ID_AGENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
        valid_agt = set(dim_agent["ID_AGENT"].astype(str))
        mask_agt = ~df["ID_AGENT"].astype(str).isin(valid_agt)
        stats.record("N7 - FK ID_AGENT orphelin → 0", mask_agt.sum())
        df.loc[mask_agt, "ID_AGENT"] = UNKNOWN_ID

    stats.n_output = len(df)
    return df, stats


def clean_fact_impaye(df_raw: pd.DataFrame,
                      dim_police: pd.DataFrame,
                      dim_agent: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """
    Règles appliquées :
      I1 — Montants négatifs → ABS; null → 0.0
      I2 — MT_ACP > MT_PTT → plafonné à MT_PTT (acompte ne dépasse pas la prime)
      I3 — Dates parse multi-format
    I4 — FK orphelines → imputation ID=0
    """
    stats = CleaningStats("FACT_IMPAYE")
    df = df_raw.copy()
    stats.n_input = len(df)

    # I1
    for col in ["MT_PNN","MT_TAXE","MT_PTT","MT_COMMISSION","MT_ACP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").abs().fillna(0.0)

    # I2
    mask = df["MT_ACP"] > df["MT_PTT"]
    stats.record("I2 - MT_ACP > MT_PTT → plafonné", mask.sum())
    df.loc[mask, "MT_ACP"] = df.loc[mask, "MT_PTT"]

    # I3
    df["DATE_EMISSION"]  = parse_dates_multi_format(df["DATE_EMISSION"])
    df["DATE_SITUATION"] = parse_dates_multi_format(df["DATE_SITUATION"])

    # I4
    df["ID_POLICE"] = pd.to_numeric(df["ID_POLICE"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    df["ID_AGENT"] = pd.to_numeric(df["ID_AGENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    valid_pol = set(dim_police["ID_POLICE"].astype(str))
    valid_agt = set(dim_agent["ID_AGENT"].astype(str))
    mask_pol_fk = ~df["ID_POLICE"].astype(str).isin(valid_pol)
    mask_agt_fk = ~df["ID_AGENT"].astype(str).isin(valid_agt)
    stats.record("I4a - FK ID_POLICE orpheline → 0", mask_pol_fk.sum())
    stats.record("I4b - FK ID_AGENT orpheline → 0", mask_agt_fk.sum())
    df.loc[mask_pol_fk, "ID_POLICE"] = UNKNOWN_ID
    df.loc[mask_agt_fk, "ID_AGENT"] = UNKNOWN_ID

    stats.n_output = len(df)
    return df, stats


def clean_fact_sinistre(df_raw: pd.DataFrame,
                          dim_police: pd.DataFrame,
                          dim_client: pd.DataFrame,
                          dim_agent: pd.DataFrame,
                          dim_vehicule: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    stats = CleaningStats("FACT_SINISTRE")
    df = df_raw.copy()
    stats.n_input = len(df)

    # S1 - Uniqueness
    n = df.duplicated("NUM_SINISTRE").sum()
    stats.record("S1 - doublons NUM_SINISTRE", n)
    df = df.drop_duplicates("NUM_SINISTRE")

    # S2 - FK validations
    df["ID_POLICE"] = pd.to_numeric(df["ID_POLICE"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    df["ID_CLIENT"] = pd.to_numeric(df["ID_CLIENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    df["ID_AGENT"] = pd.to_numeric(df["ID_AGENT"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    
    valid_pol = set(dim_police["ID_POLICE"].astype(str))
    valid_cli = set(dim_client["ID_CLIENT"].astype(str))
    valid_agt = set(dim_agent["ID_AGENT"].astype(str))
    
    mask_pol = ~df["ID_POLICE"].astype(str).isin(valid_pol)
    mask_cli = ~df["ID_CLIENT"].astype(str).isin(valid_cli)
    mask_agt = ~df["ID_AGENT"].astype(str).isin(valid_agt)
    
    stats.record("S2a - FK ID_POLICE orphelin → 0", mask_pol.sum())
    stats.record("S2b - FK ID_CLIENT orphelin → 0", mask_cli.sum())
    stats.record("S2c - FK ID_AGENT orphelin → 0", mask_agt.sum())
    
    df.loc[mask_pol, "ID_POLICE"] = UNKNOWN_ID
    df.loc[mask_cli, "ID_CLIENT"] = UNKNOWN_ID
    df.loc[mask_agt, "ID_AGENT"] = UNKNOWN_ID
    
    # ID_VEHICULE nullable
    if "ID_VEHICULE" in df.columns:
        df["ID_VEHICULE"] = pd.to_numeric(df["ID_VEHICULE"], errors="coerce").astype("Int64")

    # S3 - Dates
    df["DATE_SURVENANCE"] = parse_dates_multi_format(df["DATE_SURVENANCE"])
    df["DATE_DECLARATION"] = parse_dates_multi_format(df["DATE_DECLARATION"])
    mask_dates = (df["DATE_SURVENANCE"].notna() & df["DATE_DECLARATION"].notna() & 
                 (df["DATE_SURVENANCE"] > df["DATE_DECLARATION"]))
    stats.record("S3 - DATE_SURVENANCE > DECLARATION → swap", mask_dates.sum())
    tmp = df.loc[mask_dates, "DATE_SURVENANCE"].copy()
    df.loc[mask_dates, "DATE_SURVENANCE"] = df.loc[mask_dates, "DATE_DECLARATION"]
    df.loc[mask_dates, "DATE_DECLARATION"] = tmp

    # S4 - Financials
    for col in ["MT_EVALUATION", "MT_PAYE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").abs().fillna(0.0)

    # S5 - Status
    valid_etats = {"Ouvert", "Clos", "Refusé"}
    df["ETAT_SINISTRE"] = df["ETAT_SINISTRE"].astype(str).str.strip().str.capitalize()
    mask_etat = ~df["ETAT_SINISTRE"].isin(valid_etats)
    stats.record("S5 - ETAT_SINISTRE invalide → Ouvert", mask_etat.sum())
    df.loc[mask_etat, "ETAT_SINISTRE"] = "Ouvert"

    stats.n_output = len(df)
    return df, stats

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — Great Expectations (inline, sans serveur GE)
# ─────────────────────────────────────────────────────────────────────────────

def validate_post_cleaning(tables: dict) -> dict:
    """
    Validation après nettoyage. Toutes les règles doivent passer à 100%.
    Retourne un rapport de validation.
    """
    fe  = tables.get("FACT_EMISSION")
    dp  = tables.get("DIM_POLICE")
    da  = tables.get("DIM_AGENT")
    dc  = tables.get("DIM_CLIENT")
    fi  = tables.get("FACT_IMPAYE")

    rules_results = []

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        rules_results.append({"rule": name, "status": status, "detail": detail})
        icon = "✅" if passed else "❌"
        log.info(f"  GE {icon} {name}" + (f" — {detail}" if detail and not passed else ""))

    # ── FACT_EMISSION ──
    check("FE: NUM_QUITTANCE unique",
          fe["NUM_QUITTANCE"].duplicated().sum() == 0)
    check("FE: ETAT_QUIT ∈ {E,P,A}",
          fe["ETAT_QUIT"].isin({"E","P","A"}).all())
    check("FE: BRANCHE ∈ {AUTO,IRDS,SANTE}",
          fe["BRANCHE"].isin({"AUTO","IRDS","SANTE"}).all())
    check("FE: ANNEE_ECHEANCE ∈ [2018,2025]",
          fe["ANNEE_ECHEANCE"].between(2018, 2025).all())
    check("FE: MT_PNET ≥ 0",
          (fe["MT_PNET"] >= 0).all(),
          f"{(fe['MT_PNET'] < 0).sum()} négatifs")
    check("FE: MT_PTT ≥ 0",
          (fe["MT_PTT"] >= 0).all())
    ptt_calc = (fe["MT_PNET"] + fe["MT_FGA"] + fe["MT_TIMBRE"] + fe["MT_TAXE"]).round(1)
    diff_ptt = (ptt_calc - fe["MT_PTT"].round(1)).abs()
    check("FE: MT_PTT = PNET+FGA+TIMBRE+TAXE (±0.5)",
          (diff_ptt <= 0.5).all(),
          f"{(diff_ptt > 0.5).sum()} violations")
    fga_calc = (fe["MT_RC"] * 0.25).round(1)
    diff_fga = (fga_calc - fe["MT_FGA"].round(1)).abs()
    check("FE: MT_FGA = MT_RC × 0.25 (±0.5)",
          (diff_fga <= 0.5).all(),
          f"{(diff_fga > 0.5).sum()} violations")

    # FK
    valid_pol = set(dp["ID_POLICE"].astype(str))
    valid_agt = set(da["ID_AGENT"].astype(str))
    orphan_pol = (~fe["ID_POLICE"].astype(str).isin(valid_pol)).sum()
    orphan_agt = (~fe["ID_AGENT"].astype(str).isin(valid_agt)).sum()
    check("FE: ID_POLICE FK valide", orphan_pol == 0, f"{orphan_pol} orphelins")
    check("FE: ID_AGENT FK valide",  orphan_agt == 0, f"{orphan_agt} orphelins")

    # ── DIM_POLICE ──
    check("DP: ID_POLICE unique",
          dp["ID_POLICE"].duplicated().sum() == 0)
    check("DP: SITUATION ∈ {V,R,T,S,A}",
          dp["SITUATION"].isin({"V","R","T","S","A"}).all())
    check("DP: BRANCHE ∈ {AUTO,IRDS,SANTE}",
          dp["BRANCHE"].isin({"AUTO","IRDS","SANTE"}).all())
    dp_auto = dp[dp["BRANCHE"] == "AUTO"]
    bm_null_pct = dp_auto["BONUS_MALUS"].isna().mean() if len(dp_auto) > 0 else 0
    check("DP: BONUS_MALUS AUTO null < 5%",
          bm_null_pct < 0.05,
          f"{bm_null_pct:.1%} nulls sur polices AUTO")

    # ── DIM_CLIENT ──
    check("DC: ID_CLIENT unique",
          dc["ID_CLIENT"].duplicated().sum() == 0)
    check("DC: TYPE_PERSONNE ∈ {P,M}",
          dc["TYPE_PERSONNE"].isin({"P","M"}).all())

    # ── FACT_IMPAYE ──
    check("FI: MT_ACP ≤ MT_PTT (acompte cohérent)",
          (fi["MT_ACP"] <= fi["MT_PTT"] + 0.01).all())
    check("FI: MT_PTT ≥ 0",
          (fi["MT_PTT"] >= 0).all())

    n_pass = sum(1 for r in rules_results if r["status"] == "PASS")
    n_fail = sum(1 for r in rules_results if r["status"] == "FAIL")
    score  = n_pass / len(rules_results) * 100 if rules_results else 0

    return {
        "n_rules": len(rules_results),
        "n_pass":  n_pass,
        "n_fail":  n_fail,
        "score":   round(score, 1),
        "results": rules_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOAD — UPSERT vers PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def load_table(engine, df: pd.DataFrame, table_name: str, pk_col: str):
    """
    Chargement UPSERT idempotent :
      1. Écrit dans une table temporaire _staging
      2. INSERT ... ON CONFLICT (pk) DO UPDATE
    Ainsi re-exécuter le script est sûr.
    """
    staging = f"{table_name}_staging"

    # Normaliser les noms de colonnes en minuscule pour correspondre au schéma SQL
    df_load = df.copy()
    df_load.columns = [c.lower() for c in df_load.columns]

    # Les FACT doivent être un snapshot propre complet pour éviter de garder
    # des lignes sales d'anciens runs (ex: années invalides supprimées au clean).
    if table_name.startswith("dwh_fact_"):
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
        df_load.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            chunksize=5000,
            method="multi",
        )
        log.info(f"  ✅ {table_name} : {len(df_load):,} lignes chargées (TRUNCATE+INSERT)")
        return

    # Écriture staging (remplace à chaque run)
    df_load.to_sql(staging, engine, if_exists="replace", index=False,
                   chunksize=5000, method="multi")

    # UPSERT
    cols = [c for c in df_load.columns]
    cols_str    = ", ".join(cols)
    excluded    = ", ".join(f"EXCLUDED.{c}" for c in cols if c != pk_col.lower())
    update_cols = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk_col.lower())

    upsert_sql = f"""
        INSERT INTO {table_name} ({cols_str})
        SELECT {cols_str} FROM {staging}
        ON CONFLICT ({pk_col.lower()}) DO UPDATE SET {update_cols};
    """

    with engine.begin() as conn:
        conn.execute(text(upsert_sql))
        conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))

    log.info(f"  ✅ {table_name} : {len(df_load):,} lignes chargées (UPSERT)")


def load_table_append(engine, df: pd.DataFrame, table_name: str):
    """Pour les tables sans PK simple (FACT_EMISSION_DETAIL, etc.) — INSERT simple."""
    df_load = df.copy()
    df_load.columns = [c.lower() for c in df_load.columns]
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
    df_load.to_sql(table_name, engine, if_exists="append", index=False,
                   chunksize=5000, method="multi")
    log.info(f"  ✅ {table_name} : {len(df_load):,} lignes chargées (TRUNCATE+INSERT)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False,
         target_table: str | None = None,
         skip_validate: bool = False):

    start_time = datetime.now()
    log.info("=" * 70)
    log.info("  MAGHREBIA ASSURANCE — PIPELINE PHASE 2")
    log.info(f"  Start : {start_time:%Y-%m-%d %H:%M:%S}")
    log.info(f"  Mode  : {'DRY-RUN (pas d écriture DB)' if dry_run else 'PRODUCTION'}")
    log.info("=" * 70)

    all_stats: list[dict] = []
    engine = None

    if not dry_run:
        log.info("\n[0/5] Connexion PostgreSQL...")
        try:
            engine = create_engine(DB_URL, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("  ✅ Connexion OK")
        except Exception as e:
            log.error(f"  ❌ Connexion échouée : {e}")
            log.warning("  → Mode dry-run activé automatiquement")
            dry_run = True

    if not dry_run:
        ensure_database_schema(engine)

    # ── STEP 1 : EXTRACT ──────────────────────────────────────────────────
    log.info("\n[1/5] EXTRACT — Lecture CSV bruts...")
    raw = {}
    try:
        raw["DIM_AGENT"]   = load_csv("DIM_AGENT")
        raw["DIM_CLIENT"]  = load_csv("DIM_CLIENT")
        raw["DIM_POLICE"]  = load_csv("DIM_POLICE")
        raw["DIM_VEHICULE"] = load_csv("DIM_VEHICULE")
        raw["FACT_EMISSION"]   = load_csv("FACT_EMISSION")
        raw["FACT_ANNULATION"] = load_csv("FACT_ANNULATION")
        raw["FACT_IMPAYE"]     = load_csv("FACT_IMPAYE")
        raw["FACT_EMISSION_DETAIL"] = load_csv("FACT_EMISSION_DETAIL")
        raw["FACT_SINISTRE"]   = load_csv("FACT_SINISTRE")
    except FileNotFoundError as e:
        log.error(f"  ❌ {e}")
        sys.exit(1)

    # ── STEP 2 : CLEAN ────────────────────────────────────────────────────
    log.info("\n[2/5] CLEAN — Application des 34 règles de nettoyage...")
    cleaned = {}

    log.info("  → DIM_AGENT")
    cleaned["DIM_AGENT"], s = clean_dim_agent(raw["DIM_AGENT"])
    all_stats.append(s.summary())

    log.info("  → DIM_CLIENT")
    cleaned["DIM_CLIENT"], s = clean_dim_client(raw["DIM_CLIENT"])
    all_stats.append(s.summary())

    log.info("  → DIM_POLICE")
    cleaned["DIM_POLICE"], s = clean_dim_police(
        raw["DIM_POLICE"], cleaned["DIM_AGENT"], cleaned["DIM_CLIENT"]
    )
    all_stats.append(s.summary())

    log.info("  → DIM_VEHICULE")
    cleaned["DIM_VEHICULE"], s = clean_dim_vehicule(
        raw["DIM_VEHICULE"], cleaned["DIM_POLICE"]
    )
    all_stats.append(s.summary())

    log.info("  → FACT_EMISSION")
    cleaned["FACT_EMISSION"], s = clean_fact_emission(
        raw["FACT_EMISSION"], cleaned["DIM_POLICE"], cleaned["DIM_AGENT"]
    )
    all_stats.append(s.summary())

    log.info("  → FACT_ANNULATION")
    cleaned["FACT_ANNULATION"], s = clean_fact_annulation(
        raw["FACT_ANNULATION"], cleaned["DIM_POLICE"], cleaned["DIM_AGENT"]
    )
    all_stats.append(s.summary())

    log.info("  → FACT_IMPAYE")
    cleaned["FACT_IMPAYE"], s = clean_fact_impaye(
        raw["FACT_IMPAYE"], cleaned["DIM_POLICE"], cleaned["DIM_AGENT"]
    )
    all_stats.append(s.summary())

    log.info("  → FACT_SINISTRE")
    cleaned["FACT_SINISTRE"], s = clean_fact_sinistre(
        raw["FACT_SINISTRE"], cleaned["DIM_POLICE"], cleaned["DIM_CLIENT"], 
        cleaned["DIM_AGENT"], cleaned["DIM_VEHICULE"]
    )
    all_stats.append(s.summary())

    # FACT_EMISSION_DETAIL — imputation au lieu de suppression
    s_detail = CleaningStats("FACT_EMISSION_DETAIL")
    fed = raw["FACT_EMISSION_DETAIL"].copy()
    s_detail.n_input = len(fed)

    emission_ref = cleaned["FACT_EMISSION"][["NUM_QUITTANCE", "ID_POLICE", "ANNEE_ECHEANCE"]].copy()
    emission_ref["NUM_QUITTANCE"] = emission_ref["NUM_QUITTANCE"].astype(str)
    valid_quit = set(emission_ref["NUM_QUITTANCE"])

    police_to_quit = (
        emission_ref.dropna(subset=["ID_POLICE"])
        .groupby("ID_POLICE")["NUM_QUITTANCE"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )
    fallback_quit = mode_or_default(emission_ref["NUM_QUITTANCE"], "UNKQ000000")

    fed["NUM_QUITTANCE"] = fed["NUM_QUITTANCE"].astype(str)
    mask_orphan_quit = ~fed["NUM_QUITTANCE"].isin(valid_quit)
    s_detail.record("D1 - NUM_QUITTANCE orphelin → imputation", mask_orphan_quit.sum())
    imputed_quit = pd.to_numeric(fed.loc[mask_orphan_quit, "ID_POLICE"], errors="coerce").map(police_to_quit)
    fed.loc[mask_orphan_quit, "NUM_QUITTANCE"] = imputed_quit.fillna(fallback_quit)

    fed["ID_POLICE"] = pd.to_numeric(fed["ID_POLICE"], errors="coerce").fillna(UNKNOWN_ID).astype(int)
    valid_pol = set(cleaned["DIM_POLICE"]["ID_POLICE"].astype(str))
    mask_orphan_pol = ~fed["ID_POLICE"].astype(str).isin(valid_pol)
    s_detail.record("D2 - FK ID_POLICE orphelin → 0", mask_orphan_pol.sum())
    fed.loc[mask_orphan_pol, "ID_POLICE"] = UNKNOWN_ID

    fallback_year = int(mode_or_default(emission_ref["ANNEE_ECHEANCE"], DEFAULT_YEAR))
    quit_to_year = emission_ref.set_index("NUM_QUITTANCE")["ANNEE_ECHEANCE"].to_dict()
    fed["ANNEE_ECHEANCE"] = pd.to_numeric(fed["ANNEE_ECHEANCE"], errors="coerce")
    mask_bad_year = fed["ANNEE_ECHEANCE"].isna() | ~fed["ANNEE_ECHEANCE"].between(2018, 2025)
    s_detail.record("D3 - ANNEE_ECHEANCE invalide → imputation", mask_bad_year.sum())
    fed.loc[mask_bad_year, "ANNEE_ECHEANCE"] = (
        fed.loc[mask_bad_year, "NUM_QUITTANCE"].map(quit_to_year).fillna(fallback_year)
    )
    fed["ANNEE_ECHEANCE"] = pd.to_numeric(fed["ANNEE_ECHEANCE"], errors="coerce").fillna(fallback_year).astype(int)

    for col in ["MT_PRIME", "MT_FRAIS"]:
        fed[col] = pd.to_numeric(fed[col], errors="coerce").abs().fillna(0.0)

    s_detail.n_output = len(fed)
    all_stats.append(s_detail.summary())
    cleaned["FACT_EMISSION_DETAIL"] = fed
    log.info(
        f"  → FACT_EMISSION_DETAIL : {s_detail.n_input:,} → {s_detail.n_output:,} lignes "
        f"(imputation sans suppression)"
    )

    # ── STEP 3 : VALIDATE ─────────────────────────────────────────────────
    validation_report = {}
    if not skip_validate:
        log.info("\n[3/5] VALIDATE — Great Expectations post-cleaning...")
        validation_report = validate_post_cleaning(cleaned)
        score = validation_report["score"]
        n_p   = validation_report["n_pass"]
        n_f   = validation_report["n_fail"]
        total = validation_report["n_rules"]
        log.info(f"  Score qualité : {score:.1f}% ({n_p}/{total} règles)")
        if score < 80:
            log.error(f"  ❌ Score < 80% — pipeline interrompu")
            sys.exit(1)
        else:
            log.info(f"  ✅ Validation réussie")
    else:
        log.info("\n[3/5] VALIDATE — Skip (--skip-validate)")

    # ── STEP 4 : LOAD ─────────────────────────────────────────────────────
    if dry_run:
        log.info("\n[4/5] LOAD — DRY-RUN, aucune écriture DB")
        for name, df in cleaned.items():
            log.info(f"  [DRY] {name} : {len(df):,} lignes prêtes")
    else:
        log.info("\n[4/5] LOAD — Chargement PostgreSQL...")
        try:
            if not target_table or target_table == "DIM_AGENT":
                load_table(engine, cleaned["DIM_AGENT"],   "dim_agent",   "ID_AGENT")
            if not target_table or target_table == "DIM_CLIENT":
                load_table(engine, cleaned["DIM_CLIENT"],  "dim_client",  "ID_CLIENT")
            if not target_table or target_table == "DIM_POLICE":
                load_table(engine, cleaned["DIM_POLICE"],  "dim_police",  "ID_POLICE")
            if not target_table or target_table == "DIM_VEHICULE":
                load_table(engine, cleaned["DIM_VEHICULE"], "dim_vehicule", "ID_VEHICULE")
            if not target_table or target_table == "FACT_EMISSION":
                load_table(engine, cleaned["FACT_EMISSION"], "dwh_fact_emission", "NUM_QUITTANCE")
            if not target_table or target_table == "FACT_EMISSION_DETAIL":
                load_table_append(engine, cleaned["FACT_EMISSION_DETAIL"], "dwh_fact_emission_detail")
            if not target_table or target_table == "FACT_ANNULATION":
                load_table(engine, cleaned["FACT_ANNULATION"], "dwh_fact_annulation", "NUM_QUITTANCE")
            if not target_table or target_table == "FACT_IMPAYE":
                load_table_append(engine, cleaned["FACT_IMPAYE"], "dwh_fact_impaye")
            if not target_table or target_table == "FACT_SINISTRE":
                load_table(engine, cleaned["FACT_SINISTRE"], "dwh_fact_sinistre", "NUM_SINISTRE")
        except Exception as e:
            log.error(f"  ❌ Erreur chargement : {e}")
            raise

    # ── STEP 5 : REPORT ───────────────────────────────────────────────────
    log.info("\n[5/5] REPORT — Génération rapport qualité JSON...")
    duration = (datetime.now() - start_time).total_seconds()

    # Résumé console
    log.info("\n" + "=" * 70)
    log.info("  RÉSUMÉ DES CORRECTIONS APPLIQUÉES")
    log.info("=" * 70)
    total_corrections = 0
    total_suppressed  = 0
    total_added = 0
    for s in all_stats:
        tc = s["total_corrections"]
        td = s["n_dropped"]
        ta = s.get("n_added", 0)
        total_corrections += tc
        total_suppressed  += td
        total_added += ta
        log.info(f"  {s['table']:30s} → {s['n_input']:>7,} → {s['n_output']:>7,}"
                 f"  (−{td:,} supprimés, +{ta:,} ajoutés, {tc:,} corrections)")
    log.info("-" * 70)
    log.info(f"  {'TOTAL':30s}    Total corrections : {total_corrections:,}")
    log.info(f"  {'':30s}    Total suppressions: {total_suppressed:,}")
    log.info(f"  {'':30s}    Total ajouts      : {total_added:,}")

    if validation_report:
        log.info(f"\n  Score GE post-nettoyage : {validation_report['score']}%  "
                 f"({validation_report['n_pass']}/{validation_report['n_rules']} règles)")

    log.info(f"\n  Durée totale : {duration:.1f}s")
    log.info("=" * 70)

    # Sauvegarde JSON
    report = {
        "run_date":    start_time.isoformat(),
        "duration_s":  round(duration, 1),
        "dry_run":     dry_run,
        "cleaning":    all_stats,
        "validation":  validation_report,
        "volumes":     {name: len(df) for name, df in cleaned.items()},
    }
    report_path = REPORT_DIR / f"import_report_{start_time:%Y%m%d_%H%M}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    log.info(f"\n  💾 Rapport : {report_path}")

    # Sauvegarde CSV nettoyés (pour audit / DVC)
    clean_dir = CLEAN_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name, df in cleaned.items():
        df.to_csv(clean_dir / f"{name}_clean.csv", index=False, encoding="utf-8-sig")
    log.info(f"  💾 CSV nettoyés : {clean_dir}/")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maghrebia — Pipeline Cleaning + Loading (TDSP Phase 2)"
    )
    parser.add_argument("--dry-run",       action="store_true",
                        help="Pas d'écriture en base, rapport seul")
    parser.add_argument("--table",         type=str, default=None,
                        help="Charger une seule table (ex: FACT_EMISSION)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Sauter la validation Great Expectations")
    parser.add_argument("--data-dir",      type=str, default=None,
                        help="Répertoire CSV (override DATA_DIR)")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    report = main(
        dry_run=args.dry_run,
        target_table=args.table,
        skip_validate=args.skip_validate,
    )

    sys.exit(0 if not report.get("validation") or report["validation"].get("score", 100) >= 80 else 1)
