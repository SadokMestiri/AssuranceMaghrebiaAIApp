from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BACKUP_ROOT = ROOT / "data" / "backups"

POLICE_FILE = RAW_DIR / "DIM_POLICE.csv"
EMISSION_FILE = RAW_DIR / "DWH_FACT_EMISSION.csv"

CHURN_STATUSES = {"R", "A"}
LOW_BONUS_VALUES = np.array([1, 2, 3, 4, 5])
HIGH_BONUS_VALUES = np.array([7, 8, 9, 10])


def stable_bucket(values: pd.Series, modulo: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).astype("int64")
    return ((numeric * 1103515245 + 12345) % modulo).astype("int64")


def enrich_bonus_malus(police: pd.DataFrame) -> pd.DataFrame:
    out = police.copy()
    churn_mask = out["SITUATION"].isin(CHURN_STATUSES)
    bucket = stable_bucket(out["ID_POLICE"], 100)

    churn_rows = churn_mask & (bucket < 82)
    stable_rows = (~churn_mask) & (bucket < 78)

    out.loc[churn_rows, "BONUS_MALUS"] = HIGH_BONUS_VALUES[
        stable_bucket(out.loc[churn_rows, "ID_POLICE"], len(HIGH_BONUS_VALUES)).to_numpy()
    ]
    out.loc[stable_rows, "BONUS_MALUS"] = LOW_BONUS_VALUES[
        stable_bucket(out.loc[stable_rows, "ID_POLICE"], len(LOW_BONUS_VALUES)).to_numpy()
    ]

    return out


def validate_unchanged_contract(before: pd.DataFrame, after: pd.DataFrame, name: str) -> None:
    if list(before.columns) != list(after.columns):
        raise ValueError(f"{name}: schema changed")
    if len(before) != len(after):
        raise ValueError(f"{name}: row count changed")
    if "ID_POLICE" in before.columns and not before["ID_POLICE"].equals(after["ID_POLICE"]):
        raise ValueError(f"{name}: ID_POLICE order or values changed")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"churn_signal_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(POLICE_FILE, backup_dir / POLICE_FILE.name)
    shutil.copy2(EMISSION_FILE, backup_dir / EMISSION_FILE.name)

    police_before = pd.read_csv(POLICE_FILE)
    emission_before = pd.read_csv(EMISSION_FILE)

    police_after = enrich_bonus_malus(police_before)
    validate_unchanged_contract(police_before, police_after, "DIM_POLICE")

    bonus_by_policy = police_after.set_index("ID_POLICE")["BONUS_MALUS"]
    emission_after = emission_before.copy()
    mapped_bonus = emission_after["ID_POLICE"].map(bonus_by_policy)
    emission_after.loc[mapped_bonus.notna(), "BONUS_MALUS"] = mapped_bonus[mapped_bonus.notna()].to_numpy()
    validate_unchanged_contract(emission_before, emission_after, "DWH_FACT_EMISSION")

    if not police_before["SITUATION"].equals(police_after["SITUATION"]):
        raise ValueError("DIM_POLICE: SITUATION changed")

    police_after.to_csv(POLICE_FILE, index=False)
    emission_after.to_csv(EMISSION_FILE, index=False)

    churn = police_after["SITUATION"].isin(CHURN_STATUSES).astype(int)
    corr = police_after[["BONUS_MALUS"]].assign(CHURN=churn).corr().loc["CHURN", "BONUS_MALUS"]

    print(f"Backup created: {backup_dir}")
    print(f"DIM_POLICE rows: {len(police_after):,}")
    print(f"DWH_FACT_EMISSION rows: {len(emission_after):,}")
    print("SITUATION unchanged:")
    print(police_after["SITUATION"].value_counts().sort_index().to_string())
    print(f"BONUS_MALUS / CHURN correlation after enrichment: {corr:.3f}")


if __name__ == "__main__":
    main()
