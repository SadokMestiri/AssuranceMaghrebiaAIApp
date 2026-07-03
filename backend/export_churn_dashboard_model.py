from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent / "models"
ARTIFACT_PATH = MODELS_DIR / "churn_notebook_model.pkl"

RANDOM_STATE = 42
FEATURES = [
    "bonus_malus",
    "nb_quittances",
    "mt_pnet_moy",
    "mt_pnet_total",
    "mt_commission_moy",
    "bonus_malus_moy",
    "bonus_malus_max",
    "bonus_malus_min",
    "nb_sinistres",
    "mt_eval_moy",
    "mt_paye_moy",
    "nb_impayes",
    "taux_impaye",
    "anciennete_jours",
    "nb_branches",
    "prime_x_anciennete",
    "impaye_x_prime",
]


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}.csv", low_memory=False)


def build_features() -> pd.DataFrame:
    police = read_raw("DIM_POLICE")
    emission = read_raw("DWH_FACT_EMISSION")
    sinistre = read_raw("DWH_FACT_SINISTRE")
    impaye = read_raw("DWH_FACT_IMPAYE")

    police["CHURN"] = police["SITUATION"].isin(["R", "A"]).astype(int)
    police["DATE_EFFET"] = pd.to_datetime(police["DATE_EFFET"], errors="coerce")
    police["anciennete_jours"] = (pd.Timestamp.today() - police["DATE_EFFET"]).dt.days.clip(0)

    df = police[["ID_POLICE", "BRANCHE", "BONUS_MALUS", "anciennete_jours", "CHURN"]].rename(
        columns={
            "ID_POLICE": "id_police",
            "BRANCHE": "branche",
            "BONUS_MALUS": "bonus_malus",
        }
    )

    emission["MT_PNET"] = pd.to_numeric(emission["MT_PNET"], errors="coerce").fillna(0)
    emission["MT_COMMISSION"] = pd.to_numeric(emission["MT_COMMISSION"], errors="coerce").fillna(0)
    emission["BONUS_MALUS"] = pd.to_numeric(emission["BONUS_MALUS"], errors="coerce")
    em_agg = emission.groupby("ID_POLICE").agg(
        nb_quittances=("NUM_QUITTANCE", "count"),
        mt_pnet_moy=("MT_PNET", "mean"),
        mt_pnet_total=("MT_PNET", "sum"),
        mt_commission_moy=("MT_COMMISSION", "mean"),
        bonus_malus_moy=("BONUS_MALUS", "mean"),
        bonus_malus_max=("BONUS_MALUS", "max"),
        bonus_malus_min=("BONUS_MALUS", "min"),
        nb_branches=("BRANCHE", "nunique"),
    ).reset_index().rename(columns={"ID_POLICE": "id_police"})
    df = df.merge(em_agg, on="id_police", how="left")

    if not sinistre.empty:
        sinistre["MT_EVALUATION"] = pd.to_numeric(sinistre["MT_EVALUATION"], errors="coerce").fillna(0)
        sinistre["MT_PAYE"] = pd.to_numeric(sinistre["MT_PAYE"], errors="coerce").fillna(0)
        sin_agg = sinistre.groupby("ID_POLICE").agg(
            nb_sinistres=("NUM_SINISTRE", "count"),
            mt_eval_moy=("MT_EVALUATION", "mean"),
            mt_paye_moy=("MT_PAYE", "mean"),
        ).reset_index().rename(columns={"ID_POLICE": "id_police"})
        df = df.merge(sin_agg, on="id_police", how="left")

    if not impaye.empty:
        imp_agg = impaye.groupby("ID_POLICE").agg(
            nb_impayes=("NUM_QUITTANCE", "count"),
        ).reset_index().rename(columns={"ID_POLICE": "id_police"})
        df = df.merge(imp_agg, on="id_police", how="left")

    df["nb_impayes"] = df["nb_impayes"].fillna(0)
    df["nb_quittances"] = df["nb_quittances"].fillna(0)
    df["taux_impaye"] = df["nb_impayes"] / df["nb_quittances"].replace(0, np.nan)
    df["taux_impaye"] = df["taux_impaye"].fillna(0).clip(0, 1)
    df["prime_x_anciennete"] = df["mt_pnet_moy"].fillna(0) * df["anciennete_jours"].fillna(0)
    df["impaye_x_prime"] = df["taux_impaye"].fillna(0) * df["mt_pnet_moy"].fillna(0)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def main() -> None:
    df = build_features()
    X = df[FEATURES]
    y = df["CHURN"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train_s = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test_s = scaler.transform(imputer.transform(x_test))

    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_s, y_train)
    proba = model.predict_proba(x_test_s)[:, 1]

    thresholds = np.round(np.linspace(0.01, 0.99, 199), 3)
    rows = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "accuracy": accuracy_score(y_test, pred),
            }
        )

    best = max(rows, key=lambda row: row["f1"])
    pred = (proba >= best["threshold"]).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
    }

    artifact = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": FEATURES,
        "feature_defaults": X.median(numeric_only=True).to_dict(),
        "threshold": float(best["threshold"]),
        "source": "notebook:v3-dashboard-export",
        "notebook": "churn_prediction_v3.ipynb",
        "metrics": metrics,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved: {ARTIFACT_PATH}")
    print(f"Threshold F1: {best['threshold']:.3f}")
    print(metrics)


if __name__ == "__main__":
    main()
