"""
Maghrebia â€” Service Isolation Forest
AdaptÃ© pour les KPIs de Maghrebia (AUTO, IRDS, SANTE)
"""

import os
import logging
import psycopg2
import pandas as pd
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://maghrebia:maghrebia@postgres:5432/maghrebia")

FEATURES = [
    "ratio_combine_pct",
    "primes_acquises_tnd",
    "cout_sinistres_tnd",
    "nb_sinistres",
    "taux_resiliation_pct",
    "provision_totale_tnd",
    "nb_suspicions_fraude",
]

def _clean(val):
    if isinstance(val, Decimal): return float(val)
    return float(val) if val else 0.0

def detect_anomalies(departement: Optional[str] = None, contamination: float = 0.1, nb_mois_recent: int = 24) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    conn = psycopg2.connect(DATABASE_URL.replace("+psycopg2", ""))
    cur = conn.cursor()
    
    where = "WHERE departement = %s" if departement else ""
    params = [departement] if departement else []
    
    cur.execute(f"SELECT annee, mois, periode, departement, " + ", ".join(FEATURES) + 
                f" FROM kpis_mensuels {where} ORDER BY annee DESC, mois DESC LIMIT %s", 
                params + [nb_mois_recent * (1 if departement else 3)])
    
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows: return {"error": "Aucune donnÃ©e"}

    cols = ["annee","mois","periode","departement"] + FEATURES
    df = pd.DataFrame(rows, columns=cols)
    for f in FEATURES: df[f] = df[f].apply(_clean)

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    clf.fit(X_scaled)
    
    scores = clf.decision_function(X_scaled)
    labels = clf.predict(X_scaled)
    df["anomaly_score"] = scores
    df["is_anomaly"] = labels == -1

    s_min, s_max = scores.min(), scores.max()
    df["risk_score"] = ((s_max - scores) / (s_max - s_min) * 100).round(1)

    anomalies = df[df["is_anomaly"]].copy()
    normaux = df[~df["is_anomaly"]].copy()

    def row_to_dict(row):
        z_scores = {}
        dept_data = df[df["departement"] == row["departement"]][FEATURES]
        for f in FEATURES:
            mean = dept_data[f].mean()
            std = dept_data[f].std()
            if std > 0: z_scores[f] = abs((row[f] - mean) / std)
        top_features = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            "periode": row["periode"],
            "departement": row["departement"],
            "risk_score": row["risk_score"],
            "features_deviantes": [{"feature": f, "z_score": round(z, 2), "valeur": round(row[f], 2)} for f, z in top_features],
            "ratio_combine_pct": round(row["ratio_combine_pct"], 2),
            "nb_sinistres": int(row["nb_sinistres"]),
            "taux_resiliation_pct": round(row["taux_resiliation_pct"], 2),
            "nb_suspicions_fraude": int(row["nb_suspicions_fraude"]),
        }

    return {
        "nb_anomalies": len(anomalies),
        "nb_normaux": len(normaux),
        "contamination": contamination,
        "departement": departement or "Tous",
        "anomalies": [row_to_dict(r) for _, r in anomalies.sort_values("risk_score", ascending=False).iterrows()],
        "stats": {
            "score_moyen_anomalies": round(df[df["is_anomaly"]]["risk_score"].mean(), 1) if len(anomalies) > 0 else 0,
            "top_departement": anomalies["departement"].value_counts().idxmax() if len(anomalies) > 0 else None,
        }
    }
