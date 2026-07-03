"""
fix_segmentation.py
────────────────────
Rebuilds segmentation_model.pkl with a properly balanced K=5 K-Means.

Fixes from the notebook:
  - SP_RATIO removed from cluster features (IQR≈0 → RobustScaler blows up)
  - COUT_MOY_SIN removed (sparse, redundant with NB_SINISTRES)
  - FLAG_FLOTTE removed (near-binary, sparse)
  - K raised from 4 → 5 for business-actionable segments
  - Added clipping of BONUS_MALUS_MOY to reasonable range
  - Export path fixed to backend/models/ (not backend/notebooks/)

Run with: .venv\Scripts\python.exe fix_segmentation.py
"""
import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

BASE    = r'C:\Users\LENOVO\Desktop\PFE_\maghrebia'
RAW     = os.path.join(BASE, 'data', 'raw')
OUT_PKL = os.path.join(BASE, 'backend', 'models', 'segmentation_model.pkl')

REF_DATE = pd.Timestamp('today').normalize()  # reference date for recency/ancienneté

# ── Cleaner feature set — SP_RATIO excluded (zero-inflated kills RobustScaler)
CLUSTER_FEATURES = [
    'RECENCY_DAYS',
    'NB_POLICES',
    'NB_QUITTANCES',
    'TOTAL_PRIMES',
    'PRIME_PAR_POLICE',
    'NB_BRANCHES',
    'NB_POLICES_ACTIVES',
    'TAUX_RESILIATION',
    'ANCIENNETE_JOURS',
    'BONUS_MALUS_MOY',
    'NB_SINISTRES',
    'TAUX_IMPAYE',
    'NB_IMPAYES',
    'FLAG_MORALE',
    'FLAG_MULTI_BRANCHES',
]

# Columns to log1p-transform (heavy right-skew)
LOG_COLS = [
    'TOTAL_PRIMES', 'PRIME_PAR_POLICE', 'NB_QUITTANCES',
    'NB_SINISTRES', 'NB_IMPAYES', 'ANCIENNETE_JOURS',
]

K_OPTIMAL = 5

ACTION_PLAN = {
    '🏆 VIP — Haute Valeur':     'Programme fidélité premium — upsell multi-branches & garanties exclusives',
    '🎯 Fidèles Rentables':      'Renouvellement proactif — offre ancienneté + extension de garanties',
    '🌱 Clients Potentiels':     'Campagne activation — améliorer engagement digital + offre découverte',
    '⚠️ Clients à Risque':      'Intervention urgente agent — plan de paiement & rétention ciblée',
    '😴 Clients Dormants':       'Campagne réactivation — offre remise spéciale retour portefeuille',
    '🏢 Entreprises / Flottes':  'Contrat cadre entreprise — tarification flotte & gestionnaire dédié',
}

RADAR_DIMS = [
    ('Valeur',       'TOTAL_PRIMES'),
    ('Fidélité',     'ANCIENNETE_JOURS'),
    ('Engagement',   'NB_POLICES'),
    ('Fiabilité',    'TAUX_IMPAYE'),       # inverted: lower = better
    ('Rentabilité',  'TAUX_RESILIATION'),  # inverted: lower = better
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(name, date_cols=None):
    path = os.path.join(RAW, name)
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


def build_master():
    print('Loading raw data...')
    pol = load_csv('DIM_POLICE.csv',          ['DATE_EFFET', 'DATE_ECHEANCE'])
    cli = load_csv('DIM_CLIENT.csv',          ['DATE_NAISSANCE'])
    em  = load_csv('DWH_FACT_EMISSION.csv',   ['DATE_EMISSION'])
    sin = load_csv('DWH_FACT_SINISTRE.csv',   ['DATE_SURVENANCE'])
    imp = load_csv('DWH_FACT_IMPAYE.csv',     ['DATE_EMISSION'])
    ann = load_csv('DWH_FACT_ANNULATION.csv', ['DATE_ANNULATION'])

    # Normalise column names
    for df in [pol, cli, em, sin, imp, ann]:
        df.columns = df.columns.str.strip().str.upper()

    # Police base
    pol['BRANCHE']     = pol.get('BRANCHE', pd.Series('', index=pol.index))
    pol['SITUATION']   = pol.get('SITUATION', pd.Series('', index=pol.index))
    pol['BONUS_MALUS'] = pd.to_numeric(pol.get('BONUS_MALUS', pd.Series(1.0, index=pol.index)), errors='coerce').fillna(1.0)
    # FLAG_MORALE from client type (M = personne morale = company)
    cli['TYPE_PERSONNE'] = cli.get('TYPE_PERSONNE', pd.Series('P', index=cli.index)).str.strip().str.upper()
    cli_morale = cli[['ID_CLIENT', 'TYPE_PERSONNE']].copy()
    cli_morale['FLAG_MORALE'] = (cli_morale['TYPE_PERSONNE'] == 'M').astype(int)
    pol = pol.merge(cli_morale[['ID_CLIENT', 'FLAG_MORALE']], on='ID_CLIENT', how='left')
    pol['FLAG_MORALE'] = pol['FLAG_MORALE'].fillna(0).astype(int)

    pol_cli = pol[['ID_POLICE', 'ID_CLIENT', 'BRANCHE', 'SITUATION',
                   'BONUS_MALUS', 'DATE_EFFET', 'FLAG_MORALE']].copy()

    # Emission per client
    em['PRIME'] = em.get('MT_PTT', pd.Series(0.0, index=em.index)).fillna(
                  em.get('MT_PNET', pd.Series(0.0, index=em.index)).fillna(0))
    em_c = em.merge(pol_cli[['ID_POLICE', 'ID_CLIENT']], on='ID_POLICE', how='left')

    # Recency
    recency = (em_c.groupby('ID_CLIENT')['DATE_EMISSION'].max().reset_index()
               .rename(columns={'DATE_EMISSION': 'LAST_EM'}))
    recency['RECENCY_DAYS'] = (REF_DATE - recency['LAST_EM']).dt.days.clip(0)

    # Frequency + Monetary
    em_agg = em_c.groupby('ID_CLIENT').agg(
        NB_QUITTANCES=('NUM_QUITTANCE', 'count'),
        TOTAL_PRIMES=('PRIME', 'sum'),
    ).reset_index()

    # Police aggregation per client
    pol_agg = pol_cli.groupby('ID_CLIENT').agg(
        NB_POLICES         =('ID_POLICE',   'count'),
        NB_BRANCHES        =('BRANCHE',     'nunique'),
        NB_POLICES_ACTIVES =('SITUATION',   lambda x: (x == 'V').sum()),
        NB_POLICES_RESIL   =('SITUATION',   lambda x: (x == 'R').sum()),
        BONUS_MALUS_MOY    =('BONUS_MALUS', 'mean'),
        FLAG_MORALE        =('FLAG_MORALE', 'max'),
        DATE_EFFET_MIN     =('DATE_EFFET',  'min'),
    ).reset_index()
    pol_agg['ANCIENNETE_JOURS'] = (REF_DATE - pol_agg['DATE_EFFET_MIN']).dt.days.clip(0)
    pol_agg['TAUX_RESILIATION'] = np.where(
        pol_agg['NB_POLICES'] > 0,
        pol_agg['NB_POLICES_RESIL'] / pol_agg['NB_POLICES'], 0
    )

    # Sinistre — FACT_SINISTRE already has ID_CLIENT, no merge needed
    sin['MT_PAYE'] = pd.to_numeric(sin.get('MT_PAYE', pd.Series(0.0, index=sin.index)), errors='coerce').fillna(0)
    sin_agg = sin.groupby('ID_CLIENT').agg(
        NB_SINISTRES  =('NUM_SINISTRE', 'count'),
        COUT_SINISTRES=('MT_PAYE',      'sum'),
    ).reset_index()

    # Impayés
    imp_c = imp.merge(pol_cli[['ID_POLICE', 'ID_CLIENT']], on='ID_POLICE', how='left')
    imp_agg = imp_c.groupby('ID_CLIENT').agg(
        NB_IMPAYES=('NUM_QUITTANCE', 'count'),
    ).reset_index()

    # Assemble
    master = (recency[['ID_CLIENT', 'RECENCY_DAYS']]
              .merge(em_agg,                  on='ID_CLIENT', how='left')
              .merge(pol_agg,                 on='ID_CLIENT', how='left')
              .merge(sin_agg,                 on='ID_CLIENT', how='left')
              .merge(imp_agg,                 on='ID_CLIENT', how='left'))

    # Fill zeros
    for col in ['NB_QUITTANCES','TOTAL_PRIMES','NB_POLICES','NB_BRANCHES','NB_POLICES_ACTIVES',
                'NB_POLICES_RESIL','NB_SINISTRES','COUT_SINISTRES','NB_IMPAYES',
                'BONUS_MALUS_MOY','FLAG_MORALE']:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    master['TAUX_IMPAYE']     = np.where(master['NB_QUITTANCES'] > 0,
                                         master['NB_IMPAYES'] / master['NB_QUITTANCES'], 0)
    master['PRIME_PAR_POLICE']= np.where(master['NB_POLICES'] > 0,
                                         master['TOTAL_PRIMES'] / master['NB_POLICES'], 0)
    master['FLAG_MULTI_BRANCHES'] = (master['NB_BRANCHES'] > 1).astype(int)

    # SP_RATIO — computed but NOT used in clustering (causes scaling issues)
    master['SP_RATIO'] = np.where(
        master['TOTAL_PRIMES'] > 0,
        master['COUT_SINISTRES'] / master['TOTAL_PRIMES'], 0
    ).clip(0, 1.5)

    # CLV estimate
    prime_ann = master['TOTAL_PRIMES'] / (master['ANCIENNETE_JOURS'].fillna(365).clip(30) / 365.25)
    master['CLV_ESTIME'] = (
        prime_ann.fillna(0)
        * 3
        * (1 - master['TAUX_RESILIATION'].clip(0, 1).fillna(0.13))
        * (1 - master['SP_RATIO'].clip(0, 1).fillna(0))
        * (1 - master['TAUX_IMPAYE'].clip(0, 1).fillna(0))
    ).clip(0)

    # Only clients with polices
    master = master[master['NB_POLICES'] > 0].copy().reset_index(drop=True)
    print(f'Master: {len(master):,} clients, {master.shape[1]} cols')
    return master


def prep_features(master):
    feats = [c for c in CLUSTER_FEATURES if c in master.columns]
    df = master[feats].fillna(0).copy()

    # Clip BONUS_MALUS to [0.5, 3.5] — valid bonus-malus range
    if 'BONUS_MALUS_MOY' in df.columns:
        df['BONUS_MALUS_MOY'] = df['BONUS_MALUS_MOY'].clip(0.5, 3.5)

    # Cap monetary/count features at 99th percentile before log (save caps for inference)
    p99_caps = {}
    for col in feats:
        p99 = float(df[col].quantile(0.99))
        p99_caps[col] = p99
        if p99 > 0:
            df[col] = df[col].clip(upper=p99)

    # Log1p transform heavy-skew columns
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df, feats, p99_caps


def assign_names(cp: pd.DataFrame) -> dict:
    """
    Rank-based unique assignment — avoids fragile thresholds.

    Strategy (each label assigned at most once):
      VIP        → highest TOTAL_PRIMES
      Dormants   → highest RECENCY_DAYS (most inactive)
      Fidèles    → highest ANCIENNETE_JOURS among remaining (long-tenure active)
      Entreprises→ smallest remaining cluster (niche segment)
      Potentiels → fallback for the rest
    """
    remaining = list(cp.index)
    assigned: dict = {}

    # 1. VIP — richest clients
    vip = cp.loc[remaining, 'TOTAL_PRIMES'].idxmax()
    assigned[vip] = '🏆 VIP — Haute Valeur'
    remaining.remove(vip)

    # 2. Dormants — most inactive (highest recency)
    dorm = cp.loc[remaining, 'RECENCY_DAYS'].idxmax()
    assigned[dorm] = '😴 Clients Dormants'
    remaining.remove(dorm)

    # 3. Fidèles — most tenured among the rest
    anc_col = 'ANCIENNETE_JOURS' if 'ANCIENNETE_JOURS' in cp.columns else 'TOTAL_PRIMES'
    fid = cp.loc[remaining, anc_col].idxmax()
    assigned[fid] = '🎯 Fidèles Rentables'
    remaining.remove(fid)

    # 4. Entreprises — smallest remaining niche cluster (# of clients not in cp —
    #    use cluster SIZE stored as 'NB_CLIENTS' if available, else proxy with lowest primes
    #    among what's left)
    if 'NB_CLIENTS' in cp.columns:
        ent = cp.loc[remaining, 'NB_CLIENTS'].idxmin()
    else:
        ent = cp.loc[remaining, 'TOTAL_PRIMES'].idxmin()
    assigned[ent] = '🏢 Entreprises / Flottes'
    remaining.remove(ent)

    # 5. Potentiels — whatever is left
    for idx in remaining:
        assigned[idx] = '🌱 Clients Potentiels'

    return assigned


def build_radar(row_norm: pd.Series) -> list[dict]:
    result = []
    for label, col in RADAR_DIMS:
        val = float(row_norm.get(col, 0.5))
        if label in ('Fiabilité', 'Rentabilité'):
            val = 1.0 - val
        result.append({'metric': label, 'value': round(val, 3)})
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
master = build_master()
df_feats, used_feats, p99_caps = prep_features(master)

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(df_feats.fillna(0))
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f'\nFitting K-Means (k={K_OPTIMAL})...')
kmeans = KMeans(n_clusters=K_OPTIMAL, init='k-means++', n_init=20,
                max_iter=500, random_state=42)
master['CLUSTER'] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, master['CLUSTER'], sample_size=min(3000, len(master)))
db  = davies_bouldin_score(X_scaled, master['CLUSTER'])
ch  = calinski_harabasz_score(X_scaled, master['CLUSTER'])

print(f'\nMetrics (k={K_OPTIMAL}):')
print(f'  Silhouette       : {sil:.4f}')
print(f'  Davies-Bouldin   : {db:.4f}')
print(f'  Calinski-Harabasz: {ch:.0f}')
print('\nCluster sizes:')
sizes = master['CLUSTER'].value_counts().sort_index()
for c, n in sizes.items():
    print(f'  Cluster {c}: {n:,} ({n/len(master)*100:.1f}%)')

# Name clusters — rank-based unique assignment
cp               = master.groupby('CLUSTER')[used_feats].mean()
cp['NB_CLIENTS'] = master.groupby('CLUSTER').size()
cp_norm          = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)
cluster_colors = {
    '🏆 VIP — Haute Valeur':     '#f1c40f',
    '🎯 Fidèles Rentables':      '#2ecc71',
    '🌱 Clients Potentiels':     '#3498db',
    '⚠️ Clients à Risque':      '#e74c3c',
    '😴 Clients Dormants':       '#e67e22',
    '🏢 Entreprises / Flottes':  '#9b59b6',
}
cluster_names = assign_names(cp)
master['CLUSTER_NOM'] = master['CLUSTER'].map(cluster_names)

print('\nCluster names:')
for idx, name in sorted(cluster_names.items()):
    n = (master['CLUSTER'] == idx).sum()
    safe = name.encode('ascii', 'replace').decode('ascii')
    print(f'  {idx}: {safe} - {n:,} clients ({n/len(master)*100:.1f}%)')

# Build artifact
art = {
    'kmeans':           kmeans,
    'scaler':           scaler,
    'cluster_features': used_feats,
    'log_cols':         [c for c in LOG_COLS if c in used_feats],
    'p99_caps':         p99_caps,
    'ref_date':         REF_DATE,
    'cluster_names':    cluster_names,
    'cluster_colors':   cluster_colors,
    'action_plan':      ACTION_PLAN,
    'radar_dims':       RADAR_DIMS,
    'metrics': {
        'silhouette':        round(sil, 4),
        'davies_bouldin':    round(db, 4),
        'calinski_harabasz': round(ch, 0),
        'k':                 K_OPTIMAL,
        'n_clients':         len(master),
    },
    'source': 'fixed_script',
}

import pickle
with open(OUT_PKL, 'wb') as f:
    pickle.dump(art, f)
print(f'\nSaved to {OUT_PKL}')

# Verify segment quality
print('\nSegment summary:')
seg_summary = master.groupby('CLUSTER_NOM').agg(
    N=('ID_CLIENT','count'),
    PRIMES_MOY=('TOTAL_PRIMES','mean'),
    RECENCY_MOY=('RECENCY_DAYS','mean'),
    NB_SIN_MOY=('NB_SINISTRES','mean'),
    IMPAYE_MOY=('TAUX_IMPAYE','mean'),
    CLV_MOY=('CLV_ESTIME','mean'),
).sort_values('PRIMES_MOY', ascending=False)
print(seg_summary.round(1).to_string())
