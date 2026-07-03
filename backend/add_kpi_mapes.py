"""
Patch script: computes per-KPI 6-month holdout Prophet MAPE and injects it
into the existing forecast_model.pkl without re-running the full notebook.
Run with: .venv\Scripts\python.exe add_kpi_mapes.py
"""
import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

BASE   = r'C:\Users\LENOVO\Desktop\PFE_\maghrebia'
PKL    = os.path.join(BASE, 'backend', 'models', 'forecast_model.pkl')
RAW    = r'C:\Users\LENOVO\Desktop\PFE_\maghrebia\data\raw'

SINISTRE_KPIS = {'COUT_SINISTRES', 'NB_SINISTRES', 'SP_RATIO'}
HOLDOUT = 6

ALL_KPIS = [
    'PRIMES_ACQUISES', 'COUT_SINISTRES', 'NB_SINISTRES', 'NB_QUITTANCES',
    'TAUX_RESILIATION', 'SP_RATIO', 'MT_IMPAYE', 'MT_COMMISSION',
]

def load_csv(name, date_cols=None):
    path = os.path.join(RAW, f'DWH_{name}.csv')
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    if date_cols:
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def strip_trailing_zeros(ts):
    if ts.empty or not (ts > 0).any():
        return ts
    return ts.loc[:ts[ts > 0].index.max()]

print('Loading raw data...')
emission   = load_csv('FACT_EMISSION',   ['DATE_EMISSION'])
sinistre   = load_csv('FACT_SINISTRE',   ['DATE_SURVENANCE'])
annulation = load_csv('FACT_ANNULATION', ['DATE_ANNULATION'])
impaye     = load_csv('FACT_IMPAYE',     ['DATE_EMISSION'])

emission['PRIMES_ACQUISES'] = emission['MT_PTT'].fillna(emission['MT_PNET'].fillna(0))
emission['PERIODE']   = emission['DATE_EMISSION'].dt.to_period('M')
sinistre['PERIODE']   = sinistre['DATE_SURVENANCE'].dt.to_period('M')
annulation['PERIODE'] = annulation['DATE_ANNULATION'].dt.to_period('M')
impaye['PERIODE']     = impaye['DATE_EMISSION'].dt.to_period('M')

me = emission.groupby('PERIODE').agg(
    PRIMES_ACQUISES=('PRIMES_ACQUISES','sum'), NB_QUITTANCES=('NUM_QUITTANCE','count'),
    MT_COMMISSION=('MT_COMMISSION','sum')).reset_index()
ms = sinistre.groupby('PERIODE').agg(
    COUT_SINISTRES=('MT_PAYE','sum'), NB_SINISTRES=('NUM_SINISTRE','count')).reset_index()
ma = annulation.groupby('PERIODE').agg(NB_RESILIATIONS=('NUM_QUITTANCE','count')).reset_index()
mi = impaye.groupby('PERIODE').agg(MT_IMPAYE=('MT_PTT','sum')).reset_index()

kpi = me.merge(ms,on='PERIODE',how='left').merge(ma,on='PERIODE',how='left').merge(mi,on='PERIODE',how='left').fillna(0)
kpi['DATE']            = kpi['PERIODE'].dt.to_timestamp()
kpi['SP_RATIO']        = np.where(kpi['PRIMES_ACQUISES']>0, kpi['COUT_SINISTRES']/kpi['PRIMES_ACQUISES'], 0)
kpi['TAUX_RESILIATION']= np.where(kpi['NB_QUITTANCES']>0,  kpi['NB_RESILIATIONS']/kpi['NB_QUITTANCES'],   0)
kpi = kpi[kpi['DATE'] >= '2019-01-01'].reset_index(drop=True)
print(f'KPI table: {len(kpi)} rows ({kpi.DATE.min().date()} to {kpi.DATE.max().date()})')

print('\nComputing per-KPI Prophet holdout MAPE...')
kpi_mapes = {}
for kpi_col in ALL_KPIS:
    ts = kpi.set_index('DATE')[kpi_col].asfreq('MS').fillna(0)
    if kpi_col in SINISTRE_KPIS:
        ts = strip_trailing_zeros(ts)
    ts = ts[ts > 0]
    if len(ts) < HOLDOUT + 6:
        print(f'  ⚠  {kpi_col}: only {len(ts)} months — skipped')
        kpi_mapes[kpi_col] = None
        continue

    ts_tr = ts.iloc[:-HOLDOUT]
    ts_te = ts.iloc[-HOLDOUT:]
    mode  = 'multiplicative' if ts.min() > 0 else 'additive'
    df_tr = ts_tr.reset_index().rename(columns={'DATE':'ds', kpi_col:'y'})

    m = Prophet(seasonality_mode=mode, yearly_seasonality=True,
                weekly_seasonality=False, interval_width=0.90,
                changepoint_prior_scale=0.05)
    m.fit(df_tr)
    fut  = m.make_future_dataframe(periods=HOLDOUT, freq='MS')
    fc   = m.predict(fut).set_index('ds')['yhat'].iloc[-HOLDOUT:]
    idx  = ts_te.index.intersection(fc.index)
    mape = mean_absolute_percentage_error(ts_te.loc[idx], fc.loc[idx]) * 100
    kpi_mapes[kpi_col] = round(float(mape), 2)
    print(f'  ✅ {kpi_col}: MAPE = {mape:.1f}%')

print(f'\nkpi_mapes: {kpi_mapes}')

print(f'\nPatching pkl at {PKL}...')
art = joblib.load(PKL)
art['kpi_mapes'] = kpi_mapes
joblib.dump(art, PKL)
print('✅ pkl updated with kpi_mapes')
