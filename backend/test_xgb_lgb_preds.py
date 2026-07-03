import pickle
import pandas as pd
import numpy as np

with open('/app/models/risk_scoring_model.pkl', 'rb') as f:
    art = pickle.load(f)

features = art['features']
defaults = art['feature_defaults']

# Test varying age client and age vehicule
for age_c, age_v in [(20, 15), (40, 5), (65, 1)]:
    row = {c: defaults.get(c, 0.0) for c in features}
    row['AGE_CLIENT'] = age_c
    row['AGE_VEHICULE'] = age_v
    # Derived LOG_VALEUR_VEH to match
    import math
    row['LOG_VALEUR_VEH'] = math.log1p(10000)
    row['LOG_VALEUR_NEUF'] = math.log1p(13000)
    
    df_row = pd.DataFrame([row], columns=features)
    lgb_f = float(art['lgb_freq'].predict(df_row)[0])
    xgb_f = float(art['xgb_freq'].predict(df_row)[0])
    lgb_s = float(art['lgb_sev'].predict(df_row)[0])
    xgb_s = float(art['xgb_sev'].predict(df_row)[0])
    
    print(f"AgeClient={age_c}, AgeVeh={age_v} | lgb_freq={lgb_f:.3f} | xgb_freq={xgb_f:.3f} | lgb_sev={lgb_s:.3f} | xgb_sev={xgb_s:.3f}")
