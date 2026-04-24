import os

file_path = r"c:\Users\LENOVO\Desktop\PFE_\maghrebia\dags\import_data.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add Extract load_csv
extract_old = """        raw["FACT_IMPAYE"]     = load_csv("FACT_IMPAYE")
        raw["FACT_EMISSION_DETAIL"] = load_csv("FACT_EMISSION_DETAIL")"""
extract_new = """        raw["FACT_IMPAYE"]     = load_csv("FACT_IMPAYE")
        raw["FACT_EMISSION_DETAIL"] = load_csv("FACT_EMISSION_DETAIL")
        raw["FACT_SINISTRE"]   = load_csv("FACT_SINISTRE")"""
content = content.replace(extract_old, extract_new)

# 2. Add Cleaning function
cleaning_func = """def clean_fact_sinistre(df_raw: pd.DataFrame,
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
"""
if "def clean_fact_sinistre" not in content:
    content = content.replace("# ─────────────────────────────────────────────────────────────────────────────\n# VALIDATION", cleaning_func + "# VALIDATION")

# 3. Call cleaning func
call_old = """    log.info("  → FACT_IMPAYE")
    cleaned["FACT_IMPAYE"], s = clean_fact_impaye(
        raw["FACT_IMPAYE"], cleaned["DIM_POLICE"], cleaned["DIM_AGENT"]
    )
    all_stats.append(s.summary())"""
call_new = """    log.info("  → FACT_IMPAYE")
    cleaned["FACT_IMPAYE"], s = clean_fact_impaye(
        raw["FACT_IMPAYE"], cleaned["DIM_POLICE"], cleaned["DIM_AGENT"]
    )
    all_stats.append(s.summary())

    log.info("  → FACT_SINISTRE")
    cleaned["FACT_SINISTRE"], s = clean_fact_sinistre(
        raw["FACT_SINISTRE"], cleaned["DIM_POLICE"], cleaned["DIM_CLIENT"], 
        cleaned["DIM_AGENT"], cleaned["DIM_VEHICULE"]
    )
    all_stats.append(s.summary())"""
if "log.info(\"  → FACT_SINISTRE\")" not in content:
    content = content.replace(call_old, call_new)

# 4. Target load string
load_old = """            if not target_table or target_table == "FACT_IMPAYE":
                load_table_append(engine, cleaned["FACT_IMPAYE"], "dwh_fact_impaye")"""
load_new = """            if not target_table or target_table == "FACT_IMPAYE":
                load_table_append(engine, cleaned["FACT_IMPAYE"], "dwh_fact_impaye")
            if not target_table or target_table == "FACT_SINISTRE":
                load_table(engine, cleaned["FACT_SINISTRE"], "dwh_fact_sinistre", "NUM_SINISTRE")"""
if "load_table(engine, cleaned[\"FACT_SINISTRE\"]" not in content:
    content = content.replace(load_old, load_new)
    
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("import_data.py patched successfully!")