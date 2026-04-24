"""
Maghrebia Assurance — Great Expectations Suite
TDSP Phase 2 — Data Quality Validation
Exécuter APRÈS import_data.py
"""

# ═══════════════════════════════════════════════════════════
# NOTE : Ce script utilise la logique GE v0.18 en mode
# "Checkpoint" sur les CSV bruts. Adapter le chemin DATA_DIR.
# ═══════════════════════════════════════════════════════════

import pandas as pd
import os, sys, json
from datetime import datetime

DATA_DIR = os.environ.get("DATA_DIR", "../data/raw")
REPORT_DIR = os.environ.get("REPORT_DIR", "../reports/data_quality")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_csv(pattern):
    for f in os.listdir(DATA_DIR):
        if pattern.lower() in f.lower() and f.endswith(".csv"):
            return pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
    raise FileNotFoundError(f"CSV contenant '{pattern}' introuvable dans {DATA_DIR}")


def run_suite(df: pd.DataFrame, suite_name: str, expectations: list) -> dict:
    """Exécute une suite d'expectations sur un DataFrame."""
    results = []
    for exp in expectations:
        func = exp["func"]
        kwargs = exp.get("kwargs", {})
        desc = exp["desc"]
        try:
            result = func(df, **kwargs)
            results.append({"rule": desc, "passed": result["passed"], "details": result.get("details", "")})
        except Exception as e:
            results.append({"rule": desc, "passed": False, "details": str(e)})

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"])
    print(f"\n{'='*60}")
    print(f"Suite: {suite_name}")
    print(f"  ✅ {n_pass} règles validées | ❌ {n_fail} règles échouées")
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} {r['rule']}")
        if not r["passed"]:
            print(f"      → {r['details']}")
    return {"suite": suite_name, "passed": n_pass, "failed": n_fail, "results": results}


# ─── Helpers ─────────────────────────────────────────────

def expect_values_in_set(df, column, valid_set):
    mask = df[column].isin(valid_set)
    n_fail = (~mask).sum()
    vals = df[~mask][column].value_counts().head(5).to_dict()
    return {"passed": n_fail == 0,
            "details": f"{n_fail} valeurs invalides → {vals}"}


def expect_no_nulls(df, column):
    n = df[column].isnull().sum()
    return {"passed": n == 0, "details": f"{n} nulls"}


def expect_between(df, column, min_val=None, max_val=None):
    s = df[column].dropna()
    fails = 0
    if min_val is not None: fails += (s < min_val).sum()
    if max_val is not None: fails += (s > max_val).sum()
    return {"passed": fails == 0, "details": f"{fails} valeurs hors [{min_val}, {max_val}]"}


def expect_formula(df, result_col, formula_cols, operation="sum", tolerance=0.5):
    if operation == "sum":
        computed = df[formula_cols].sum(axis=1)
    diff = (computed - df[result_col]).abs()
    n_fail = (diff > tolerance).sum()
    return {"passed": n_fail == 0,
            "details": f"{n_fail} lignes hors tolérance (écart max={diff.max():.2f})"}


def expect_fk(df, fk_col, ref_df, ref_col):
    ref_ids = set(ref_df[ref_col].dropna())
    orphans = (~df[fk_col].isin(ref_ids)).sum()
    return {"passed": orphans == 0, "details": f"{orphans} clés orphelines"}


def expect_unique(df, column):
    n_dup = df[column].duplicated().sum()
    return {"passed": n_dup == 0, "details": f"{n_dup} doublons"}


def expect_null_pct_below(df, column, threshold):
    pct = df[column].isnull().mean()
    return {"passed": pct <= threshold,
            "details": f"{pct:.1%} nulls (seuil={threshold:.0%})"}


def expect_positive(df, column):
    n = (df[column].dropna() < 0).sum()
    return {"passed": n == 0, "details": f"{n} valeurs négatives"}


# ─── SUITES ──────────────────────────────────────────────

def suite_fact_emission(fe, dim_police, dim_agent):
    return run_suite(fe, "FACT_EMISSION", [
        {"func": expect_no_nulls,      "kwargs": {"column": "NUM_QUITTANCE"}, "desc": "NUM_QUITTANCE sans null"},
        {"func": expect_unique,        "kwargs": {"column": "NUM_QUITTANCE"}, "desc": "NUM_QUITTANCE unique (PK)"},
        {"func": expect_values_in_set, "kwargs": {"column": "ETAT_QUIT", "valid_set": {"E","P","A"}},
         "desc": "ETAT_QUIT ∈ {E, P, A}"},
        {"func": expect_values_in_set, "kwargs": {"column": "BRANCHE", "valid_set": {"AUTO","IRDS","SANTE"}},
         "desc": "BRANCHE ∈ {AUTO, IRDS, SANTE}"},
        {"func": expect_between,       "kwargs": {"column": "ANNEE_ECHEANCE", "min_val": 2018, "max_val": 2025},
         "desc": "ANNEE_ECHEANCE ∈ [2018, 2025]"},
        {"func": expect_positive,      "kwargs": {"column": "MT_PNET"}, "desc": "MT_PNET ≥ 0"},
        {"func": expect_positive,      "kwargs": {"column": "MT_PTT"},  "desc": "MT_PTT ≥ 0"},
        {"func": expect_formula,
         "kwargs": {"result_col": "MT_PTT",
                    "formula_cols": ["MT_PNET","MT_FGA","MT_TIMBRE","MT_TAXE"],
                    "tolerance": 0.5},
         "desc": "MT_PTT ≈ MT_PNET + MT_FGA + MT_TIMBRE + MT_TAXE (±0.5)"},
        {"func": expect_fk,            "kwargs": {"fk_col": "ID_POLICE", "ref_df": dim_police, "ref_col": "ID_POLICE"},
         "desc": "ID_POLICE FK valide → DIM_POLICE"},
        {"func": expect_fk,            "kwargs": {"fk_col": "ID_AGENT",  "ref_df": dim_agent,  "ref_col": "ID_AGENT"},
         "desc": "ID_AGENT FK valide → DIM_AGENT"},
        {"func": expect_values_in_set, "kwargs": {"column": "PERIODICITE", "valid_set": {"A","S","T","C"}},
         "desc": "PERIODICITE ∈ {A, S, T, C}"},
    ])


def suite_dim_police(dp, dim_agent, dim_client):
    return run_suite(dp, "DIM_POLICE", [
        {"func": expect_no_nulls,      "kwargs": {"column": "ID_POLICE"}, "desc": "ID_POLICE sans null"},
        {"func": expect_unique,        "kwargs": {"column": "ID_POLICE"}, "desc": "ID_POLICE unique (PK)"},
        {"func": expect_values_in_set, "kwargs": {"column": "SITUATION", "valid_set": {"V","R","T","S","A"}},
         "desc": "SITUATION ∈ {V, R, T, S, A}"},
        {"func": expect_values_in_set, "kwargs": {"column": "BRANCHE", "valid_set": {"AUTO","IRDS","SANTE"}},
         "desc": "BRANCHE ∈ {AUTO, IRDS, SANTE}"},
        {"func": expect_between,       "kwargs": {"column": "BONUS_MALUS", "min_val": 0.5, "max_val": 3.5},
         "desc": "BONUS_MALUS ∈ [0.5, 3.5]"},
        {"func": expect_fk,            "kwargs": {"fk_col": "ID_AGENT",  "ref_df": dim_agent,  "ref_col": "ID_AGENT"},
         "desc": "ID_AGENT FK valide → DIM_AGENT"},
        {"func": expect_fk,            "kwargs": {"fk_col": "ID_CLIENT", "ref_df": dim_client, "ref_col": "ID_CLIENT"},
         "desc": "ID_CLIENT FK valide → DIM_CLIENT"},
    ])


def suite_dim_client(dc):
    return run_suite(dc, "DIM_CLIENT", [
        {"func": expect_no_nulls,      "kwargs": {"column": "ID_CLIENT"}, "desc": "ID_CLIENT sans null"},
        {"func": expect_unique,        "kwargs": {"column": "ID_CLIENT"}, "desc": "ID_CLIENT unique (PK)"},
        {"func": expect_values_in_set, "kwargs": {"column": "TYPE_PERSONNE", "valid_set": {"P","M"}},
         "desc": "TYPE_PERSONNE ∈ {P, M} (après normalisation)"},
        {"func": expect_no_nulls,      "kwargs": {"column": "DATE_NAISSANCE"}, 
         "desc": "DATE_NAISSANCE sans null (0%)"},
        {"func": expect_values_in_set, "kwargs": {"column": "SEXE", "valid_set": {"M","F",None}},
         "desc": "SEXE ∈ {M, F} ou null"},
    ])


def suite_fact_impaye(fi, dim_police, dim_agent):
    return run_suite(fi, "FACT_IMPAYE", [
        {"func": expect_no_nulls,      "kwargs": {"column": "NUM_QUITTANCE"}, "desc": "NUM_QUITTANCE sans null"},
        {"func": expect_positive,      "kwargs": {"column": "MT_PTT"}, "desc": "MT_PTT ≥ 0"},
        {"func": expect_fk,            "kwargs": {"fk_col": "ID_POLICE", "ref_df": dim_police, "ref_col": "ID_POLICE"},
         "desc": "ID_POLICE FK valide → DIM_POLICE"},
        {"func": expect_fk,            "kwargs": {"fk_col": "ID_AGENT",  "ref_df": dim_agent,  "ref_col": "ID_AGENT"},
         "desc": "ID_AGENT FK valide → DIM_AGENT"},
    ])


# ─── MAIN ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MAGHREBIA ASSURANCE — GREAT EXPECTATIONS VALIDATION")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    fe = load_csv("FACT_EMISSION.csv".split(".")[0])
    dp = load_csv("DIM_POLICE")
    da = load_csv("DIM_AGENT")
    dc = load_csv("DIM_CLIENT")
    fi = load_csv("FACT_IMPAYE")

    # Normalisation légère avant validation (ce que Phase 2 fait)
    fe["ETAT_QUIT"] = fe["ETAT_QUIT"].str.strip().str.upper()
    dc["TYPE_PERSONNE"] = dc["TYPE_PERSONNE"].str.strip().str.upper()

    all_results = []
    all_results.append(suite_fact_emission(fe, dp, da))
    all_results.append(suite_dim_police(dp, da, dc))
    all_results.append(suite_dim_client(dc))
    all_results.append(suite_fact_impaye(fi, dp, da))

    total_pass = sum(r["passed"] for r in all_results)
    total_fail = sum(r["failed"] for r in all_results)
    total_rules = total_pass + total_fail
    score = total_pass / total_rules * 100

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ GLOBAL — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Règles validées : {total_pass} / {total_rules}")
    print(f"  Score qualité   : {score:.1f}%")
    print(f"  Statut          : {'✅ PASS' if score >= 80 else '⚠️  ATTENTION' if score >= 60 else '❌ FAIL'}")
    print(f"{'='*60}")

    # Sauvegarde rapport JSON
    report = {
        "run_date": datetime.now().isoformat(),
        "global_score": round(score, 1),
        "total_rules": total_rules,
        "passed": total_pass,
        "failed": total_fail,
        "suites": all_results,
    }
    report_path = os.path.join(REPORT_DIR, f"ge_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Rapport JSON sauvegardé : {report_path}")

    sys.exit(0 if score >= 80 else 1)
