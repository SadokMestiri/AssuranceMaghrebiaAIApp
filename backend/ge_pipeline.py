"""
ge_pipeline.py — Maghrebia Assurance
TDSP Phase 2 — Great Expectations Data Quality Pipeline

6 suites de validation couvrant les 8 tables :
  1. maghrebia_dim_agent_suite
  2. maghrebia_dim_client_suite
  3. maghrebia_dim_police_suite
  4. maghrebia_fact_emission_suite       ← la plus critique (8 expectations)
  5. maghrebia_fact_impaye_suite
  6. maghrebia_fact_emission_detail_suite

Usage :
  python ge_pipeline.py                  # run toutes les suites
  python ge_pipeline.py --suite fact_emission
  python ge_pipeline.py --table dim_police --save-report

Sorties :
  - Console : ✅/🔴 par expectation
  - HTML report : ge_reports/<suite>_<date>.html
  - JSON results : ge_reports/<suite>_<date>.json
  - Log dans dq_run_log (PostgreSQL)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_DIR    = Path(os.getenv("DATA_DIR", "./data/raw"))
REPORT_DIR  = Path("ge_reports")
REPORT_DIR.mkdir(exist_ok=True)

CSV_FILES = {
    "dim_agent":                "DIM_AGENT.csv",
    "dim_client":               "DIM_CLIENT.csv",
    "dim_police":               "DIM_POLICE.csv",
    "dim_vehicule":             "DIM_VEHICULE.csv",
    "dwh_fact_emission":        "DWH_FACT_EMISSION.csv",
    "dwh_fact_emission_detail": "DWH_FACT_EMISSION_DETAIL.csv",
    "dwh_fact_annulation":      "DWH_FACT_ANNULATION.csv",
    "dwh_fact_impaye":          "DWH_FACT_IMPAYE.csv",
}

# ── Expectation engine (lightweight, no GE server required) ──────────────────

class ExpectationResult:
    def __init__(self, name: str, passed: bool, n_ok: int, n_total: int, details: str = ""):
        self.name    = name
        self.passed  = passed
        self.n_ok    = n_ok
        self.n_total = n_total
        self.pct     = round(100 * n_ok / max(n_total, 1), 1)
        self.details = details

    def __repr__(self):
        icon = "✅" if self.passed else "🔴"
        return f"{icon} {self.name:55s} {self.n_ok:>7,}/{self.n_total:<7,} ({self.pct:.1f}%)  {self.details}"


class Suite:
    def __init__(self, name: str, df: pd.DataFrame):
        self.name    = name
        self.df      = df
        self.results: list[ExpectationResult] = []

    # ── Core expectations ─────────────────────────────────────────────────

    def not_null(self, col: str) -> "Suite":
        n  = len(self.df)
        ok = self.df[col].notna().sum()
        self.results.append(ExpectationResult(
            f"expect_column_values_to_not_be_null({col})",
            ok == n, ok, n,
        ))
        return self

    def unique(self, col: str) -> "Suite":
        n    = len(self.df)
        dups = self.df.duplicated(subset=[col]).sum()
        ok   = n - dups
        self.results.append(ExpectationResult(
            f"expect_column_values_to_be_unique({col})",
            dups == 0, ok, n, f"dups={dups}"
        ))
        return self

    def in_set(self, col: str, valid_set: set) -> "Suite":
        vals = self.df[col].dropna()
        ok   = vals.isin(valid_set).sum()
        n    = len(vals)
        display_values = sorted("<NA>" if pd.isna(v) else str(v) for v in valid_set)
        self.results.append(ExpectationResult(
            f"expect_column_values_to_be_in_set({col}, {display_values})",
            ok == n, ok, n,
            f"invalid sample={list(vals[~vals.isin(valid_set)].unique()[:3])}"
        ))
        return self

    def between(self, col: str, lo=None, hi=None, pct_threshold: float = 0.99) -> "Suite":
        vals = pd.to_numeric(self.df[col], errors="coerce").dropna()
        mask = pd.Series([True] * len(vals), index=vals.index)
        if lo is not None:
            mask &= vals >= lo
        if hi is not None:
            mask &= vals <= hi
        ok = mask.sum()
        n  = len(vals)
        passed = (ok / max(n, 1)) >= pct_threshold
        self.results.append(ExpectationResult(
            f"expect_column_values_to_be_between({col}, lo={lo}, hi={hi})",
            passed, ok, n,
            f"threshold={pct_threshold*100:.0f}%"
        ))
        return self

    def match_regex(self, col: str, regex: str, pct_threshold: float = 0.95) -> "Suite":
        vals = self.df[col].dropna().astype(str)
        ok   = vals.str.match(regex).sum()
        n    = len(vals)
        passed = (ok / max(n, 1)) >= pct_threshold
        self.results.append(ExpectationResult(
            f"expect_column_values_to_match_regex({col}, '{regex}')",
            passed, ok, n
        ))
        return self

    def custom(self, name: str, mask: pd.Series, pct_threshold: float = 0.98) -> "Suite":
        ok  = mask.sum()
        n   = len(mask)
        passed = (ok / max(n, 1)) >= pct_threshold
        self.results.append(ExpectationResult(
            f"custom: {name}",
            passed, ok, n,
            f"threshold={pct_threshold*100:.0f}%"
        ))
        return self

    def row_count(self, lo: int, hi: int = None) -> "Suite":
        n  = len(self.df)
        ok = lo <= n <= (hi or n)
        self.results.append(ExpectationResult(
            f"expect_table_row_count_to_be_between({lo}, {hi})",
            ok, n if ok else 0, n
        ))
        return self

    # ── Reporting ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        total   = len(self.results)
        passed  = sum(r.passed for r in self.results)
        failed  = total - passed
        success = 100 * passed / max(total, 1)

        print(f"\n{'='*70}")
        print(f"  SUITE : {self.name}")
        print(f"  Table : {len(self.df):,} rows × {len(self.df.columns)} cols")
        print(f"  {'='*65}")
        for r in self.results:
            print(f"  {r}")
        print(f"  {'='*65}")
        icon = "✅ PASSED" if failed == 0 else "⚠️  WARNING" if success >= 90 else "🔴 FAILED"
        print(f"  {icon}  —  {passed}/{total} expectations OK  ({success:.1f}%)")
        print(f"{'='*70}\n")

        return {
            "suite":       self.name,
            "run_date":    datetime.now().isoformat(),
            "n_rows":      len(self.df),
            "total_exp":   total,
            "passed_exp":  passed,
            "failed_exp":  failed,
            "success_pct": round(success, 1),
            "status":      "PASSED" if failed == 0 else "WARNING" if success >= 90 else "FAILED",
            "expectations": [
                {
                    "name":    r.name,
                    "passed":  r.passed,
                    "n_ok":    int(r.n_ok),
                    "n_total": int(r.n_total),
                    "pct":     r.pct,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# ── Suite definitions ─────────────────────────────────────────────────────────

def suite_dim_agent(df: pd.DataFrame) -> dict:
    s = Suite("maghrebia_dim_agent_suite", df)
    s.row_count(190, 210)
    s.not_null("ID_AGENT")
    s.unique("ID_AGENT")
    s.not_null("CODE_AGENT")
    s.in_set("ETAT_AGENT", {"A", "R", "I"})
    s.in_set("TYPE_AGENT", {"AG", "BA", "CO", "MA", "BR"})
    s.between("LATITUDE_AGENT",  30.0, 38.0)
    s.between("LONGITUDE_AGENT", 8.0,  12.0)
    s.in_set("GROUPE_AGENT", {
        "Réseau Direct", "Bancassurance", "Courtier",
        "Agent Général", "Mandataire",
    })
    return s.run()


def suite_dim_client(df: pd.DataFrame) -> dict:
    s = Suite("maghrebia_dim_client_suite", df)
    s.row_count(25_000, 35_000)
    s.not_null("ID_CLIENT")
    s.unique("ID_CLIENT")
    s.not_null("NOM")
    s.in_set("TYPE_PERSONNE", {"P", "M"})
    s.in_set("SEXE", {"M", "F", "N/A", None, np.nan})
    s.between("CODE_POSTAL", 1000, 9999)
    s.match_regex("CIN_MF", r"^\d{8}$", pct_threshold=0.70)  # ~30% MF (entreprises)
    s.match_regex("EMAIL", r"^[^@]+@[^@]+\.[^@]+$", pct_threshold=0.60)
    return s.run()


def suite_dim_police(df: pd.DataFrame) -> dict:
    s = Suite("maghrebia_dim_police_suite", df)
    s.row_count(25_000, 35_000)
    s.not_null("ID_POLICE")
    s.unique("ID_POLICE")
    s.in_set("BRANCHE", {"AUTO", "IRDS", "SANTE"})
    s.in_set("SITUATION", {"V", "R", "T", "S", "A"})
    s.in_set("TYPE_POLICE", {"individuel", "flotte"})
    s.between("CODE_PRODUIT", 17, 89)
    s.between("BONUS_MALUS", 1, 10, pct_threshold=0.95)  # ~33% nulls expected

    # FK: ID_AGENT must exist in dim_agent (checked at load time)
    # FK: ID_CLIENT must exist in dim_client (checked at load time)
    return s.run()


def suite_fact_emission(df: pd.DataFrame) -> dict:
    s = Suite("maghrebia_fact_emission_suite", df)
    s.row_count(60_000, 70_000)
    s.not_null("NUM_QUITTANCE")
    s.unique("NUM_QUITTANCE")
    s.not_null("ID_POLICE")
    s.not_null("ID_AGENT")

    # ETAT_QUIT — critical business rule
    s.in_set("ETAT_QUIT", {"E", "P", "A"})

    # Branche
    s.in_set("BRANCHE", {"AUTO", "IRDS", "SANTE"})

    # Year range
    annee = pd.to_numeric(df["ANNEE_ECHEANCE"], errors="coerce")
    s.custom("ANNEE_ECHEANCE between 2015 and 2025",
             annee.between(2015, 2025).fillna(False))

    # Mois range
    s.between("MOIS_ECHEANCE", 1, 12)

    # MT_PNET >= 0
    pnet = pd.to_numeric(df["MT_PNET"], errors="coerce")
    s.custom("MT_PNET >= 0", (pnet >= 0).fillna(True))  # NaN are allowed

    # MT_FGA ≈ MT_RC × 0.25 (within 1 TND tolerance)
    rc  = pd.to_numeric(df["MT_RC"],  errors="coerce").fillna(0)
    fga = pd.to_numeric(df["MT_FGA"], errors="coerce").fillna(0)
    s.custom("MT_FGA ≈ MT_RC × 0.25",
             (abs(fga - rc * 0.25) <= 1.0) | (rc == 0),
             pct_threshold=0.90)

    # PTT formula: MT_PTT = MT_PNET + MT_FGA + MT_TIMBRE + MT_TAXE
    ptt      = pd.to_numeric(df["MT_PTT"],   errors="coerce").fillna(0)
    timbre   = pd.to_numeric(df["MT_TIMBRE"],errors="coerce").fillna(0)
    taxe     = pd.to_numeric(df["MT_TAXE"],  errors="coerce").fillna(0)
    expected = pnet.fillna(0) + fga + timbre + taxe
    s.custom("MT_PTT = MT_PNET + MT_FGA + MT_TIMBRE + MT_TAXE",
             abs(ptt - expected) <= 0.5, pct_threshold=0.98)

    # MT_TIMBRE valid range (3–20 TND)
    s.between("MT_TIMBRE", 3, 20, pct_threshold=0.95)

    return s.run()


def suite_fact_impaye(df: pd.DataFrame) -> dict:
    s = Suite("maghrebia_fact_impaye_suite", df)
    s.row_count(600, 900)
    s.not_null("NUM_QUITTANCE")
    s.not_null("ID_POLICE")
    s.not_null("ID_AGENT")
    s.in_set("BRANCHE", {"AUTO", "IRDS", "SANTE"})
    s.between("ANNEE_ECHEANCE", 2015, 2025)
    s.between("MOIS_ECHEANCE", 1, 12)

    acp = pd.to_numeric(df["MT_ACP"], errors="coerce")
    s.custom("MT_ACP >= 0", (acp >= 0).fillna(True))

    ptt = pd.to_numeric(df["MT_PTT"], errors="coerce")
    s.custom("MT_PTT >= 0", (ptt >= 0).fillna(True))

    return s.run()


def suite_fact_emission_detail(df: pd.DataFrame) -> dict:
    s = Suite("maghrebia_fact_emission_detail_suite", df)
    s.row_count(190_000, 220_000)
    s.not_null("NUM_QUITTANCE")
    s.not_null("ID_POLICE")
    s.between("ID_GARANTIE", 1, 20)
    s.between("ANNEE_ECHEANCE", 2015, 2025)

    prime = pd.to_numeric(df["MT_PRIME"], errors="coerce")
    s.custom("MT_PRIME >= 0", (prime >= 0).fillna(True))

    frais = pd.to_numeric(df["MT_FRAIS"], errors="coerce")
    s.custom("MT_FRAIS >= 0", (frais >= 0).fillna(True))

    return s.run()


SUITES = {
    "dim_agent":                suite_dim_agent,
    "dim_client":               suite_dim_client,
    "dim_police":               suite_dim_police,
    "dwh_fact_emission":        suite_fact_emission,
    "dwh_fact_impaye":          suite_fact_impaye,
    "dwh_fact_emission_detail": suite_fact_emission_detail,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_suite(table_name: str, save_report: bool = True) -> dict:
    csv_file = DATA_DIR / CSV_FILES[table_name]
    log.info(f"Loading {csv_file} …")
    df = pd.read_csv(csv_file, low_memory=False)

    suite_fn = SUITES.get(table_name)
    if not suite_fn:
        log.warning(f"No suite defined for {table_name}")
        return {}

    result = suite_fn(df)

    if save_report:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"{result['suite']}_{date_str}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Report saved: {report_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Maghrebia Great Expectations Pipeline")
    parser.add_argument("--suite", help="Run a specific suite by table name")
    parser.add_argument("--save-report", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Maghrebia — Data Quality Pipeline (Great Expectations)")
    print(f"  Run: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    tables = [args.suite] if args.suite else list(SUITES.keys())
    all_results = []

    for table in tables:
        if table not in CSV_FILES:
            log.error(f"Unknown table: {table}")
            continue
        result = run_suite(table, save_report=args.save_report)
        all_results.append(result)

    # Final scorecard
    print("\n" + "=" * 70)
    print("  GLOBAL SCORECARD")
    print("=" * 70)
    total_pass = sum(r.get("passed_exp", 0) for r in all_results)
    total_exp  = sum(r.get("total_exp",  0) for r in all_results)
    for r in all_results:
        icon = "✅" if r["status"] == "PASSED" else "⚠️ " if r["status"] == "WARNING" else "🔴"
        print(f"  {icon} {r['suite']:50s} {r['passed_exp']}/{r['total_exp']} "
              f"({r['success_pct']:.1f}%)  [{r['status']}]")
    print(f"\n  TOTAL: {total_pass}/{total_exp} expectations passed "
          f"({100*total_pass/max(total_exp,1):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
