"""
TDSP import wrapper.

This backend entrypoint delegates all data loading to the strict TDSP script
so downstream services (API, KPIs, drift, alerts) consume cleaned data only.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_database_url_from_env() -> str:
    user = os.environ.get("POSTGRES_USER", "maghrebia")
    password = os.environ.get("POSTGRES_PASSWORD", "maghrebia")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "maghrebia")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _find_tdsp_import_script() -> Path:
    env_script = os.environ.get("TDSP_IMPORT_SCRIPT")
    candidates: list[Path] = []

    if env_script:
        candidates.append(Path(env_script))

    candidates.extend(
        [
            Path("/app/tdsp_dags/import_data.py"),
            Path(__file__).resolve().parents[1] / "dags" / "import_data.py",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"TDSP import_data.py introuvable. Tried: {tried}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strict TDSP cleaning + validation + loading pipeline"
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without DB writes")
    parser.add_argument("--table", type=str, default=None, help="Load only one table")
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip post-clean validation (not recommended)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override DATA_DIR")
    args, passthrough = parser.parse_known_args()

    tdsp_script = _find_tdsp_import_script()

    env = os.environ.copy()

    if args.data_dir:
        env["DATA_DIR"] = args.data_dir
    env.setdefault("DATA_DIR", "./data/raw")

    data_dir = Path(env["DATA_DIR"])
    env.setdefault("CLEAN_DIR", str(data_dir.parent / "clean"))
    env.setdefault("REPORT_DIR", "./reports/data_quality")
    env.setdefault("DATABASE_URL", _build_database_url_from_env())

    cmd = [sys.executable, str(tdsp_script)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.table:
        cmd.extend(["--table", args.table])
    if args.skip_validate:
        cmd.append("--skip-validate")
    if passthrough:
        cmd.extend(passthrough)

    completed = subprocess.run(cmd, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
