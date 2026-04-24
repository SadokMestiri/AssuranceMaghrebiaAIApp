from __future__ import annotations

import argparse
import json

from ml_pipeline import train_and_persist_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train impaye risk model and persist artifact + metadata."
    )
    parser.add_argument("--year-from", type=int, default=None, help="Minimum annee_echeance")
    parser.add_argument("--year-to", type=int, default=None, help="Maximum annee_echeance")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="temporal",
        help="Split strategy: temporal or random",
    )
    parser.add_argument(
        "--no-promote-to-champion",
        action="store_true",
        help="Train challenger without promoting it to champion",
    )
    args = parser.parse_args()

    try:
        result = train_and_persist_model(
            year_from=args.year_from,
            year_to=args.year_to,
            test_size=args.test_size,
            random_state=args.random_state,
            split_strategy=args.split_strategy,
            promote_to_champion=not args.no_promote_to_champion,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 1

    print(json.dumps({"status": "ok", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())