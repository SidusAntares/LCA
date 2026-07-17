#!/usr/bin/env python3
"""Return zero only when every requested run has a usable checkpoint and result."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compat import load_torch_file
from run_config import parse_run_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run-ids", default="0")
    args = parser.parse_args()
    src, tgt = [part.strip() for part in args.scenario.split(",")]
    scenario = f"{src}_to_{tgt}"
    result_path = args.experiment_dir / "results.csv"
    if not result_path.is_file():
        return 1
    with result_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for run_id in parse_run_ids(args.run_ids):
        checkpoint = args.experiment_dir / f"{scenario}_run_{run_id}" / "checkpoint.pt"
        row_ok = any(
            row.get("scenario") == scenario
            and row.get("run") == str(run_id)
            and row.get("status") == "success"
            for row in rows
        )
        if not checkpoint.is_file() or not row_ok:
            return 1
        try:
            state = load_torch_file(checkpoint)
            if not isinstance(state, dict) or state.get("last") is None or state.get("best") is None:
                return 1
        except Exception:
            return 1
    print("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

