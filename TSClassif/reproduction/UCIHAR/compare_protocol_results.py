#!/usr/bin/env python3
"""Compare 20->9 seed-0 outputs from the two training protocols."""

import argparse
import csv
import json
from pathlib import Path


def _float(value):
    return float(value)


def compare_rows(paper, clean):
    best_delta = abs(
        _float(paper["best_clean_macro_f1"])
        - _float(clean["best_clean_macro_f1"])
    )
    last_delta = abs(
        _float(paper["last_clean_macro_f1"])
        - _float(clean["last_clean_macro_f1"])
    )
    return {
        "scenario": "20_to_9",
        "run_id": 0,
        "best_state_sha256_equal": (
            paper["best_state_sha256"] == clean["best_state_sha256"]
        ),
        "last_state_sha256_equal": (
            paper["last_state_sha256"] == clean["last_state_sha256"]
        ),
        "best_epoch_equal": paper["best_epoch"] == clean["best_epoch"],
        "paper_best_epoch": paper["best_epoch"],
        "clean_best_epoch": clean["best_epoch"],
        "best_clean_macro_f1_equal": best_delta <= 1e-12,
        "best_clean_macro_f1_difference": best_delta,
        "last_clean_macro_f1_difference": last_delta,
        "paper_best_clean_macro_f1": _float(paper["best_clean_macro_f1"]),
        "clean_best_clean_macro_f1": _float(clean["best_clean_macro_f1"]),
        "paper_last_clean_macro_f1": _float(paper["last_clean_macro_f1"]),
        "clean_last_clean_macro_f1": _float(clean["last_clean_macro_f1"]),
        "model_parameters_completely_equal": (
            paper["best_state_sha256"] == clean["best_state_sha256"]
            and paper["last_state_sha256"] == clean["last_state_sha256"]
        ),
    }


def read_single_row(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_rows(
        read_single_row(args.paper),
        read_single_row(args.clean),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

