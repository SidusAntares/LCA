#!/usr/bin/env python3
"""Merge independently produced HAR result files using only the standard library."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


FIELDS = [
    "scenario", "run", "acc", "f1_score", "auroc", "status",
    "runtime_seconds", "peak_gpu_memory_mb", "checkpoint_path",
    "git_commit", "python_version", "torch_version", "cuda_version",
]
METRICS = ["acc", "f1_score", "auroc", "runtime_seconds", "peak_gpu_memory_mb"]
OFFICIAL_SCENARIOS = [
    "18_to_14", "6_to_13", "20_to_9", "7_to_18", "19_to_11",
    "17_to_18", "9_to_19", "2_to_12", "12_to_3", "17_to_14",
]


def numeric_values(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summary_row(name, run, status, rows, std=False):
    result = {field: "" for field in FIELDS}
    result.update({"scenario": name, "run": run, "status": status})
    for metric in METRICS:
        values = numeric_values(rows, metric)
        if not values:
            continue
        result[metric] = statistics.stdev(values) if std and len(values) > 1 else (
            0.0 if std else statistics.mean(values)
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("LCA_all_result"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("LCA_all_result") / "HAR" / "results_merged.csv",
    )
    parser.add_argument("--expected-scenarios", nargs="*", default=OFFICIAL_SCENARIOS)
    parser.add_argument("--expected-run-ids", default="0")
    args = parser.parse_args()
    output_resolved = args.output.resolve()
    rows = []
    seen = set()
    for path in sorted(args.input_root.rglob("results.csv")):
        if path.resolve() == output_resolved:
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                scenario = raw.get("scenario", "")
                run = raw.get("run", "")
                if scenario in {"mean", "std"} or run in {"mean", "std", "-"}:
                    continue
                row = {field: raw.get(field, "") for field in FIELDS}
                row["status"] = row["status"] or "success"
                row["checkpoint_path"] = row["checkpoint_path"] or str(
                    path.parent / f"{scenario}_run_{run}" / "checkpoint.pt"
                )
                if row["status"] == "success" and not Path(row["checkpoint_path"]).is_file():
                    row["status"] = "missing_checkpoint"
                key = (scenario, run, row["checkpoint_path"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

    indexed = {(row["scenario"], str(row["run"])): row for row in rows}
    for path in sorted(args.input_root.rglob("failed_runs.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    failure = json.loads(line)
                    scenario = str(failure["scenario"])
                    run = str(failure["run"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                key = (scenario, run)
                if key not in indexed or indexed[key]["status"] != "success":
                    failed_row = {field: "" for field in FIELDS}
                    failed_row.update({"scenario": scenario, "run": run, "status": "failed"})
                    indexed[key] = failed_row

    try:
        expected_runs = [str(int(value.strip())) for value in args.expected_run_ids.split(",")]
    except ValueError as exc:
        parser.error(f"invalid --expected-run-ids: {exc}")
    for scenario in args.expected_scenarios:
        for run in expected_runs:
            key = (scenario, run)
            if key not in indexed:
                missing_row = {field: "" for field in FIELDS}
                missing_row.update({"scenario": scenario, "run": run, "status": "missing"})
                indexed[key] = missing_row

    rows = sorted(indexed.values(), key=lambda row: (row["scenario"], str(row["run"])))

    successful = [row for row in rows if row["status"] == "success"]
    grouped = {}
    for row in successful:
        grouped.setdefault(row["scenario"], []).append(row)
    summaries = []
    for scenario, scenario_rows in sorted(grouped.items()):
        summaries.append(summary_row(scenario, "mean", "summary_mean", scenario_rows))
        summaries.append(summary_row(scenario, "std", "summary_std", scenario_rows, std=True))
    if successful:
        summaries.append(summary_row("ALL", "mean", "summary_mean", successful))
        summaries.append(summary_row("ALL", "std", "summary_std", successful, std=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows + summaries)
    incomplete = [row for row in rows if row["status"] != "success"]
    print(
        f"merged {len(rows)} run rows from {len(grouped)} successful scenarios into {args.output}; "
        f"incomplete={len(incomplete)}"
    )
    return 0 if rows and not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
