#!/usr/bin/env python3
"""Merge formal clean evaluations and enforce the UCIHAR stop gate."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PAPER_RESULTS = {
    "18_to_14": 1.0000,
    "6_to_13": 1.0000,
    "20_to_9": 0.6946,
    "7_to_18": 0.9108,
    "19_to_11": 0.9963,
    "17_to_18": 0.9601,
    "9_to_19": 0.9692,
    "2_to_12": 1.0000,
    "12_to_3": 1.0000,
    "17_to_14": 0.9673,
}
PAPER_ORDER = list(PAPER_RESULTS)
DIAGNOSTIC_SCENARIOS = ["20_to_9", "7_to_18", "9_to_19"]


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values):
    return [value for value in values if math.isfinite(value)]


def is_legacy(row):
    return row.get("is_legacy", "false").lower() in ("true", "1", "yes")


def read_rows(paths):
    rows_by_key = {}
    fieldnames = None
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                key = (row.get("scenario"), row.get("run_id"), is_legacy(row))
                if key in rows_by_key:
                    raise ValueError(f"duplicate run row: {key}")
                rows_by_key[key] = row
    if not fieldnames:
        raise ValueError("no input rows")
    order = {scenario: index for index, scenario in enumerate(PAPER_ORDER)}
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            order.get(row.get("scenario"), len(order)),
            int(row.get("run_id", -1)),
        ),
    )
    return fieldnames, rows


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(output_dir, rows, summary_rows, decision):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    summary_by_scenario = {row["scenario"]: row for row in summary_rows}
    expected_scenarios = (
        DIAGNOSTIC_SCENARIOS if decision["phase"] == "diagnostic" else PAPER_ORDER
    )
    full_complete = decision["completed_runs"] == 30 and len(summary_rows) == 10

    if not full_complete:
        conclusion = "完整 Table 5 复现尚未完成"
    else:
        overall_gap = decision["overall_absolute_difference"]
        within_004 = sum(
            as_float(row["absolute_difference"]) <= 0.04 for row in summary_rows
        )
        hard_gap = as_float(
            summary_by_scenario.get("20_to_9", {}).get("absolute_difference")
        )
        if (
            overall_gap <= 0.01
            and decision["tasks_with_error_le_0.02"] >= 8
            and not decision["stop"]
        ):
            conclusion = "强复现"
        elif (
            overall_gap <= 0.02
            and within_004 >= 7
            and hard_gap <= 0.15
            and not decision["stop"]
        ):
            conclusion = "可接受复现"
        elif overall_gap <= 0.05:
            conclusion = "部分复现"
        else:
            conclusion = "复现失败"

    seed_lines = []
    for scenario in expected_scenarios:
        scenario_rows = sorted(
            grouped.get(scenario, []), key=lambda row: int(row["run_id"])
        )
        values = ", ".join(
            f"seed {row['run_id']}={as_float(row['best_clean_macro_f1']):.4f}"
            for row in scenario_rows
        )
        seed_lines.append(f"- {scenario.replace('_to_', '→')}: {values or 'pending'}")

    comparison = [
        "| 任务 | 论文 F1 | clean mean | std | 绝对差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["scenario"] not in expected_scenarios:
            continue
        comparison.append(
            f"| {row['scenario'].replace('_to_', '→')} | "
            f"{as_float(row['paper_f1']):.4f} | "
            f"{as_float(row['reproduced_mean_f1']):.4f} | "
            f"{as_float(row['reproduced_std_f1']):.4f} | "
            f"{as_float(row['absolute_difference']):.4f} |"
        )

    reasons = ", ".join(decision["reasons"]) or "未触发"
    warnings = ", ".join(decision["warnings"]) or "无"
    report = f"""# LCA UCIHAR Table 5 Reproduction Report

## 当前结论

- 状态：{conclusion}
- 阶段：{decision['phase']}
- 完成运行：{decision['completed_runs']}/{decision['expected_runs']}
- 停止门禁：{reasons}
- 警告：{warnings}
- 本阶段论文均值：{decision['paper_overall_mean']}
- 本阶段复现均值：{decision['reproduced_overall_mean']}
- 同任务集合总体绝对差：{decision['overall_absolute_difference']}
- 逐任务平均绝对误差：{decision['mean_absolute_error']}

诊断阶段只计算 20→9、7→18、9→19，论文均值为 0.8582。只有 full 阶段才计算十任务论文均值 0.94983。

## 协议审计结论

`paper_code_protocol` 明确复现公开代码：每个 epoch 结束时读取 target test 并调用评价；评价函数调用 `eval()`，公开循环随后没有恢复 `train()`。因此从第二个 epoch 起，训练在 eval 模式下继续，仓库中的 Dropout 与 BatchNorm 行为会改变。

`baseline_clean_protocol` 不在训练期间读取 target test 标签，每个 epoch 显式调用 `train()`，训练完成后才分别对 best/last checkpoint 做一次无状态 clean 评价。20→9 seed 0 的双协议比较完成前，不能假定二者参数相同。

评价输出根据运行元数据填写来源：只有 `metric_protocol=official_stateful_no_reset` 才填写 `official_reported_f1`；其他协议填写 `current_reported_f1`。`get_features_returns_z`、`metric_reset_fixed` 和代码指纹来自运行时审计，不是常量。审计不满足 corrected public implementation 时评价直接失败。

正式 18→14 必须由冻结后同一协议重新产生 seed 0/1/2。冻结前 checkpoint 只能单独写入 `legacy_18_to_14_run0.csv`，且不得进入 formal CSV、completed runs 或复现结论。

## 每个 seed 的 clean macro-F1

{chr(10).join(seed_lines)}

## 论文对照

{chr(10).join(comparison)}

## 门禁说明

阻断条件包括：运行不完整、clean F1 非有限、20→9 差异超过 0.15、诊断三任务平均绝对误差超过 0.10、best/last 异常分离，或使用 target 真实标签选 checkpoint。official 与 clean F1 差异超过 0.05 仅记录 warning，不单独阻断完整实验。
"""
    (Path(output_dir) / "REPRODUCTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def summarize(input_paths, output_dir, diagnostic=False, target_label_selection=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_fields, all_rows = read_rows(input_paths)
    rows = [row for row in all_rows if not is_legacy(row)]
    write_csv(output_dir / "raw_runs.csv", raw_fields, rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)

    summary_rows = []
    for scenario in PAPER_ORDER:
        scenario_rows = grouped.get(scenario)
        if not scenario_rows:
            continue
        values = finite(
            [as_float(row.get("best_clean_macro_f1")) for row in scenario_rows]
        )
        reproduced_mean = statistics.mean(values) if values else float("nan")
        reproduced_std = statistics.stdev(values) if len(values) > 1 else float("nan")
        paper_f1 = PAPER_RESULTS[scenario]
        summary_rows.append({
            "scenario": scenario,
            "paper_f1": paper_f1,
            "reproduced_mean_f1": reproduced_mean,
            "reproduced_std_f1": reproduced_std,
            "absolute_difference": abs(reproduced_mean - paper_f1),
            "best_seed_f1": max(values) if values else float("nan"),
            "worst_seed_f1": min(values) if values else float("nan"),
        })

    summary_fields = [
        "scenario", "paper_f1", "reproduced_mean_f1", "reproduced_std_f1",
        "absolute_difference", "best_seed_f1", "worst_seed_f1",
    ]
    write_csv(output_dir / "scenario_summary.csv", summary_fields, summary_rows)
    write_csv(output_dir / "paper_comparison.csv", summary_fields, summary_rows)

    reasons = []
    warnings = []
    expected = DIAGNOSTIC_SCENARIOS if diagnostic else PAPER_ORDER
    expected_runs = len(expected) * 3
    completed = [
        row for row in rows
        if row.get("scenario") in expected
        and row.get("status") == "success"
        and math.isfinite(as_float(row.get("best_clean_macro_f1")))
    ]
    if len(completed) != expected_runs:
        reasons.append("incomplete_runs")

    all_nan_tasks = [
        scenario for scenario in expected
        if scenario in grouped and not finite([
            as_float(row.get("best_clean_macro_f1")) for row in grouped[scenario]
        ])
    ]
    if all_nan_tasks:
        reasons.append("all_nan_task")

    summary_by_scenario = {row["scenario"]: row for row in summary_rows}
    if (
        "20_to_9" in summary_by_scenario
        and as_float(summary_by_scenario["20_to_9"]["absolute_difference"]) > 0.15
    ):
        reasons.append("20_to_9_difference_over_0.15")

    diagnostic_errors = finite([
        as_float(summary_by_scenario[scenario]["absolute_difference"])
        for scenario in DIAGNOSTIC_SCENARIOS if scenario in summary_by_scenario
    ])
    if (
        diagnostic and len(diagnostic_errors) == 3
        and statistics.mean(diagnostic_errors) > 0.10
    ):
        reasons.append("diagnostic_mean_absolute_error_over_0.10")

    best_last_gaps = finite([
        abs(as_float(row.get("best_clean_macro_f1"))
            - as_float(row.get("last_clean_macro_f1")))
        for row in rows
    ])
    if any(gap > 0.15 for gap in best_last_gaps):
        reasons.append("best_last_gap_over_0.15")

    official_clean_gaps = finite([
        abs(as_float(row.get("official_reported_f1"))
            - as_float(row.get("best_clean_macro_f1")))
        for row in rows
    ])
    if any(gap > 0.05 for gap in official_clean_gaps):
        warnings.append("official_clean_gap_over_0.05")
    if target_label_selection:
        reasons.append("target_label_used_for_selection")

    selected = [row for row in summary_rows if row["scenario"] in expected]
    reproduced_means = finite([as_float(row["reproduced_mean_f1"]) for row in selected])
    absolute_differences = finite([as_float(row["absolute_difference"]) for row in selected])
    scenario_stds = finite([as_float(row["reproduced_std_f1"]) for row in selected])
    paper_overall_mean = statistics.mean(PAPER_RESULTS[item] for item in expected)
    decision = {
        "phase": "diagnostic" if diagnostic else "full",
        "stop": bool(reasons),
        "reasons": reasons,
        "warnings": warnings,
        "completed_runs": len(completed),
        "expected_runs": expected_runs,
        "reproduced_overall_mean": statistics.mean(reproduced_means) if reproduced_means else None,
        "paper_overall_mean": paper_overall_mean,
        "overall_absolute_difference": (
            abs(statistics.mean(reproduced_means) - paper_overall_mean)
            if reproduced_means else None
        ),
        "mean_absolute_error": statistics.mean(absolute_differences) if absolute_differences else None,
        "tasks_with_error_le_0.01": sum(value <= 0.01 for value in absolute_differences),
        "tasks_with_error_le_0.02": sum(value <= 0.02 for value in absolute_differences),
        "tasks_with_error_le_0.05": sum(value <= 0.05 for value in absolute_differences),
        "mean_seed_std": statistics.mean(scenario_stds) if scenario_stds else None,
        "mean_best_last_gap": statistics.mean(best_last_gaps) if best_last_gaps else None,
        "mean_official_clean_gap": statistics.mean(official_clean_gaps) if official_clean_gaps else None,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_report(output_dir, rows, summary_rows, decision)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--target-label-selection", action="store_true")
    args = parser.parse_args()
    decision = summarize(
        args.input, args.output_dir, diagnostic=args.diagnostic,
        target_label_selection=args.target_label_selection,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 2 if decision["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
