#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="reproduction/UCIHAR"
TRAINING_PROTOCOL="${TRAINING_PROTOCOL:?Set TRAINING_PROTOCOL after reviewing the protocol comparison and diagnostics}"
case "$TRAINING_PROTOCOL" in
  paper_code_protocol) METRIC_PROTOCOL=official_stateful_no_reset ;;
  baseline_clean_protocol) METRIC_PROTOCOL=stateless_current ;;
  *) echo "Invalid TRAINING_PROTOCOL: $TRAINING_PROTOCOL" >&2; exit 64 ;;
esac
SAVE_DIR="LCA_UCIHAR_${TRAINING_PROTOCOL}"
LEGACY_CHECKPOINT_DIR="LCA_single_result/HAR/LCA_HAR_18_to_14_type1"
mkdir -p "$OUT/logs"
export WANDB_MODE=disabled

scenarios=(18,14 6,13 20,9 7,18 19,11 17,18 9,19 2,12 12,3 17,14)
tags=(18_to_14 6_to_13 20_to_9 7_to_18 19_to_11 17_to_18 9_to_19 2_to_12 12_to_3 17_to_14)

launch_task() {
  local gpu="$1"
  local scenario="$2"
  local tag="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python run.py \
    --phase train \
    --save_dir "$SAVE_DIR" \
    --exp_name "UCIHAR_${tag}" \
    --da_method LCA \
    --dataset HAR \
    --data_path dataset \
    --scenario "$scenario" \
    --num_runs 3 \
    --run_ids 0,1,2 \
    --num_epochs 40 \
    --training_protocol "$TRAINING_PROTOCOL" \
    --metric_protocol "$METRIC_PROTOCOL" \
    --type type1 \
    --lr 0.001 \
    --device cuda:0 \
    >"$OUT/logs/${tag}.log" 2>&1 &
  task_pid=$!
}

pids=()
for index in "${!scenarios[@]}"; do
  gpu=$((index % 4))
  launch_task "$gpu" "${scenarios[$index]}" "${tags[$index]}"
  pids+=("$task_pid")
  if [[ "${#pids[@]}" -eq 4 || "$index" -eq 9 ]]; then
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=$?
    done
    if [[ "$status" -ne 0 ]]; then
      echo "A full-matrix training process failed; see $OUT/logs." >&2
      exit "$status"
    fi
    pids=()
  fi
done

evaluation_files=()
for index in "${!scenarios[@]}"; do
  scenario="${scenarios[$index]}"
  tag="${tags[$index]}"
  output="$OUT/eval_${tag}.csv"
  CUDA_VISIBLE_DEVICES=0 python "$OUT/evaluate_checkpoints.py" \
    --experiment-dir "$SAVE_DIR/HAR/LCA_UCIHAR_${tag}" \
    --data-path dataset \
    --scenario "$scenario" \
    --run-ids 0,1,2 \
    --device cuda:0 \
    --output "$output"
  evaluation_files+=("$output")
done

if [[ -f "$LEGACY_CHECKPOINT_DIR/18_to_14_run_0/checkpoint.pt" ]]; then
  CUDA_VISIBLE_DEVICES=0 python "$OUT/evaluate_checkpoints.py" \
    --experiment-dir "$LEGACY_CHECKPOINT_DIR" \
    --data-path dataset \
    --scenario 18,14 \
    --run-ids 0 \
    --device cuda:0 \
    --legacy \
    --output "$OUT/legacy_18_to_14_run0.csv"
fi

python "$OUT/summarize_results.py" \
  --output-dir "$OUT" \
  --input "${evaluation_files[@]}"
