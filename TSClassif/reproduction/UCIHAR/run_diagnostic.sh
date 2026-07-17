#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="reproduction/UCIHAR"
TRAINING_PROTOCOL="${TRAINING_PROTOCOL:?Set TRAINING_PROTOCOL to paper_code_protocol or baseline_clean_protocol}"
case "$TRAINING_PROTOCOL" in
  paper_code_protocol) METRIC_PROTOCOL=official_stateful_no_reset ;;
  baseline_clean_protocol) METRIC_PROTOCOL=stateless_current ;;
  *) echo "Invalid TRAINING_PROTOCOL: $TRAINING_PROTOCOL" >&2; exit 64 ;;
esac
SAVE_DIR="LCA_UCIHAR_${TRAINING_PROTOCOL}"
mkdir -p "$OUT/logs"
export WANDB_MODE=disabled

run_task() {
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
    >"$OUT/logs/${tag}.log" 2>&1
}

run_task 0 20,9 20_to_9 &
pid0=$!
run_task 1 7,18 7_to_18 &
pid1=$!
run_task 2 9,19 9_to_19 &
pid2=$!

status=0
for pid in "$pid0" "$pid1" "$pid2"; do
  wait "$pid" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  echo "A diagnostic training process failed; see $OUT/logs." >&2
  exit "$status"
fi

evaluate_task() {
  local gpu="$1"
  local scenario="$2"
  local tag="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python "$OUT/evaluate_checkpoints.py" \
    --experiment-dir "$SAVE_DIR/HAR/LCA_UCIHAR_${tag}" \
    --data-path dataset \
    --scenario "$scenario" \
    --run-ids 0,1,2 \
    --device cuda:0 \
    --output "$OUT/diagnostic_${tag}.csv"
}

evaluate_task 0 20,9 20_to_9 &
pid0=$!
evaluate_task 1 7,18 7_to_18 &
pid1=$!
evaluate_task 2 9,19 9_to_19 &
pid2=$!

status=0
for pid in "$pid0" "$pid1" "$pid2"; do
  wait "$pid" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  echo "A diagnostic clean evaluation failed." >&2
  exit "$status"
fi

if python "$OUT/summarize_results.py" \
  --diagnostic \
  --output-dir "$OUT" \
  --input \
    "$OUT/diagnostic_20_to_9.csv" \
    "$OUT/diagnostic_7_to_18.csv" \
    "$OUT/diagnostic_9_to_19.csv"
then
  echo "Diagnostic passed. Review diagnostic files, then run run_full.sh manually."
  exit 0
else
  code=$?
  if [[ "$code" -eq 2 ]]; then
    echo "Diagnostic stop criterion triggered; full matrix was not started." >&2
  fi
  exit "$code"
fi
