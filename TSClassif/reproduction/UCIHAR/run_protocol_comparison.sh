#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="reproduction/UCIHAR"
SAVE_DIR="LCA_UCIHAR_protocol_compare"
mkdir -p "$OUT/logs"
export WANDB_MODE=disabled

run_protocol() {
  local training_protocol="$1"
  local metric_protocol="$2"
  CUDA_VISIBLE_DEVICES=0 python run.py \
    --phase train \
    --save_dir "$SAVE_DIR" \
    --exp_name "20_to_9_${training_protocol}" \
    --da_method LCA \
    --dataset HAR \
    --data_path dataset \
    --scenario 20,9 \
    --num_runs 1 \
    --run_ids 0 \
    --num_epochs 40 \
    --training_protocol "$training_protocol" \
    --metric_protocol "$metric_protocol" \
    --type type1 \
    --lr 0.001 \
    --device cuda:0 \
    >"$OUT/logs/protocol_compare_${training_protocol}.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 python "$OUT/evaluate_checkpoints.py" \
    --experiment-dir "$SAVE_DIR/HAR/LCA_20_to_9_${training_protocol}" \
    --data-path dataset \
    --scenario 20,9 \
    --run-ids 0 \
    --device cuda:0 \
    --output "$OUT/protocol_compare_${training_protocol}.csv"
}

run_protocol paper_code_protocol official_stateful_no_reset
run_protocol baseline_clean_protocol stateless_current

python "$OUT/compare_protocol_results.py" \
  --paper "$OUT/protocol_compare_paper_code_protocol.csv" \
  --clean "$OUT/protocol_compare_baseline_clean_protocol.csv" \
  --output "$OUT/protocol_comparison_20_to_9_seed0.json"

echo "Protocol comparison complete. Review protocol_comparison_20_to_9_seed0.json before choosing the diagnostic protocol."

