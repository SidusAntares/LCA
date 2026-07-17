#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

export WANDB_MODE=disabled

SAVE_DIR=${SAVE_DIR:-LCA_all_result}
RUN_IDS=${RUN_IDS:-0}
NUM_RUNS=${NUM_RUNS:-3}
NUM_EPOCHS=${NUM_EPOCHS:-40}
TRAINING_PROTOCOL=${TRAINING_PROTOCOL:-paper_code_protocol}
case "$TRAINING_PROTOCOL" in
  paper_code_protocol) METRIC_PROTOCOL=official_stateful_no_reset ;;
  baseline_clean_protocol) METRIC_PROTOCOL=stateless_current ;;
  *) echo "invalid TRAINING_PROTOCOL: $TRAINING_PROTOCOL" >&2; exit 64 ;;
esac

GPU0=("18,14" "19,11" "12,3")
GPU1=("6,13" "17,18" "17,14")
GPU2=("20,9" "9,19")
GPU3=("7,18" "2,12")

run_gpu_queue() {
  local gpu_id=$1
  shift
  local queue_status=0
  local pair src tgt exp_name exp_dir log_dir pid

  for pair in "$@"; do
    IFS=',' read -r src tgt <<< "$pair"
    exp_name="HAR_${src}_to_${tgt}_type1"
    exp_dir="$SAVE_DIR/HAR/LCA_${exp_name}"
    log_dir="$exp_dir/launcher_logs"
    mkdir -p "$log_dir"

    if python tools/check_checkpoint.py \
      --experiment-dir "$exp_dir" --scenario "$pair" --run-ids "$RUN_IDS" >/dev/null 2>&1; then
      echo "skip complete task ${src}->${tgt}"
      continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
      --phase train \
      --save_dir "$SAVE_DIR" \
      --exp_name "$exp_name" \
      --da_method LCA \
      --dataset HAR \
      --data_path dataset \
      --scenario "$pair" \
      --num_runs "$NUM_RUNS" \
      --run_ids "$RUN_IDS" \
      --num_epochs "$NUM_EPOCHS" \
      --training_protocol "$TRAINING_PROTOCOL" \
      --metric_protocol "$METRIC_PROTOCOL" \
      --type type1 \
      --lr 0.001 \
      --device cuda:0 \
      >"$log_dir/stdout.log" 2>"$log_dir/stderr.log" &
    pid=$!
    echo "$pid" >"$log_dir/pid"
    if ! wait "$pid"; then
      echo "task ${src}->${tgt} failed; see $log_dir/stderr.log" >&2
      queue_status=1
    fi
  done
  return "$queue_status"
}

run_gpu_queue 0 "${GPU0[@]}" & worker0=$!
run_gpu_queue 1 "${GPU1[@]}" & worker1=$!
run_gpu_queue 2 "${GPU2[@]}" & worker2=$!
run_gpu_queue 3 "${GPU3[@]}" & worker3=$!

status=0
for worker in "$worker0" "$worker1" "$worker2" "$worker3"; do
  if ! wait "$worker"; then
    status=1
  fi
done

exit "$status"
