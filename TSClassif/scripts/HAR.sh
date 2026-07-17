#!/bin/bash

python run.py --phase train --save_dir LCA_all_result --exp_name HAR_type1 \
  --da_method LCA --dataset HAR --data_path dataset --num_runs 3 \
  --training_protocol paper_code_protocol \
  --metric_protocol official_stateful_no_reset \
  --type type1 --lr 0.001 --device cuda:0
