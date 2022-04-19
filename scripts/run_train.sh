#! /bin/bash

set -ex

python scripts/fact_verification.py \
  --model_name_or_path ../old/hff/albert-base-vitaminc-fever \
  --tasks_names fever vitaminc \
  --data_dir data \
  --do_train \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_steps 500 \
  --save_step 100 \
  --dataset_size 100 \
  --tasks_ratios 0.1 0.9 \
  --output_dir models/newmodel5 \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
