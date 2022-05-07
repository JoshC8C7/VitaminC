#! /bin/bash

set -ex

python scripts/fact_verification.py \
  --model_name_or_path models/bias_trained/fever_only/shallow \
  --tasks_names vitc-alb-base \
  --data_dir data \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate 2e-5 \
  --overwrite_cache \
  --output_dir results/baseteachers/fever-only/vitc-train-set/1from2 \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
