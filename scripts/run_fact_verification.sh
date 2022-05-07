#! /bin/bash

set -ex

python scripts/fact_verification.py \
  --model_name_or_path models/bias_trained/vitc_fever/shallow \
  --tasks_names fever-alb-base \
  --data_dir data \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate 2e-5 \
  --overwrite_cache \
  --loss_fn plain \
  --output_dir results/shallow/fever-vitc/fever-train-set/0from2 \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
