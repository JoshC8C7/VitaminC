#! /bin/bash

set -ex

python scripts/fact_verification.py \
  --model_name_or_path models/fever \
  --tasks_names fever \
  --data_dir data \
  --do_test \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_steps 50000 \
  --save_step 10000 \
  --overwrite_cache \
  --do_predict\
  --output_dir results/fever_albert \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
