#! /bin/bash

set -ex

python scripts/fact_verification.py \
  --model_name_or_path albert-base-v2 \
  --tasks_names fever \
  --loss_fn PoE \
  --bias_name known_ent_overweight\
  --data_dir data \
  --do_train \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_steps 50000 \
  --save_step 10000 \
  --overwrite_output_dir \
  --overwrite_cache \
  --output_dir models/bias_trained/fever_only/poe/known_ent_overweight \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
