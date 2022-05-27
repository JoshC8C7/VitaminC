#! /bin/bash

set -ex

python scripts/fact_verification.py \
  --model_name_or_path bert-base-uncased \
  --data_dir data \
  --do_predict \
  --test_tasks vitc-alb-base \
  --max_seq_length 256 \
  --per_device_eval_batch_size 128 \
  --overwrite_cache \
  --output_dir results2/teachbertpredsfever/vitc/1s \
  "$@"

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
