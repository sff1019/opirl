#!/bin/bash

EXPERT_PATH_DIR="experts/sac/custom_ant/expert"

python run_opirl.py \
  --algo opirl \
  --use_bc_reg \
  --learn_alpha \
  --seed 2 \
  --max_timesteps 1000000 \
  --env_name CustomAnt \
  --expert_path_dir $EXPERT_PATH_DIR \
  --save_dir results/opirl/custom_ant \
  --save_model \
  --save_model_interval 100000
