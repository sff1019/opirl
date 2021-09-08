#!/bin/bash

MODEL_DIR="experts/rewards/big_ant/models"
EXPERT_PATH_DIR="experts/sac/custom_ant/expert"

python run_transfer_opirl.py \
  --algo opirl \
  --load_model_dir $MODEL_DIR \
  --learn_alpha \
  --seed 42 \
  --actor_lr 0.00003 \
  --max_timesteps 1000000 \
  --env_name BigAnt \
  --expert_path_dir $EXPERT_PATH_DIR \
  --save_dir results/opirl/big_ant
