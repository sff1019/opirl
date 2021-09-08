#!/bin/bash

EXPERT_PATH_DIR="experts/sac/point_maze/expert"

python run_opirl.py \
  --algo opirl \
  --normalize_states \
  --learn_alpha \
  --use_bc_reg \
  --seed 0 \
  --max_timesteps 1000000 \
  --buffer_size 300000 \
  --disc_lr 0.003 \
  --env_name PointMaze-Left \
  --expert_path_dir $EXPERT_PATH_DIR \
  --save_dir results/opirl/pointmaze_left \
  --reward_clip_max 0.0
