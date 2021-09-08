#!/bin/bash

MODEL_DIR="experts/rewards/pointmaze_right/models"
EXPERT_PATH_DIR="experts/sac/point_maze/expert"

python run_transfer_opirl.py \
  --algo opirl \
  --normalize_states \
  --learn_alpha \
  --seed 42 \
  --max_timesteps 1000000 \
  --load_model_dir $MODEL_DIR \
  --env_name PointMaze-Right \
  --expert_path_dir $EXPERT_PATH_DIR \
  --save_dir results/opirl/pointmaze_right
