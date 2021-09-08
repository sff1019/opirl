#!/bin/bash
MODEL_DIR="experts/rewards/amputated_ant/models"
EXPERT_PATH_DIR="experts/sac/custom_ant/expert"

python run_transfer_opirl.py \
  --algo opirl \
  --load_model_dir $MODEL_DIR \
  --learn_alpha \
  --seed 42 \
  --max_timesteps 1000000 \
  --sample_batch_size 64 \
  --imitator_updates_per_step 0 \
  --env_name AmputatedAnt \
  --expert_path_dir $EXPERT_PATH_DIR \
  --save_dir results/amputated_ant
