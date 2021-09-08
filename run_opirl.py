import os
import random

import numpy as np
import tensorflow as tf

from src.algos.airl import AIRLGP
from src.algos.opirl_policy import Policy
from src.envs.initialize_env import initialize_env
from src.experiments.irl_train_eval import IRLTrainer
import src.misc.data_utils as data_utils
from src.misc.keras_utils import save_keras_model
import src.misc.load_expert as load_expert
import src.wrappers as wrappers

if tf.config.experimental.list_physical_devices('GPU'):
    gpu = 0
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(cur_device, enable=True)

if __name__ == '__main__':
    parser = IRLTrainer.get_argument()
    parser.add_argument('--n_paths', type=int, default=20)
    args = parser.parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = initialize_env(args.env_name, seed=args.seed)
    eval_env = initialize_env(args.env_name, seed=args.seed)

    # Sample expert trajectories
    expert_path_dirs = args.expert_path_dir_head
    expert_trajs = load_expert.restore_latest_n_traj(
        expert_path_dirs,
        max_steps=env._max_episode_steps,
        n_path=args.n_paths,
        shuffle=True,
        shuffle_n_path=args.num_trajectories)

    expert_states = expert_trajs['obses']
    expert_actions = expert_trajs['acts']
    expert_next_states = expert_trajs['next_obses']
    expert_dones = expert_trajs['dones']
    expert_logps = expert_trajs['logps']

    # Normalize states
    if args.normalize_states:
        shift = -np.mean(expert_states, 0)
        scale = 1.0 / (np.std(expert_states, 0) + 1e-3)
        expert_states = (expert_states + shift) * scale
        expert_next_states = (expert_next_states + shift) * scale
    else:
        shift = None
        scale = None

    # Add absorbing states
    env = wrappers.create_il_env(env, args.seed, shift, scale)
    eval_env = wrappers.create_il_env(eval_env, args.seed + 1, shift, scale)
    unwrap_env = env

    while hasattr(unwrap_env, 'env'):
        if isinstance(unwrap_env, wrappers.NormalizeBoxActionWrapper):
            expert_actions = unwrap_env.reverse_action(expert_actions)
            break
        unwrap_env = unwrap_env.env

    (expert_states, expert_actions, expert_next_states, expert_dones,
     expert_logps) = data_utils.add_absorbing_states(expert_states,
                                                     expert_actions,
                                                     expert_next_states,
                                                     expert_dones,
                                                     expert_logps, env)

    policy = Policy(env.observation_space,
                    env.action_space,
                    env.observation_space.shape[0],
                    env.action_space.shape[0],
                    critic_lr=args.critic_lr,
                    actor_lr=args.actor_lr,
                    use_init_states=args.use_init_states,
                    use_bc_reg=args.use_bc_reg,
                    bc_reg_coeff=args.bc_reg_coeff / args.sample_batch_size,
                    actor_coeff=args.actor_reg_coeff,
                    algae_alpha=args.algae_alpha,
                    exponent=args.f_exponent)

    imitator = AIRLGP(env.observation_space,
                      env.action_space,
                      disc_lr=args.disc_lr,
                      reward_shaping=args.reward_shaping,
                      rew_clip_max=args.reward_clip_max,
                      rew_clip_min=args.reward_clip_min)

    trainer = IRLTrainer(policy,
                         imitator,
                         env,
                         eval_env,
                         args,
                         expert_states,
                         expert_actions,
                         expert_next_states,
                         expert_dones,
                         expert_logps,
                         shift=shift,
                         scale=scale,
                         use_init_states=args.use_init_states)
    trainer()

    # Save models for transfer learning
    if trainer._save_model:
        save_keras_model(imitator.rew_net,
                         os.path.join(trainer._model_dir, 'final_rew_net'))
        save_keras_model(policy.actor.trunk,
                         os.path.join(trainer._model_dir, 'final_actor'))
