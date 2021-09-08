import os
import random

import numpy as np
import tensorflow as tf

from src.algos.sac import SAC
from src.envs.initialize_env import initialize_env
from src.experiments.train_eval import Trainer
from src.misc import keras_utils
from src.misc.keras_utils import save_keras_model
import src.misc.load_expert as load_expert
import src.wrappers as wrappers

if tf.config.experimental.list_physical_devices('GPU'):
    gpu = 0
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


def main():
    parser = Trainer.get_argument()
    parser.add_argument('--n_path', type=int, default=20)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rew_net = keras_utils.load_keras_model(
        os.path.join(args.load_model_dir, 'final_rew_net'))

    @tf.function
    def inference(states, actions):
        if len(states.shape) == len(actions.shape) == 1:
            states = tf.expand_dims(states, axis=0)
            states = tf.cast(states, tf.float32)
            actions = tf.expand_dims(actions, axis=0)
            actions = tf.cast(actions, tf.float32)

        inputs = tf.concat([states, actions], -1)
        rewards = rew_net(inputs, training=False)

        if args.reward_clip_max is not None:
            clip_max = tf.int32.max if args.reward_clip_max is None else args.reward_clip_max  # noqa
            clip_min = tf.int32.min
            rewards = tf.clip_by_value(rewards, clip_min, clip_max)

        return rewards

    # Create real envs to use for training and evaluation
    env = initialize_env(args.env_name, seed=args.seed, inference_fn=inference)
    eval_env = initialize_env(args.env_name, seed=args.seed + 1)

    expert_path_dirs = args.expert_path_dir_head
    expert_trajs = load_expert.restore_latest_n_traj(
        expert_path_dirs,
        max_steps=env._max_episode_steps,
        n_path=args.n_path,
        shuffle=True,
        shuffle_n_path=args.num_trajectories)

    expert_states = expert_trajs['obses']
    expert_next_states = expert_trajs['next_obses']

    if args.normalize_states:
        shift = -np.mean(expert_states, 0)
        scale = 1.0 / (np.std(expert_states, 0) + 1e-3)
        expert_states = (expert_states + shift) * scale
        expert_next_states = (expert_next_states + shift) * scale
    else:
        shift = None
        scale = None

    env = wrappers.create_il_env(env, args.seed, shift, scale)
    eval_env = wrappers.create_il_env(eval_env, args.seed + 1, shift, scale)

    if args.buffer_size is None:
        args.buffer_size = args.max_timesteps * 2

    policy = SAC(env.observation_space,
                 env.action_space,
                 env.observation_space.shape[0],
                 env.action_space.shape[0],
                 actor_lr=args.actor_lr,
                 critic_lr=args.critic_lr,
                 learn_alpha=args.learn_alpha,
                 alpha_init=0.2,
                 memory_capacity=args.buffer_size,
                 n_warmup=args.num_random_actions)

    trainer = Trainer(policy,
                      env,
                      eval_env,
                      args,
                      inference_fn=inference,
                      shift=shift,
                      scale=scale)

    trainer()

    if args.save_model:
        save_keras_model(rew_net, os.path.join(trainer._model_dir, 'rew_net'))
        save_keras_model(policy.actor.trunk,
                         os.path.join(trainer._model_dir, 'final_actor'))


if __name__ == '__main__':
    main()
