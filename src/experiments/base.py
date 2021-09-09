import argparse

import numpy as np


class BaseTrainer:
    def __init__(self):
        pass

    def _set_from_args(self, args):
        self._algo = args.algo
        self._absorbing_per_episode = args.absorbing_per_episode
        self._actor_update_freq = args.actor_update_freq
        self._discount = args.discount
        self._eval_interval = args.eval_interval
        self._max_timesteps = args.max_timesteps
        self._num_random_actions = args.num_random_actions
        self._sample_batch_size = args.sample_batch_size
        self._start_training_timesteps = args.start_training_timesteps
        self._target_entropy = args.target_entropy
        self._tau = args.tau
        self._imitator_updates_per_step = args.imitator_updates_per_step
        self._policy_updates_per_step = args.policy_updates_per_step
        self._updates_per_step = args.updates_per_step
        self._use_init_states = args.use_init_states
        self._log_env_diagnostics = args.log_env_diagnostics

        # Save
        self._save_model = args.save_model
        self._save_model_interval = args.save_model_interval
        self._load_model_dir = args.load_model_dir
        self._save_test_movie = args.save_test_movie
        self._save_n_movies = args.save_n_movies
        # List of indices for saving videos
        n_test = int(args.max_timesteps / args.eval_interval)
        # List of index of testing
        eval_interval_indices = [
            args.eval_interval * i for i in range(n_test + 1) if i > 0
        ]
        if len(eval_interval_indices) < args.save_n_movies:
            self._save_movie_indices = eval_interval_indices
        else:
            # List of index for saving videos. Always save the last video.
            save_movie_indices = np.array_split(eval_interval_indices,
                                                args.save_n_movies)
            self._save_movie_indices = []
            for idx, indices in enumerate(save_movie_indices):
                self._save_movie_indices.append(indices[0])
                if idx == len(save_movie_indices) - 1:
                    self._save_movie_indices.append(indices[-1])
        print(self._save_movie_indices)

    def get_argument(parser=None):
        # yapf: disable
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # Environment configs
        parser.add_argument('--algo', type=str, default='value_dice',
                            choices=['opirl'],
                            help='Algorithm to use to compute reward.')
        parser.add_argument('--env_name', type=str, default='HalfCheetah-v2',
                            help='Environment for training/evaluation.')
        parser.add_argument('--max_timesteps', type=int, default=int(1e5),
                            help='Max timesteps to train.')
        parser.add_argument('--normalize_states', action='store_true',
                            help='Normalize states using expert stats.')
        parser.add_argument('--sample_batch_size', type=int, default=256,
                            help='Batch size.')
        parser.add_argument('--seed', type=int, default=0,
                            help='Fixed random seed for training.')
        # Training configs (General)
        parser.add_argument('--absorbing_per_episode', type=int, default=10,
                            help='A number of absorbing states per episode to add.')
        parser.add_argument('--buffer_size', type=int, default=None,
                            help='Size of replay buffer')
        parser.add_argument('--eval_interval', type=int, default=int(1e4),
                            help='Evaluate every N timesteps.')
        parser.add_argument('--hidden_size', type=int, default=256,
                            help='Hidden size.')
        parser.add_argument('--num_random_actions', type=int, default=int(2e3),
                            help='Fill replay buffer with N random actions.')
        parser.add_argument('--updates_per_step', type=int, default=1,
                            help='Updates per time step.')
        parser.add_argument('--imitator_updates_per_step', type=int, default=1,
                            help='Updates imitator per step')
        parser.add_argument('--policy_updates_per_step', type=int, default=1,
                            help='Updates imitator per step')
        parser.add_argument(
            '--start_training_timesteps', type=int, default=int(1e3),
            help='Start training when replay buffer contains N timesteps.')
        # Training configs (Policy)
        parser.add_argument('--actor_lr', type=float, default=1e-5,
                            help='Actor learning rate.')
        parser.add_argument('--actor_update_freq', type=int, default=2,
                            help='Update actor every N steps.')
        parser.add_argument('--critic_lr', type=float, default=1e-3,
                            help='Critic learning rate.')
        parser.add_argument('--discount', type=float, default=0.99,
                            help='Discount used for returns.')
        parser.add_argument('--learn_alpha', action='store_true',
                            help='Whether to learn temperature for SAC.')
        parser.add_argument('--nu_lr', type=float, default=1e-3,
                            help='nu network learning rate.')
        parser.add_argument('--replay_regularization', type=float, default=0.1,
                            help='Amount of replay mixing.')
        parser.add_argument('--sac_alpha', type=float, default=0.1,
                            help='SAC temperature.')
        # Training configs (Imitator)
        parser.add_argument('--algae_alpha', type=float, default=0.01,
                            help='ALGAE alpha.')
        parser.add_argument('--disc_lr', type=float, default=1e-5,
                            help='Discriminator learning rate.')
        parser.add_argument('--f_exponent', type=float, default=1.5,
                            help='Exponent for f.')
        parser.add_argument(
            '--target_entropy', type=float, default=None,
            help='(optional) target_entropy for training actor. If None, -env.action_space.shape[0] is used.')  # noqa
        parser.add_argument('--tau', type=float, default=0.005,
                            help='Soft update coefficient for the target network.')
        # Other (IRL)
        parser.add_argument('--expert_path_dir_head', default=None,
                            help='Path to expert trajectories')
        parser.add_argument('--num_trajectories', type=int, default=1,
                            help='Number of trajectories to use.')
        parser.add_argument('--use_discrim_as_reward', action='store_true')
        parser.add_argument('--use_bc_reg', action='store_true')
        # Other
        parser.add_argument('--log_env_diagnostics', action='store_true')
        parser.add_argument('--use_init_states', action='store_true',
                            help='Use init states.')
        parser.add_argument('--bc_reg_coeff', type=float, default=1.)
        parser.add_argument('--actor_reg_coeff', type=float, default=1.e-3)
        parser.add_argument('--reward_type', type=str, default='opirl',
                            choices=['airl', 'gail', 'opirl'])
        parser.add_argument('--reward_shaping', action='store_true')
        parser.add_argument('--reward_clip_max', type=float, default=None)
        parser.add_argument('--reward_clip_min', type=float, default=None)
        # Save
        parser.add_argument('--load_model_dir', type=str, default=None,
                            help='Directory of model to load')
        parser.add_argument('--save_dir', type=str, default=None,
                            help='Directory to save results to.')
        parser.add_argument('--save_model', action='store_true',
                            help='Save model every `--save_model_interval` if set True')  # noqa
        parser.add_argument('--save_model_interval', type=int, default=1e4,
                            help='Interval to save model')
        parser.add_argument('--save_test_movie', action='store_true',
                            help='Save rendering results')
        parser.add_argument('--save_n_movies', type=int, default=10,
                            help='Number of videos to save')
        parser.add_argument('--wandb_entity', type=str, default=None,
                            help='Entity name')
        parser.add_argument('--wandb_project', type=str, default=None,
                            help='Project name')
        parser.add_argument('--log_to_wandb', action='store_true',
                            help='To log results to Wandb')
        return parser
