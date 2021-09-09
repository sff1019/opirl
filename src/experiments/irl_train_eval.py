import os
import datetime
import time
from absl import logging

import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec

from src.experiments.base import BaseTrainer
from src.misc.logger import Logger
from src.misc.replay_buffer import get_replay_buffer, add_samples_to_replay_buffer
from src.misc.video_recorder import VideoRecorder

if tf.config.experimental.list_physical_devices('GPU'):
    gpu = 0
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class IRLTrainer(BaseTrainer):
    def __init__(self,
                 policy,
                 imitator,
                 env,
                 eval_env,
                 args,
                 expert_states,
                 expert_actions,
                 expert_next_states,
                 expert_dones,
                 expert_logps,
                 shift=None,
                 scale=None,
                 no_expert=False,
                 buffer_size=None,
                 use_init_states=False):
        self._set_from_args(args)
        self.env = env
        self.eval_env = eval_env

        self.policy = policy
        self.imitator = imitator

        self._shift = shift
        self._scale = scale

        if buffer_size is None:
            buffer_size = args.max_timesteps * 2
        self.replay_buffer = get_replay_buffer(env, size=buffer_size)
        self.replay_buffer_iter = iter(
            self.replay_buffer.as_dataset(
                sample_batch_size=args.sample_batch_size))

        self.policy_replay_buffer = get_replay_buffer(env, size=buffer_size)
        self.policy_replay_buffer_iter = iter(
            self.policy_replay_buffer.as_dataset(
                sample_batch_size=args.sample_batch_size))

        self._use_init_states = use_init_states
        if use_init_states:
            init_spec = tensor_spec.TensorSpec(
                [env.observation_space.shape[0]], tf.float32, 'observation')
            self.init_replay_buffer = get_replay_buffer(
                env, size=args.max_timesteps, spec=init_spec)
            self.init_replay_buffer_iter = iter(
                self.init_replay_buffer.as_dataset(
                    sample_batch_size=self._sample_batch_size))

        if not no_expert:
            expert_states = tf.Variable(expert_states, dtype=tf.float32)
            expert_actions = tf.Variable(expert_actions, dtype=tf.float32)
            expert_next_states = tf.Variable(expert_next_states,
                                             dtype=tf.float32)
            expert_dones = tf.Variable(expert_dones, dtype=tf.float32)
            expert_logps = tf.Variable(expert_logps, dtype=tf.float32)

            expert_dataset = tf.data.Dataset.from_tensor_slices(
                (expert_states, expert_actions, expert_next_states))

            self.expert_dataset = expert_dataset.repeat().shuffle(
                expert_states.shape[0]).batch(self._sample_batch_size,
                                              drop_remainder=True)
            self.expert_dataset_iter = iter(self.expert_dataset)

        time_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S.%f')
        self._output_dir = os.path.join(args.save_dir, time_str)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        # Create directory to save final models
        if self._save_model:
            self._model_dir = os.path.join(self._output_dir, 'models')
            if not os.path.exists(self._model_dir):
                os.makedirs(self._model_dir)

        # Set up loggers
        self.logger = Logger(self._output_dir, args, agent=args.algo)
        self.video_recorder = VideoRecorder(
            self._output_dir if self._save_test_movie else None)

    def __call__(self):
        episode_return = 0
        episode_timesteps = 0
        done = True

        total_timesteps = 0
        previous_time = time.time()

        eval_returns = []
        diagnostics = {}

        while total_timesteps <= self._max_timesteps:
            if total_timesteps % self._eval_interval == 0:
                logging.info('Performing policy eval.')
                average_returns, evaluation_timesteps = self.evaluate(
                    total_timesteps)

                eval_returns.append(average_returns)

                self.logger.log('eval/average_test_return', average_returns)
                self.logger.log('eval/average_test_episode_length',
                                evaluation_timesteps)
                self.logger.dump(total_timesteps,
                                 dump_to_csv=True,
                                 dump_eval=True)
                logging.info('Eval: ave returns=%f, ave episode length=%f',
                             average_returns, evaluation_timesteps)
            if done:
                if episode_timesteps > 0:
                    current_time = time.time()
                    fps = episode_timesteps / (current_time - previous_time)
                    self.logger.log('common/training_return', episode_return)
                    self.logger.log('common/fps', fps)
                    if self._log_env_diagnostics:
                        for k, v in diagnostics.items():
                            self.logger.log(f'common/{k}', v)

                obs = self.env.reset()
                episode_return = 0
                episode_timesteps = 0
                previous_time = time.time()

            if total_timesteps < self._num_random_actions:
                action = self.env.action_space.sample()
                logp = self.policy.get_logp(np.array([obs]),
                                            np.array([action]))
            else:
                mean_action, _, logp = self.policy.actor.call(np.array([obs]))
                action = mean_action[0].numpy()
                action = (action +
                          np.random.normal(0, 0.1, size=action.shape)).clip(
                              -1, 1)
            next_obs, reward, done, diagnostics = self.env.step(action)

            # done caused by episode truncation.
            truncated_done = done and episode_timesteps + 1 == self.env._max_episode_steps

            if done and not truncated_done:
                next_obs = self.env.get_absorbing_state()

            add_samples_to_replay_buffer(self.replay_buffer, obs, action,
                                         next_obs, logp)
            add_samples_to_replay_buffer(self.policy_replay_buffer, obs,
                                         action, next_obs, logp)
            if done and not truncated_done:
                # Add several absobrsing states to absorbing states transitions.
                for abs_i in range(self._absorbing_per_episode):
                    if abs_i + episode_timesteps < self.env._max_episode_steps:  # pylint: disable=protected-access
                        obs = self.env.get_absorbing_state()
                        action = self.env.action_space.sample()
                        next_obs = self.env.get_absorbing_state()
                        logp = self.policy.actor.get_log_prob(
                            np.array([obs]), np.array([action]))

                        add_samples_to_replay_buffer(self.replay_buffer, obs,
                                                     action, next_obs, logp)
                        add_samples_to_replay_buffer(self.policy_replay_buffer,
                                                     obs, action, next_obs,
                                                     logp)

            episode_return += reward
            episode_timesteps += 1
            total_timesteps += 1

            obs = next_obs

            if total_timesteps >= self._start_training_timesteps:
                for _ in range(self._updates_per_step):
                    self._train_imitator()
                    self._train_policy()

            if done:
                self.logger.dump(total_timesteps, dump_to_csv=True)

        self.logger.dump(total_timesteps, flush_tb=True, dump_to_csv=True)

    def evaluate(self, total_timesteps, num_episodes=20):
        eval_total_timesteps = 0
        total_returns = 0

        for i in range(num_episodes):
            state = self.eval_env.reset()
            self.video_recorder.init(enabled=(i == 0))
            done = False
            while not done:
                action, _, _ = self.policy.actor(np.array([state]),
                                                 training=False)
                action = action[0].numpy()

                next_state, reward, done, _ = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)

                total_returns += reward
                eval_total_timesteps += 1
                state = next_state
            if self._save_test_movie and total_timesteps in self._save_movie_indices:
                print('saving test movie')
                self.video_recorder.save(f'{total_timesteps}.mp4',
                                         logger=self.logger,
                                         step=total_timesteps)

        return total_returns / num_episodes, eval_total_timesteps / num_episodes

    def _train_imitator(self):
        for _ in range(self._imitator_updates_per_step):
            return_dict = self.imitator.update(
                **self._get_imitator_train_kwargs())

            self.logger.log('train/disc_loss', return_dict['train/gail_loss'])

    def _train_policy(self):
        for _ in range(self._policy_updates_per_step):
            if self._algo == 'bc':
                expert_states, expert_actions, expert_next_states = next(
                    self.expert_dataset_iter)
                return_dict = self.policy.train_bc(expert_states,
                                                   expert_actions)
            else:
                kwargs = self._get_policy_train_kwargs()
                return_dict = self.policy.train(**kwargs)
                return_dict['train/imitator_reward'] = np.mean(
                    kwargs['rewards'])
            self._log_to_logger(return_dict)

    def _log_to_logger(self, return_dict):
        for k, v in return_dict.items():
            self.logger.log(k, v)

    def _get_policy_train_kwargs(self):
        states, actions, next_states, _, masks, logp = next(
            self.replay_buffer_iter)[0]

        rewards = self.imitator.predict_reward(states, actions, next_states,
                                               logp)
        kwargs = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'rewards': rewards,
            'masks': masks,
            'target_entropy': -self.env.action_space.shape[0]
        }

        if self._use_init_states:
            init_states = next(self.init_replay_buffer)[0]
        else:
            init_states = states
        kwargs['init_states'] = init_states
        kwargs['actor_update_freq'] = self._actor_update_freq

        return kwargs

    def _get_imitator_train_kwargs(self):
        expert_states, expert_actions, expert_next_states = next(
            self.expert_dataset_iter)
        policy_states, policy_actions, policy_next_states, _, _, _ = next(
            self.policy_replay_buffer_iter)[0]

        kwargs = {
            'expert_states': expert_states,
            'expert_actions': expert_actions,
            'policy_states': policy_states,
            'policy_actions': policy_actions
        }

        kwargs['expert_next_states'] = expert_next_states
        kwargs['policy_next_states'] = policy_next_states
        kwargs['actor'] = self.policy.actor

        return kwargs
