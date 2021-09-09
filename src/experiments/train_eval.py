"""Run training loop."""
from absl import logging
import datetime
import os
import time

import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.specs import tensor_spec

from src.experiments.base import BaseTrainer
from src.misc.replay_buffer import get_replay_buffer
from src.misc.logger import Logger
from src.misc.video_recorder import VideoRecorder


class Trainer(BaseTrainer):
    def __init__(self,
                 policy,
                 env,
                 eval_env,
                 args,
                 max_episode_steps=1000,
                 use_init_states=False,
                 buffer_size=None,
                 inference_fn=None,
                 shift=None,
                 scale=None,
                 **kwargs):
        self._set_from_args(args)
        self.env = env
        self.eval_env = eval_env
        self.max_episode_steps = max_episode_steps

        self.policy = policy
        self.inference_fn = inference_fn

        self._shift = shift
        self._scale = scale

        spec = (tensor_spec.TensorSpec([env.observation_space.shape[0]],
                                       tf.float32, 'observation'),
                tensor_spec.TensorSpec([env.action_space.shape[0]], tf.float32,
                                       'action'),
                tensor_spec.TensorSpec([env.observation_space.shape[0]],
                                       tf.float32, 'next_observation'),
                tensor_spec.TensorSpec([1], tf.float32, 'reward'),
                tensor_spec.TensorSpec([1], tf.float32, 'mask'))
        if buffer_size is None:
            buffer_size = args.max_timesteps
        self.replay_buffer = get_replay_buffer(env,
                                               size=buffer_size,
                                               spec=spec)
        self.replay_buffer_iter = iter(
            self.replay_buffer.as_dataset(
                sample_batch_size=self._sample_batch_size))

        self._use_init_states = use_init_states
        if use_init_states:
            init_spec = tensor_spec.TensorSpec(
                [env.observation_space.shape[0]], tf.float32, 'observation')
            self.init_replay_buffer = get_replay_buffer(
                env, size=args.max_timesteps, spec=init_spec)
            self.init_replay_buffer_iter = iter(
                self.init_replay_buffer.as_dataset(
                    sample_batch_size=self._sample_batch_size))

        time_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S.%f')
        self._output_dir = os.path.join(args.save_dir, time_str)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self.logger = Logger(self._output_dir, args, agent=args.algo)
        self.video_recorder = VideoRecorder(
            self._output_dir if self._save_test_movie else None)

        if self._save_model:
            self._model_dir = os.path.join(self._output_dir, 'models')
            if not os.path.exists(self._model_dir):
                os.makedirs(self._model_dir)

    def __call__(self):
        episode_return = 0
        episode_timesteps = 0
        done = True

        total_timesteps = 0
        previous_time = time.time()

        obs = self.env.reset()

        while total_timesteps < self._max_timesteps:
            if total_timesteps < self._num_random_actions:
                action = self.env.action_space.sample()
            else:
                _, action, _ = self.policy.actor(np.array([obs]))
                action = action[0].numpy()

            next_obs, reward, done, _ = self.env.step(action)

            if self.inference_fn is not None:
                reward = self.inference_fn(obs, action)[0][0]

            if (self.max_episode_steps is not None
                    and episode_timesteps + 1 == self.max_episode_steps):
                done = True

            if not done or episode_timesteps + 1 == self.max_episode_steps:  # pylint: disable=protected-access
                mask = 1.0
            else:
                mask = 0.0

            self.replay_buffer.add_batch(
                (np.array([obs.astype(np.float32)]),
                 np.array([action.astype(np.float32)]),
                 np.array([next_obs.astype(np.float32)
                           ]), np.array([[reward]]).astype(np.float32),
                 np.array([[mask]]).astype(np.float32)))

            episode_return += reward
            episode_timesteps += 1
            total_timesteps += 1

            obs = next_obs

            if done:
                if episode_timesteps > 0:
                    current_time = time.time()

                    self.logger.log('common/training_return', episode_return)
                    self.logger.log(
                        'common/fps',
                        episode_timesteps / (current_time - previous_time))

                obs = self.env.reset()
                episode_return = 0
                episode_timesteps = 0
                previous_time = time.time()

                if self._use_init_states:
                    self.init_replay_buffer.add_batch(
                        np.array([obs.astype(np.float32)]))

            if total_timesteps >= self._start_training_timesteps:
                for _ in range(self._updates_per_step):
                    return_dict = self.policy.train(
                        **self._get_policy_train_kwargs())
                    for k, v in return_dict.items():
                        self.logger.log(k, v)

            if total_timesteps % self._eval_interval == 0:
                average_returns, evaluation_timesteps = self.evaluate(
                    total_timesteps)
                self.logger.log('eval/average_test_return', average_returns)
                self.logger.log('eval/average_test_episode_length',
                                evaluation_timesteps)
                self.logger.dump(total_timesteps,
                                 dump_to_csv=True,
                                 dump_eval=True)
                logging.info('Eval: ave returns=%f, ave episode length=%f',
                             average_returns, evaluation_timesteps)
            if done:
                self.logger.dump(total_timesteps, dump_to_csv=True)

        self.logger.dump(total_timesteps,
                         dump_eval=True,
                         flush_tb=True,
                         dump_to_csv=True)

    def evaluate(self, total_timesteps, num_episodes=10):
        eval_total_timesteps = 0
        total_returns = 0

        for i in range(num_episodes):
            self.video_recorder.init(enabled=(i == 0))
            state = self.eval_env.reset()
            done = False
            episode_timesteps = 0
            while not done:
                action, _, _ = self.policy.actor(np.array([state]))
                action = action[0].numpy()

                next_state, reward, done, _ = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                if (self.max_episode_steps is not None
                        and episode_timesteps + 1 == self.max_episode_steps):
                    done = True

                total_returns += reward
                eval_total_timesteps += 1
                episode_timesteps += 1
                state = next_state

            if self._save_test_movie and total_timesteps in self._save_movie_indices:
                self.video_recorder.save(f'{total_timesteps}.mp4',
                                         logger=self.logger,
                                         step=total_timesteps)

        return total_returns / num_episodes, eval_total_timesteps / num_episodes

    def _get_policy_train_kwargs(self):
        states, actions, next_states, rewards, masks = next(
            self.replay_buffer_iter)[0]

        target_entropy = (-self.env.action_space.shape[0]
                          if self._target_entropy is None else
                          self._target_entropy)
        kwargs = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'rewards': rewards,
            'masks': masks,
            'target_entropy': target_entropy
        }

        return kwargs
