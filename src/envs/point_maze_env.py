"""
MIT License

Copyright (c) 2021 Kei Ohta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from src.envs.dynamic_mjc import point_mass_maze


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 direction=0,
                 maze_length=0.6,
                 sparse_reward=False,
                 no_reward=False,
                 discrete=True,
                 episode_length=100,
                 inference_fn=None):
        """
            LEFT = 0
            RIGHT = 1
            NO = 2
        """
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self._max_episode_steps = episode_length
        self.direction = direction
        self.length = maze_length
        self.discrete = discrete  # if use discrete initial positions
        self.episode_length = 0
        self.policy_contexts = None

        # For transfer
        self.inference_fn = inference_fn

        model = point_mass_maze(direction=self.direction, length=self.length)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a, particle_pos=None):
        if particle_pos is None:
            particle_pos = self.get_body_com('particle')

        vec_dist = particle_pos - self.get_body_com("target")

        reward_dist = -np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = -np.square(a).sum()
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            # max: 0, min: -0.9919494936611665 per state, action pair
            original_reward = reward_dist + 0.001 * reward_ctrl

        if (self._get_obs is not None) and (self.inference_fn is not None):
            s = np.concatenate([self._get_obs(), [0]], -1)
            reward_network = np.squeeze(self.inference_fn(s, a).numpy())
        else:
            reward_network = 0

        self.do_simulation(a, self.frame_skip)
        self.episode_length += 1
        done = self.episode_length >= self._max_episode_steps

        # Compute reward using given reward function
        if self.inference_fn is not None:
            reward = reward_network
        else:
            reward = original_reward

        return self._get_obs(), reward, done, dict(
            original_reward=original_reward,
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.5
        self.viewer.cam.lookat[0] -= 0.05
        # self.viewer.cam.lookat[1] += 0.5
        # self.viewer.cam.lookat[2] += 0.5

    # def viewer_setup(self, mode='rgb_array'):
    #     if self.viewer is None:
    #         self._get_viewer(mode)
    #
    #     # self.viewer.cam.trackbodyid = -1
    #     # self.viewer.cam.distance = 1.3

    def reset_model(self, reset_args=None, policy_contexts=None):
        self.policy_contexts = policy_contexts

        # Randomly choose goal location
        if reset_args is None:
            target_pos = [0., 0.5, 0.]
            if self.discrete:
                target_pos_x = np.random.choice(np.arange(0., 0.5, 0.04))
                target_pos[0] = target_pos_x
            else:
                gaussian_mean_list = [0.1, 0.3, 0.5]
                while True:
                    target_pos_x = np.random.normal(
                        loc=np.random.choice(gaussian_mean_list), scale=0.05)
                    if target_pos_x >= 0. and target_pos_x <= 0.6:
                        target_pos[0] = target_pos_x
                        break
        else:
            target_pos = reset_args

        # Directly overwrite body_pos to update goal location
        body_pos = self.model.body_pos.copy()
        body_pos[2] = target_pos
        self.model.body_pos[self.model._body_name2id['target']] = target_pos

        # Update state of the agent
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01)
        qpos = qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.05, high=0.05)
        self.set_state(qpos, qvel)

        self.episode_length = 0
        return self._get_obs()

    def reset(self, reset_args=None, policy_contexts=None):
        ob = self.reset_model(reset_args=reset_args,
                              policy_contexts=policy_contexts)
        return ob

    def _get_obs(self):
        if self.policy_contexts is not None:
            return np.concatenate(
                [self.get_body_com("particle"), self.policy_contexts])
        return np.concatenate(
            [self.get_body_com("particle"),
             self.get_body_com("target")])


if __name__ == "__main__":
    env = PointMazeEnv()
    env.reset()
    for _ in range(10):
        env.reset()
        for _ in range(100):
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()
