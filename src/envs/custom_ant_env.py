from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np

from src.envs.dynamic_mjc.mjc_models import (angry_ant_crippled, ant_env,
                                             big_ant_env)


class CustomAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 max_timesteps=1e3,
                 amputated=False,
                 big=False,
                 gear=150,
                 inference_fn=None):
        utils.EzPickle.__init__(self)
        self.timesteps = 0
        self._max_episode_steps = int(max_timesteps)
        self.max_timesteps = max_timesteps

        self.inference_fn = inference_fn

        if amputated:
            model = angry_ant_crippled(gear=gear)
        elif big:
            model = big_ant_env(gear=gear)
        else:
            model = ant_env(gear=gear)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        vel = self.sim.data.qvel.flat[0]
        forward_reward = vel
        # if (self._get_obs is not None) and (self.inference_fn is not None):
        #     s = np.concatenate([self._get_obs(), [0]], -1)
        #     reward_network = self.inference_fn(s, a).numpy()
        # else:
        #     reward_network = 0

        self.do_simulation(a, self.frame_skip)

        ctrl_cost = .01 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        state = self.state_vector()
        flipped = not (state[2] >= 0.2)
        flipped_rew = -1 if flipped else 0

        original_reward = forward_reward - ctrl_cost - contact_cost + flipped_rew
        # Compute reward using given reward function
        # if self.inference_fn is not None:
        #     reward = reward_network
        # else:
        reward = original_reward

        self.timesteps += 1
        done = self.timesteps >= self.max_timesteps

        ob = self._get_obs()
        return ob, reward, done, dict(original_reward=reward,
                                      reward_forward=forward_reward,
                                      reward_ctrl=-ctrl_cost,
                                      reward_contact=-contact_cost,
                                      reward_flipped=flipped_rew)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        self.timesteps = 0
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self, mode='rgb_array'):
        if self.viewer is None:
            self._get_viewer(mode)
        self.viewer.cam.distance = self.model.stat.extent * 0.4


if __name__ == "__main__":
    env = CustomAntEnv(amputated=True, gear=30)
    env.reset()
    for _ in range(10):
        env.reset()
        for _ in range(1000):
            obs, rew, done, infor = env.step(env.action_space.sample())
            env.render()
