import gym

from src.envs.custom_ant_env import CustomAntEnv
from src.envs.point_maze_env import PointMazeEnv


def initialize_env(env_name, seed=0, inference_fn=None):
    """
    Param:
        env_name: Name of the OpenAI gym environment.
    """
    if env_name == 'PointMaze-Left':
        env = PointMazeEnv(direction=0)
    elif env_name == 'PointMaze-Right':
        env = PointMazeEnv(direction=1, inference_fn=inference_fn)
    elif env_name == 'CustomAnt':
        env = CustomAntEnv(gear=30, amputated=False)
    elif env_name == 'AmputatedAnt':
        env = CustomAntEnv(gear=30, amputated=True, inference_fn=inference_fn)
    elif env_name == 'BigAnt':
        env = CustomAntEnv(gear=150, big=True, inference_fn=inference_fn)
    else:
        env = gym.make(env_name)

    env.seed(seed)
    env.action_space.seed(seed)

    return env
