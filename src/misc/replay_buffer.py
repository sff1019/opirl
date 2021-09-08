from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [
            1,
        ]  # space.n
    else:
        raise NotImplementedError(
            "Assuming to use Box or Discrete, not {}".format(type(space)))


def get_replay_buffer_spec(env):
    spec = (
        tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32,
                               'observation'),
        tensor_spec.TensorSpec([env.action_space.shape[0]], tf.float32,
                               'action'),
        tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32,
                               'next_observation'),
        tensor_spec.TensorSpec([1], tf.float32, 'reward'),
        tensor_spec.TensorSpec([1], tf.float32, 'mask'),
        tensor_spec.TensorSpec([1], tf.float32, 'logp'),
    )

    return spec


def get_replay_buffer(env, size=None, spec=None):
    if spec is None:
        spec = get_replay_buffer_spec(env)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=1, max_length=size)

    return replay_buffer


def add_samples_to_replay_buffer(replay_buffer, obs, action, next_obs, logp):
    replay_buffer.add_batch((
        np.array([obs.astype(np.float32)]),
        np.array([action.astype(np.float32)]),
        np.array([next_obs.astype(np.float32)]),
        np.array([[0]]).astype(np.float32),
        np.array([[1.0]]).astype(np.float32),
        np.array(logp).astype(np.float32),
    ))
