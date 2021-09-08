from gym.spaces import Discrete, Box
import tensorflow as tf


def get_action_components(action_space):
    if isinstance(action_space, Discrete):
        act_dim = action_space.n
        act_limit = 1.0
    elif isinstance(action_space, Box):
        act_dim = action_space.low.size
        act_limit = action_space.high[0]
    else:
        raise NotImplementedError

    return act_dim, act_limit


class Policy(tf.keras.Model):
    def __init__(self,
                 name,
                 observation_space,
                 action_space,
                 update_interval=1,
                 batch_size=256,
                 discount=0.99,
                 n_warmup=0,
                 n_training=1,
                 max_grad=10.,
                 memory_capacity=int(1e6),
                 gpu=0):
        super().__init__()
        self.policy_name = name
        self.obs_shape = observation_space.shape
        self.obs_dim = observation_space.shape[0]
        # This is used to check if input state to `get_action` is multiple
        # (batch) or single
        self.obs_ndim = len(observation_space.shape)
        if action_space is not None:
            self.act_dim, self.act_limit = get_action_components(action_space)
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.discount = discount
        self.n_warmup = n_warmup
        self.n_training = n_training
        self.memory_capacity = memory_capacity
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            import argparse
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add_argument('--n_warmup', type=int, default=int(1e4))
        parser.add_argument('--n_training', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=3.e-4)
        parser.add_argument('--gpu', type=int, default=0, help='GPU id')
        return parser


class OffPolicyAgent(Policy):
    """
    Base class for off-policy agents
    """
    def __init__(self, memory_capacity, **kwargs):
        super().__init__(memory_capacity=memory_capacity, **kwargs)

    @staticmethod
    def get_argument(parser=None):
        parser = Policy.get_argument(parser)
        parser.add_argument('--memory_capacity', type=int, default=int(1e6))
        return parser


class IRLAgent(Policy):
    def __init__(self,
                 name,
                 observation_space,
                 action_space,
                 n_training=1,
                 memory_capacity=0,
                 **kwargs):
        super().__init__(name,
                         observation_space,
                         action_space,
                         memory_capacity=memory_capacity,
                         **kwargs)
        self.n_training = n_training
