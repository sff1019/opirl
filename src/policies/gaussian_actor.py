import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-6


class Actor(tf.keras.Model):
    """Gaussian policy with TanH squashing."""
    def __init__(self,
                 state_dim,
                 action_dim,
                 loaded_actor=None,
                 use_multivariate=False):
        """Creates an actor.

    Args:
      state_dim: State size.
      action_dim: Action size.
    """
        super(Actor, self).__init__()
        self._use_multivariate = use_multivariate
        if loaded_actor is not None:
            self.trunk = loaded_actor
        else:
            self.trunk = tf.keras.Sequential([
                tf.keras.layers.Dense(256,
                                      input_shape=(state_dim, ),
                                      activation=tf.nn.relu,
                                      kernel_initializer='orthogonal'),
                tf.keras.layers.Dense(256,
                                      activation=tf.nn.relu,
                                      kernel_initializer='orthogonal'),
                tf.keras.layers.Dense(2 * action_dim,
                                      kernel_initializer='orthogonal')
            ],
                                             name='actor')

    def get_dist_and_mode(self, states):
        out = self.trunk(states)
        mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
        mode = tf.nn.tanh(mu)

        log_std = tf.nn.tanh(log_std)
        assert LOG_STD_MAX > LOG_STD_MIN
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std +
                                                                     1)
        std = tf.exp(log_std)

        if self._use_multivariate:
            dist = tfp.distributions.MultivariateNormalDiag(loc=mu,
                                                            scale_diag=std)
        else:
            dist = ds.TransformedDistribution(
                ds.Sample(ds.Normal(tf.zeros(mu.shape[:-1]),
                                    1.0,
                                    validate_args=True),
                          sample_shape=mu.shape[-1:]),
                tfp.bijectors.Chain([
                    tfp.bijectors.Tanh(validate_args=True),
                    tfp.bijectors.Shift(shift=mu),
                    tfp.bijectors.ScaleMatvecDiag(scale_diag=std)
                ]))

        return dist, mode

    @tf.function
    def get_log_prob(self, states, actions):
        dist, mode = self.get_dist_and_mode(states)
        log_probs = dist.log_prob(actions)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs

    @tf.function
    def call(self, states):
        dist, mode = self.get_dist_and_mode(states)
        samples = dist.sample()
        log_probs = dist.log_prob(samples)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return mode, samples, log_probs
