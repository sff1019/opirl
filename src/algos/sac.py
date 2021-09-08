import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from src.algos.policy_base import OffPolicyAgent
from src.misc import keras_utils
from src.misc.target_update_ops import update_target_variables
from src.policies.gaussian_actor import Actor

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = np.finfo(np.float32).eps


class Critic(tf.keras.Model):
    """A critic network that estimates a dual Q-function."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.critic1 = tf.keras.Sequential([
            tf.keras.layers.Dense(256,
                                  input_shape=(state_dim + action_dim, ),
                                  activation=tf.nn.relu,
                                  kernel_initializer='orthogonal'),
            tf.keras.layers.Dense(256,
                                  activation=tf.nn.relu,
                                  kernel_initializer='orthogonal'),
            tf.keras.layers.Dense(1, kernel_initializer='orthogonal')
        ])
        self.critic2 = tf.keras.Sequential([
            tf.keras.layers.Dense(256,
                                  input_shape=(state_dim + action_dim, ),
                                  activation=tf.nn.relu,
                                  kernel_initializer='orthogonal'),
            tf.keras.layers.Dense(256,
                                  activation=tf.nn.relu,
                                  kernel_initializer='orthogonal'),
            tf.keras.layers.Dense(1, kernel_initializer='orthogonal')
        ])

    @tf.function
    def call(self, states, actions):
        x = tf.concat([states, actions], -1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2


class SAC(OffPolicyAgent):
    """Class performing Soft Actor Critic training."""
    def __init__(self,
                 observation_space,
                 action_space,
                 state_dim,
                 action_dim,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 alpha_init=1.0,
                 learn_alpha=True,
                 discount=0.99,
                 tau=5.e-3,
                 memory_capacity=int(1e6),
                 n_warmup=int(1e4),
                 has_absorbing_states=True,
                 use_multivariate=False,
                 name='sac'):
        super().__init__(name=name,
                         observation_space=observation_space,
                         action_space=action_space,
                         memory_capacity=memory_capacity,
                         n_warmup=n_warmup)

        self.actor = Actor(state_dim,
                           action_dim,
                           use_multivariate=use_multivariate)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        update_target_variables(self.critic_target.weights,
                                self.critic.weights,
                                tau=1.)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr)
        self.avg_critic_loss = tf.keras.metrics.Mean('critic_loss',
                                                     dtype=tf.float32)

        self._discount = discount
        self._tau = tau
        self._has_absorbing_states = has_absorbing_states

        self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
        self.learn_alpha = learn_alpha
        self.alpha_optimizer = tf.keras.optimizers.Adam()

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @tf.function
    def get_logp(self, s, a):
        return self.actor.get_log_prob(s, a)

    @tf.function
    def get_action_and_logp(self, s):
        return self.actor(s)

    def fit_critic(self, states, actions, next_states, rewards, masks):
        _, next_actions, log_probs = self.actor(next_states)

        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_v = tf.minimum(target_q1, target_q2) - self.alpha * log_probs
        target_q = rewards + self._discount * masks * target_v

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.critic.variables)

            q1, q2 = self.critic(states, actions)
            critic_loss = (tf.losses.mean_squared_error(target_q, q1) +
                           tf.losses.mean_squared_error(target_q, q2))
            critic_loss = tf.reduce_mean(critic_loss)

        critic_grads = tape.gradient(critic_loss, self.critic.variables)

        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.variables))

        return critic_loss

    def fit_actor(self, states, target_entropy):
        if self._has_absorbing_states:
            is_non_absorbing_mask = tf.cast(tf.equal(states[:, -1:], 0.0),
                                            tf.float32)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor.variables)
            _, actions, log_probs = self.actor(states)
            q1, q2 = self.critic(states, actions)
            q = tf.minimum(q1, q2)
            if self._has_absorbing_states:
                actor_loss = tf.reduce_sum(
                    is_non_absorbing_mask * (self.alpha * log_probs - q)) / (
                        tf.reduce_sum(is_non_absorbing_mask) + EPS)
            else:
                actor_loss = tf.reduce_mean(self.alpha * log_probs - q)

            actor_loss += keras_utils.orthogonal_regularization(
                self.actor.trunk)

        actor_grads = tape.gradient(actor_loss, self.actor.variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([self.log_alpha])
            if self._has_absorbing_states:
                alpha_loss = tf.reduce_sum(
                    is_non_absorbing_mask * self.alpha *
                    (-log_probs - target_entropy)) / (
                        tf.reduce_sum(is_non_absorbing_mask) + EPS)
            else:
                alpha_loss = -tf.reduce_mean(
                    (self.alpha *
                     tf.stop_gradient(log_probs + target_entropy)))

        if self.learn_alpha:
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_grads, [self.log_alpha]))

        return actor_loss, alpha_loss, -log_probs

    @tf.function
    def train(self,
              states,
              actions,
              next_states,
              rewards,
              masks,
              target_entropy=0):
        critic_loss = self.fit_critic(states, actions, next_states, rewards,
                                      masks)

        self.avg_critic_loss(critic_loss)
        return_dict = {'train/critic_loss': self.avg_critic_loss.result()}
        keras_utils.my_reset_states(self.avg_critic_loss)

        actor_loss, alpha_loss, entropy = self.fit_actor(
            states, target_entropy)
        update_target_variables(self.critic_target.weights,
                                self.critic.weights,
                                tau=self._tau)

        return_dict['train/actor_loss'] = actor_loss
        return_dict['train/alpha_loss'] = alpha_loss
        return_dict['train/actor_entropy'] = entropy
        return_dict['train/alpha'] = self.alpha

        return return_dict

    @tf.function
    def train_bc(self, expert_states, expert_actions):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor.variables)
            log_probs = self.actor.get_log_prob(expert_states, expert_actions)
            actor_loss = tf.reduce_mean(
                -log_probs) + keras_utils.orthogonal_regularization(
                    self.actor.trunk)

        actor_grads = tape.gradient(actor_loss, self.actor.variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.variables))

        return_dict = {'train/actor_loss': actor_loss}
        return return_dict
