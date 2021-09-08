"""
Actor Critic for policy update in OPIRL.
The code is built on top of AlgaeDICE: https://github.com/google-research/google-research/tree/master/algae_dice
"""
import tensorflow as tf
import tensorflow_probability as tfp

from src.algos.policy_base import OffPolicyAgent
import src.misc.keras_utils as keras_utils
from src.policies.gaussian_actor import Actor

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def soft_update(net, target_net, tau=0.005):
    for var, target_var in zip(net.variables, target_net.variables):
        new_value = var * tau + target_var * (1 - tau)
        target_var.assign(new_value)


class DoubleCritic(tf.keras.Model):
    """A critic network that estimates a dual Q-function."""
    def __init__(self, state_dim, action_dim):
        super(DoubleCritic, self).__init__()
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


class Policy(OffPolicyAgent):
    """Class performing algae training."""
    def __init__(self,
                 observation_space,
                 action_space,
                 state_dim,
                 action_dim,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 discount=0.99,
                 tau=0.005,
                 alpha_init=1.0,
                 learn_alpha=True,
                 algae_alpha=1.0,
                 use_init_states=True,
                 bc_reg_coeff=1.0,
                 actor_coeff=1.e-3,
                 use_bc_reg=False,
                 use_multivariate=False,
                 exponent=2.0):

        super().__init__(name='opirl_poliy',
                         observation_space=observation_space,
                         action_space=action_space,
                         memory_capacity=None,
                         n_warmup=None)

        self.actor = Actor(state_dim, action_dim, use_multivariate=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.avg_actor_loss = tf.keras.metrics.Mean('actor_loss',
                                                    dtype=tf.float32)
        self.avg_bc_loss = tf.keras.metrics.Mean('bc_loss', dtype=tf.float32)
        self.avg_alpha_loss = tf.keras.metrics.Mean('alpha_loss',
                                                    dtype=tf.float32)
        self.avg_actor_entropy = tf.keras.metrics.Mean('actor_entropy',
                                                       dtype=tf.float32)
        self.avg_alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)
        self.use_init_states = use_init_states

        self.critic = DoubleCritic(state_dim, action_dim)
        self.critic_target = DoubleCritic(state_dim, action_dim)
        soft_update(self.critic, self.critic_target, tau=1.0)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr)
        self.avg_critic_loss = tf.keras.metrics.Mean('critic_loss',
                                                     dtype=tf.float32)

        self.discount = discount
        self.tau = tau

        self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
        self.learn_alpha = learn_alpha
        self.alpha_optimizer = tf.keras.optimizers.Adam()

        self.algae_alpha = algae_alpha
        self.exponent = exponent
        self.f = lambda resid: tf.pow(tf.abs(resid), self.exponent
                                      ) / self.exponent
        clip_resid = lambda resid: tf.clip_by_value(resid, 0.0, 1e6)
        self.fgrad = lambda resid: tf.pow(clip_resid(resid), self.exponent - 1)

        self.use_bc_reg = use_bc_reg
        self.bc_reg_coeff = bc_reg_coeff
        self.actor_coeff = actor_coeff

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @tf.function
    def get_logp(self, s, a):
        return self.actor.get_log_prob(s, a)

    @tf.function
    def get_action_and_logp(self, s):
        return self.actor(s)

    def critic_mix(self, s, a):
        target_q1, target_q2 = self.critic_target(s, a)
        target_q = tf.minimum(target_q1, target_q2)
        q1, q2 = self.critic(s, a)
        return q1 * 0.05 + target_q * 0.95, q2 * 0.05 + target_q * 0.95,

    def fit_critic(self, states, actions, next_states, rewards, masks,
                   init_states):
        _, init_actions, _ = self.actor(init_states)
        _, next_actions, next_log_probs = self.actor(next_states)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.critic.variables)

            target_q1, target_q2 = self.critic_mix(next_states, next_actions)

            target_q1 = target_q1 - self.alpha * next_log_probs
            target_q2 = target_q2 - self.alpha * next_log_probs

            target_q1 = rewards + self.discount * masks * target_q1
            target_q2 = rewards + self.discount * masks * target_q2

            q1, q2 = self.critic(states, actions)
            init_q1, init_q2 = self.critic(init_states, init_actions)

            critic_loss1 = tf.reduce_mean(
                self.f(target_q1 - q1) +
                (1 - self.discount) * init_q1 * self.algae_alpha)

            critic_loss2 = tf.reduce_mean(
                self.f(target_q2 - q2) +
                (1 - self.discount) * init_q2 * self.algae_alpha)

            critic_loss = (critic_loss1 + critic_loss2)

        critic_grads = tape.gradient(critic_loss,
                                     self.critic.trainable_variables)

        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.variables))

        return critic_loss

    def fit_actor(self, states, actions, next_states, rewards, masks,
                  target_entropy, init_states):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor.variables)
            _, init_actions, _ = self.actor(init_states)
            _, next_actions, next_log_probs = self.actor(next_states)

            target_q1, target_q2 = self.critic_mix(next_states, next_actions)
            target_q1 = target_q1 - self.alpha * next_log_probs
            target_q2 = target_q2 - self.alpha * next_log_probs
            target_q1 = rewards + self.discount * masks * target_q1
            target_q2 = rewards + self.discount * masks * target_q2

            q1, q2 = self.critic(states, actions)
            init_q1, init_q2 = self.critic(init_states, init_actions)

            actor_loss1 = -tf.reduce_mean(
                tf.stop_gradient(self.fgrad(target_q1 - q1)) *
                (target_q1 - q1) +
                (1 - self.discount) * init_q1 * self.algae_alpha)

            actor_loss2 = -tf.reduce_mean(
                tf.stop_gradient(self.fgrad(target_q2 - q2)) *
                (target_q2 - q2) +
                (1 - self.discount) * init_q2 * self.algae_alpha)

            actor_loss = (actor_loss1 + actor_loss2) / 2.0
            if self.use_bc_reg:
                bc_loss = self.bc_reg_coeff * self.bc_loss(states, actions)
                actor_loss = self.actor_coeff * actor_loss + bc_loss
            else:
                bc_loss = 0.0

        actor_grads = tape.gradient(actor_loss, self.actor.variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.variables))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([self.log_alpha])
            alpha_loss = tf.reduce_mean(self.alpha *
                                        (-next_log_probs - target_entropy))

        if self.learn_alpha:
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_grads, [self.log_alpha]))

        return actor_loss, alpha_loss, -next_log_probs, bc_loss

    def bc_loss(self, states, actions):
        _, sampled_action, _ = self.actor(states)

        q1, q2 = self.critic(states, actions)
        sampled_q1, sampled_q2 = self.critic(states, sampled_action)

        q_filter = (q1 + q2) > (sampled_q1 + sampled_q2)

        bc_loss = (sampled_action - actions)**2
        # Apply Q-filter
        bc_loss = tf.reduce_mean(tf.where(q_filter, bc_loss, 0.0))

        return bc_loss

    @tf.function
    def train(self,
              states,
              actions,
              next_states,
              rewards,
              masks,
              init_states,
              target_entropy=0,
              actor_update_freq=2):
        critic_loss = self.fit_critic(states, actions, next_states, rewards,
                                      masks, init_states)
        return_dict = {
            'train/critic_loss': critic_loss,
            'train/actor_loss': self.avg_actor_loss.result(),
            'train/bc_loss': self.avg_bc_loss.result(),
            'train/alpha_loss': self.avg_alpha_loss.result(),
            'train/actor_entropy': self.avg_actor_entropy.result(),
            'train/alpha': self.alpha
        }

        if tf.equal(self.critic_optimizer.iterations % actor_update_freq, 0):
            actor_loss, alpha_loss, entropy, bc_loss = self.fit_actor(
                states, actions, next_states, rewards, masks, target_entropy,
                init_states)
            soft_update(self.critic, self.critic_target, tau=self.tau)

            self.avg_actor_loss(actor_loss)
            self.avg_bc_loss(bc_loss)
            self.avg_alpha_loss(alpha_loss)
            self.avg_actor_entropy(entropy)
            self.avg_alpha(self.alpha)

            return_dict['train/actor_loss'] = self.avg_actor_loss.result()
            return_dict['train/alpha_loss'] = self.avg_alpha_loss.result()
            return_dict['train/actor_entropy'] = self.avg_actor_entropy.result(
            )
            return_dict['train/bc_loss'] = self.avg_bc_loss.result()
            keras_utils.my_reset_states(self.avg_actor_loss)
            keras_utils.my_reset_states(self.avg_bc_loss)
            keras_utils.my_reset_states(self.avg_alpha_loss)
            keras_utils.my_reset_states(self.avg_actor_entropy)
            keras_utils.my_reset_states(self.avg_alpha)

        return return_dict
