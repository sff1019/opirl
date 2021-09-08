import tensorflow as tf

from src.algos.policy_base import IRLAgent

Dense = tf.keras.layers.Dense
EPS = 1.e-6


class AIRLGP(IRLAgent):
    def __init__(self,
                 observation_space,
                 action_space,
                 discount=0.9,
                 disc_lr=1.e-5,
                 grad_penalty_coeff=10,
                 is_state_only=False,
                 reward_shaping=False,
                 use_discrim_as_reward=False,
                 rew_clip_max=None,
                 rew_clip_min=None,
                 gpu=0,
                 name='airl',
                 **kwargs):
        super().__init__(name,
                         observation_space,
                         action_space,
                         n_training=1,
                         **kwargs)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self._is_state_only = is_state_only
        self._reward_shaping = reward_shaping
        self._use_discrim_as_reward = use_discrim_as_reward
        self.rew_clip_max = rew_clip_max
        self.rew_clip_min = rew_clip_min

        if is_state_only:
            self.rew_net = tf.keras.Sequential([
                Dense(64, input_shape=(obs_dim, ), activation=tf.nn.relu),
                Dense(64, activation=tf.nn.relu),
                Dense(1)
            ])
        else:
            self.rew_net = tf.keras.Sequential([
                Dense(64,
                      input_shape=(obs_dim + act_dim, ),
                      activation=tf.nn.relu),
                Dense(64, activation=tf.nn.relu),
                Dense(1)
            ])
        self.rew_optim = tf.keras.optimizers.Adam(learning_rate=disc_lr)

        self.val_net = tf.keras.Sequential([
            Dense(256, input_shape=(obs_dim, ), activation=tf.nn.relu),
            Dense(256, activation=tf.nn.relu),
            Dense(1)
        ])
        self.val_optim = tf.keras.optimizers.Adam(learning_rate=disc_lr)

        self.rew_clip_max = rew_clip_max
        self.rew_clip_min = rew_clip_min

        self.grad_penalty_coeff = grad_penalty_coeff
        self.discount = discount
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    @tf.function
    def get_occupancy_ratio(self, states, actions):
        inputs = tf.concat([states, actions], -1)
        return tf.exp(self.discriminator(inputs))

    @tf.function
    def inference(self, states, actions, next_states=None):
        if len(states.shape) == len(actions.shape) == 1:
            states = tf.expand_dims(states, axis=0)
            states = tf.cast(states, tf.float32)
            actions = tf.expand_dims(actions, axis=0)
            actions = tf.cast(actions, tf.float32)

        if self._is_state_only:
            inputs = states
        else:
            inputs = tf.concat([states, actions], -1)
        rewards = self.rew_net(inputs, training=False)
        # Add reward shaping term
        if self._reward_shaping:
            assert next_states is not None, "Need to specify next_states, cannot be None"  # noqa
            rewards += self.discount * self.val_net(
                next_states, training=False) - self.val_net(states,
                                                            training=False)

        return rewards

    @tf.function
    def predict_reward(self, states, actions, next_states=None, logps=None):
        reward = self.inference(states, actions, next_states=next_states)

        if self.rew_clip_max is not None or self.rew_clip_min is not None:
            clip_max = tf.int32.max if self.rew_clip_max is None else self.rew_clip_max  # noqa
            clip_min = tf.int32.min if self.rew_clip_min is None else self.rew_clip_min  # noqa
            reward = tf.clip_by_value(reward, clip_min, clip_max)
        return reward

    @tf.function
    def update(self, expert_states, expert_actions, expert_next_states,
               policy_states, policy_actions, policy_next_states, actor):
        with tf.device(self.device):
            if self._is_state_only:
                policy_inputs = policy_states
                expert_inputs = expert_states
            else:
                policy_inputs = tf.concat([policy_states, policy_actions], -1)
                expert_inputs = tf.concat([expert_states, expert_actions], -1)

            alpha_rew = tf.random.uniform(shape=(policy_inputs.get_shape()[0],
                                                 1))
            alpha_val = tf.random.uniform(shape=(policy_inputs.get_shape()[0],
                                                 1))
            inter_rew = alpha_rew * policy_inputs + (1 -
                                                     alpha_rew) * expert_inputs
            inter_val = alpha_val * policy_states + (1 -
                                                     alpha_val) * expert_states

            with tf.GradientTape(watch_accessed_variables=False,
                                 persistent=True) as tape:
                tape.watch(self.rew_net.variables)
                tape.watch(self.val_net.variables)

                real_rewards = self.rew_net(tf.concat(expert_inputs, axis=-1))
                fake_rewards = self.rew_net(tf.concat(policy_inputs, axis=-1))

                # Compute value function shaping
                real_values = self.val_net(expert_states)
                real_next_values = self.val_net(expert_next_states)
                fake_values = self.val_net(policy_states)
                fake_next_values = self.val_net(policy_next_states)

                # Compute f(s, a, s')
                real_logps = (real_rewards + self.discount * real_next_values -
                              real_values)
                fake_logps = (fake_rewards + self.discount * fake_next_values -
                              fake_values)

                with tape.stop_recording():
                    real_logqs = actor.get_log_prob(policy_states,
                                                    policy_actions)
                    fake_logqs = actor.get_log_prob(expert_states,
                                                    expert_actions)

                real_logpq = tf.concat([real_logps, real_logqs], axis=1)
                real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logpq), logits=real_logpq)
                fake_logpq = tf.concat([fake_logps, fake_logqs], axis=1)
                fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_logpq), logits=fake_logpq)
                classification_loss = tf.reduce_mean(real_loss) \
                    + tf.reduce_mean(fake_loss)

                # Compute gradient penalty
                with tf.GradientTape(watch_accessed_variables=False) as tape2:
                    tape2.watch(inter_rew)
                    output = self.rew_net(inter_rew)

                grad_rew = tape2.gradient(output, [inter_rew])[0]
                grad_penalty = tf.reduce_mean(
                    tf.pow(tf.norm(grad_rew, axis=-1) - 1, 2))

                with tf.GradientTape(watch_accessed_variables=False) as tape3:
                    tape3.watch(inter_val)
                    output = self.val_net(inter_val)
                grad_val = tape3.gradient(output, [inter_val])[0]
                grad_penalty += tf.reduce_mean(
                    tf.pow(tf.norm(grad_val, axis=-1) - 1, 2))

                total_loss = (classification_loss +
                              self.grad_penalty_coeff * grad_penalty)

            grads_rews = tape.gradient(total_loss, self.rew_net.variables)
            grads_vals = tape.gradient(total_loss, self.val_net.variables)
            self.rew_optim.apply_gradients(
                zip(grads_rews, self.rew_net.variables))
            self.val_optim.apply_gradients(
                zip(grads_vals, self.val_net.variables))

        return_dict = {
            'train/gail_classification_loss': classification_loss,
            'train/gail_gradient_penalty': grad_penalty,
            'train/gail_loss': total_loss
        }

        return return_dict
