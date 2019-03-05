"""
DQN model.

DQN, also known as Deep Q-Network, [more explanation].
"""
import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm


class DQN(OffPolicyRLAlgorithm):
    """A DQN model based on https://arxiv.org/pdf/1509.02971.pdf."""

    def __init__(self,
                 env,
                 replay_buffer,
                 max_path_length=200,
                 qf_lr=0.001,
                 qf_optimizer=tf.train.AdamOptimizer,
                 discount=1.0,
                 name=None,
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 double_q=False,
                 **kwargs):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.double_q = double_q

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            discount=discount,
            **kwargs)

    @overrides
    def init_opt(self):
        """
        Initialize the networks and Ops.

        Assume discrete space for dqn, so action dimension
        will always be action_space.n
        """
        obs_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n

        self.obs_input = self.qf.model.networks['default'].input
        self.q_val = self.qf.model.networks['default'].output

        # build q networks
        with tf.name_scope(self.name, "DQN"):
            self.action_t_ph = tf.placeholder(tf.int32, None, name="action")
            self.reward_t_ph = tf.placeholder(tf.float32, None, name="reward")
            self.done_t_ph = tf.placeholder(tf.float32, None, name="done")
            self.next_obs_t_ph = tf.placeholder(
                tf.float32, (None, ) + obs_dim, name="next_obs")

            self.target_qval = self.qf.build_net(
                name="target_qf", input_var=self.next_obs_t_ph)

            with tf.name_scope("update_ops"):
                self._qf_update_ops = get_target_ops(
                    self.qf.get_global_vars(),
                    self.qf.get_global_vars("target_qf"))

            with tf.name_scope("td_error"):
                # Q-value of the selected action
                q_selected = tf.reduce_sum(
                    self.q_val * tf.one_hot(self.action_t_ph, action_dim),
                    axis=1)

                # r + Q'(s', argmax_a(Q(s', _)) - Q(s, a)
                if self.double_q:
                    target_qval_with_online_q = self.qf.get_qval_sym(self.qf.name, self.next_obs_t_ph)
                    future_best_q_val_action = tf.argmax(target_qval_with_online_q, 1)
                    future_best_q_val = tf.reduce_sum(self.target_qval * tf.one_hot(future_best_q_val_action, action_dim),
                        axis=1)
                else:
                # r + max_a(Q'(s', _)) - Q(s, a)
                    future_best_q_val = tf.reduce_max(self.target_qval, axis=1)

                q_best_masked = (1.0 - self.done_t_ph) * future_best_q_val
                # if done, it's just reward
                # else reward + discount * future_best_q_val
                target_q_values = self.reward_t_ph + self.discount * q_best_masked

                td_error = tf.stop_gradient(target_q_values) - q_selected
                loss = tf.square(td_error)
                self._loss = tf.reduce_mean(loss)

            with tf.name_scope("optimize_ops"):
                optimizer = self.qf_optimizer(self.qf_lr)
                if self.grad_norm_clipping is not None:
                    gradients = optimizer.compute_gradients(self._loss, var_list=self.qf.get_trainable_vars())
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
                    self._optimize_loss = optimizer.apply_gradients(gradients)
                else:
                    self._optimize_loss = optimizer.minimize(self._loss, var_list=self.qf.get_trainable_vars())

    @overrides
    def train(self, sess=None):
        """
        Train the network.

        A tf.Session can be provided, or will be created otherwise.
        """
        created_session = True if sess is None else False
        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            self.sess.__enter__()
        else:
            self.sess = sess

        self.sess.run(tf.global_variables_initializer())
        self.start_worker(self.sess)

        episode_rewards = []
        episode_qf_losses = []
        episode_length = []
        ts = 0

        self.sess.run(self._qf_update_ops, feed_dict=dict())

        obs = self.env.reset()
        episode_rewards.append(0.)

        for itr in range(self.n_epochs):
            with logger.prefix('Epoch #%d | ' % itr):
                # paths = self.obtain_samples(itr)
                # samples_data = self.process_samples(itr, paths)
                # episode_rewards.extend(samples_data["undiscounted_returns"])
                # self.log_diagnostics(paths)
                
                if self.es:
                    action, _ = self.es.get_action(
                        itr, obs, self.policy)
                else:
                    action, _ = self.policy.get_action(
                        obs)

                next_obs, reward, done, env_info = self.env.step(action)

                self.replay_buffer.add_transition(
                    observation=[obs],
                    action=[action],
                    reward=[reward],
                    terminal=[done],
                    next_observation=[next_obs],
                )

                episode_rewards[-1] += reward
                ts += 1
                obs = next_obs

                if done:
                    episode_length.append(ts)
                    ts = 0
                    episode_rewards.append(0.)
                    obs = self.env.reset()

                for train_itr in range(self.n_train_steps):
                    if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                        self.evaluate = True
                        qf_loss = self.optimize_policy(itr, None)
                        episode_qf_losses.append(qf_loss)
                        if itr % self.target_network_update_freq == 0:
                            self.sess.run(self._qf_update_ops, feed_dict=dict())

                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
                mean100ep_rewards = round(np.mean(episode_rewards[-101:-1]), 1)
                mean100ep_qf_loss = np.mean(episode_qf_losses[-101:-1])
                if self.evaluate and itr % 1000 == 0:
                    logger.record_tabular('Epoch', itr)
                    logger.record_tabular('AverageReturn',
                                          mean100ep_rewards)
                    logger.record_tabular('StdReturn', np.std(episode_rewards))
                    logger.record_tabular('QFunction/AverageQFunctionLoss',
                                          mean100ep_qf_loss)
                    logger.record_tabular('EpisodeLength', np.mean(episode_length))

                # if not self.smooth_return:
                #    episode_rewards = []
                #    episode_qf_losses = []

                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        if created_session:
            self.sess.close()

        return mean100ep_rewards

    @overrides
    def optimize_policy(self, itr, sample_data):
        """Optimize network using experiences from replay buffer."""
        transitions = self.replay_buffer.sample(self.buffer_batch_size)

        observations = transitions["observation"]
        rewards = transitions["reward"]
        actions = transitions["action"]
        next_observations = transitions["next_observation"]
        dones = transitions["terminal"]

        if len(self.env.spec.observation_space.shape) == 3:
            observations = observations.astype(np.float32) / 255.0
            next_observations = next_observations.astype(np.float32) / 255.0

        loss, _ = self.sess.run(
            [self._loss, self._optimize_loss],
            feed_dict={
                self.obs_input: observations,
                self.action_t_ph: actions,
                self.reward_t_ph: rewards,
                self.done_t_ph: dones,
                self.next_obs_t_ph: next_observations
            })

        return loss


def get_target_ops(variables, target_variables):
    """Get target network update operations."""
    init_ops = []
    assert len(variables) == len(target_variables)
    for var, target_var in zip(variables, target_variables):
        # hard update
        init_ops.append(tf.assign(target_var, var))

    return init_ops
