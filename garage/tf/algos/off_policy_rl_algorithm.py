"""
This module implements a class for off-policy rl algorithms.

Off-policy algorithms such as DQN, DDPG can inherit from it.
"""
from collections import deque

import numpy as np

from garage.algos import RLAlgorithm
from garage.misc.logger import logger


class OffPolicyRLAlgorithm(RLAlgorithm):
    """This class implements OffPolicyRLAlgorithm."""

    def __init__(
            self,
            env,
            policy,
            qf,
            replay_buffer,
            n_epoch_cycles,
            use_target=False,
            discount=0.99,
            max_path_length=100,
            n_train_steps=50,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            rollout_batch_size=1,
            reward_scale=1.,
            input_include_goal=False,
            smooth_return=True,
            exploration_strategy=None,
    ):
        """Construct an OffPolicyRLAlgorithm class."""
        self.env = env
        self.policy = policy
        self.qf = qf
        self.replay_buffer = replay_buffer
        self.n_epoch_cycles = n_epoch_cycles
        self.n_train_steps = n_train_steps
        self.buffer_batch_size = buffer_batch_size
        self.use_target = use_target
        self.discount = discount
        self.min_buffer_size = min_buffer_size
        self.rollout_batch_size = rollout_batch_size
        self.reward_scale = reward_scale
        self.evaluate = False
        self.input_include_goal = input_include_goal
        self.smooth_return = smooth_return
        self.max_path_length = max_path_length
        self.es = exploration_strategy

        self.success_history = deque(maxlen=100)
        self.episode_rewards = []
        self.episode_policy_losses = []
        self.episode_qf_losses = []
        self.epoch_ys = []
        self.epoch_qs = []

        self.init_opt()

    def log_diagnostics(self, paths):
        """Log diagnostic information on current paths."""
        self.policy.log_diagnostics(paths)
        self.qf.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure.

        If using tensorflow, this may
        include declaring all the variables and compiling functions.
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """Return data saved in the snapshot for this iteration."""
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        """Optimize policy network."""
        raise NotImplementedError

    def train_once(self, itr, paths):
        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend(paths["undiscounted_returns"])
        self.success_history.extend(paths["success_history"])
        last_average_return = np.mean(self.episode_rewards)
        self.log_diagnostics(paths)
        for train_itr in range(self.n_train_steps):
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                self.evaluate = True
                qf_loss, y, q, policy_loss = self.optimize_policy(epoch, paths)

                self.episode_policy_losses.append(policy_loss)
                self.episode_qf_losses.append(qf_loss)
                self.epoch_ys.append(y)
                self.epoch_qs.append(q)

        if itr % self.n_epoch_cycles == 0:
            logger.log("Training finished")
            logger.log("Saving snapshot #{}".format(int(epoch)))
            params = self.get_itr_snapshot(epoch, paths)
            logger.save_itr_params(epoch, params)
            logger.log("Saved")
            if self.evaluate:
                logger.record_tabular('Epoch', epoch)
                logger.record_tabular('AverageReturn',
                                      np.mean(self.episode_rewards))
                logger.record_tabular('StdReturn',
                                      np.std(self.episode_rewards))
                logger.record_tabular('Policy/AveragePolicyLoss',
                                      np.mean(self.episode_policy_losses))
                logger.record_tabular('QFunction/AverageQFunctionLoss',
                                      np.mean(self.episode_qf_losses))
                logger.record_tabular('QFunction/AverageQ',
                                      np.mean(self.epoch_qs))
                logger.record_tabular('QFunction/MaxQ', np.max(self.epoch_qs))
                logger.record_tabular('QFunction/AverageAbsQ',
                                      np.mean(np.abs(self.epoch_qs)))
                logger.record_tabular('QFunction/AverageY',
                                      np.mean(self.epoch_ys))
                logger.record_tabular('QFunction/MaxY', np.max(self.epoch_ys))
                logger.record_tabular('QFunction/AverageAbsY',
                                      np.mean(np.abs(self.epoch_ys)))
                if self.input_include_goal:
                    logger.record_tabular('AverageSuccessRate',
                                          np.mean(self.success_history))

            if not self.smooth_return:
                self.episode_rewards = []
                self.episode_policy_losses = []
                self.episode_qf_losses = []
                self.epoch_ys = []
                self.epoch_qs = []

            self.success_history.clear()

        return last_average_return
