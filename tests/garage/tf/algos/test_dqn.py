"""
This script creates a test that fails when garage.tf.algos.DDPG performance is
too low.
"""
import gym

from garage.envs import normalize
import garage.misc.logger as logger
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import GreedyPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestDQN(TfGraphTestCase):
    def test_dqn_cartpole(self):
        """Test DQN with CartPole environment."""
        logger.reset()

        max_path_length = 200
        n_epochs = 200

        env = TfEnv(normalize(gym.make("CartPole-v0")))

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=int(50000),
            time_horizon=max_path_length)

        policy = GreedyPolicy(
            env_spec=env.spec,
            max_epsilon=1.0,
            min_epsilon=0.02,
            total_step=max_path_length * n_epochs,
            decay_ratio=0.1)

        qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(64))

        algo = DQN(
            env=env,
            policy=policy,
            qf=qf,
            qf_lr=1e-3,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            n_epochs=n_epochs,
            discount=1.0,
            min_buffer_size=1e3,
            n_train_steps=100,
            smooth_return=False,
            target_network_update_freq=2,
            buffer_batch_size=32,
            dueling=False)

        last_avg_ret = algo.train(sess=self.sess)

        assert last_avg_ret > 190  # max = 200
