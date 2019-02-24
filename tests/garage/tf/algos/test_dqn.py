"""
This script creates a test that fails when garage.tf.algos.DDPG performance is
too low.
"""
import gym

from garage.envs import normalize
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import EpsilonGreedyStrategy
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestDQN(TfGraphTestCase):
    def test_dqn_cartpole(self):
        """Test DQN with CartPole environment."""
        logger.reset()
        max_path_length = 1
        n_epochs = 20000

        env = TfEnv(normalize(gym.make("CartPole-v0")))
        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=int(5000),
            time_horizon=max_path_length)
        qf = DiscreteMLPQFunction(
            env_spec=env.spec, hidden_sizes=(64, 64), dueling=False)
        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)
        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_step=max_path_length * n_epochs,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)
        algo = DQN(
            env=env,
            policy=policy,
            qf=qf,
            exploration_strategy=epilson_greedy_strategy,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            n_epochs=n_epochs,
            qf_lr=1e-4,
            discount=1.0,
            min_buffer_size=1e3,
            n_train_steps=1,
            smooth_return=False,
            double_q=False,
            target_network_update_freq=500,
            buffer_batch_size=32)

        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 80

        env.close()
