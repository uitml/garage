"""
This script creates a test that fails when garage.tf.algos.TD3 performance is
too low.
"""
import gym
import tensorflow as tf

from garage.experiment import LocalRunner
from garage.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestTD3(TfGraphTestCase):
    def test_td3_pendulum(self):
        """Test TD3 with Pendulum environment."""
        with LocalRunner() as runner:
            env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

            action_noise = OUStrategy(env.spec, sigma=0.2)

            policy = ContinuousMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=[64, 64],
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh)

            qf = ContinuousMLPQFunction(
                name="ContinuousMLPQFunction",
                env_spec=env.spec,
                hidden_sizes=[64, 64],
                hidden_nonlinearity=tf.nn.relu)

            qf2 = ContinuousMLPQFunction(
                name="ContinuousMLPQFunction2",
                env_spec=env.spec,
                hidden_sizes=[64, 64],
                hidden_nonlinearity=tf.nn.relu)

            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=int(1e6),
                time_horizon=100)

            algo = TD3(
                env,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                qf2=qf2,
                replay_buffer=replay_buffer,
                target_update_tau=1e-2,
                n_epoch_cycles=20,
                max_path_length=100,
                n_train_steps=50,
                discount=0.99,
                smooth_return=False,
                min_buffer_size=int(1e4),
                buffer_batch_size=100,
                exploration_strategy=action_noise,
                policy_optimizer=tf.train.AdamOptimizer,
                qf_optimizer=tf.train.AdamOptimizer)

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, n_epoch_cycles=20)
            assert last_avg_ret > 60
