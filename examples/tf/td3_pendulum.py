"""
This is an example to train a task with TD3 algorithm.

Here, we create a gym environment InvertedDoublePendulum
and use a TD3 with 1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import gym
import tensorflow as tf

from garage.experiment import LocalRunner, run_experiment
from garage.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(*_):
    """
    Wrap TD3 training task in the run_task function.

    :param _:
    :return:
    """
    with LocalRunner() as runner:
        env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[400, 300],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(
            name="ContinuousMLPQFunction",
            env_spec=env.spec,
            hidden_sizes=[400, 300],
            hidden_nonlinearity=tf.nn.relu)

        qf2 = ContinuousMLPQFunction(
            name="ContinuousMLPQFunction2",
            env_spec=env.spec,
            hidden_sizes=[400, 300],
            hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

        td3 = TD3(
            env,
            policy=policy,
            policy_lr=1e-4,
            qf_lr=1e-3,
            qf=qf,
            qf2=qf2,
            replay_buffer=replay_buffer,
            target_update_tau=1e-2,
            n_epoch_cycles=20,
            max_path_length=1000,
            n_train_steps=50,
            smooth_return=False,
            discount=0.99,
            buffer_batch_size=100,
            min_buffer_size=int(1e4),
            exploration_strategy=action_noise,
            policy_optimizer=tf.train.AdamOptimizer,
            qf_optimizer=tf.train.AdamOptimizer)

        runner.setup(td3, env)
        runner.train(n_epochs=500, n_epoch_cycles=20)


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=True,
)
