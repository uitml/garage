#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 528.3
    RiseTime: itr 250
"""
import gym
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def run_task(*_):
    """
    Wrap PPO training task in the run_task function.

    :param _:
    :return:
    """
    env = TfEnv(normalize(gym.make("HalfCheetah-v2")))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=None,
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            hidden_sizes=(64, 64),
            use_trust_region=True,
            step_size=0.1,
        ),
    )

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2048,
        max_path_length=100,
        n_itr=488,
        discount=0.99,
        gae_lambda=0.95,
        lr_clip_range=0.1,
        policy_ent_coeff=0.0,
        optimizer_args=dict(
            batch_size=32,
            max_epochs=10,
            tf_optimizer_args=dict(
                learning_rate=3e-4,
                epsilon=1e-5,
            ),
        ),
        plot=False,
    )

    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=73,
    plot=False,
)
