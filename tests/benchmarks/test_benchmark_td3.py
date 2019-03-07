"""
This script creates a regression test over garage-TD3.

It get Mujoco1M benchmarks from baselines benchmark, and test each task in
its trail times on garage model and baselines model. For each task, there will
be `trail` times with different random seeds. For each trail, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
"""
import datetime
import os.path as osp
import random
import unittest

from baselines.bench import benchmarks
import gym
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from garage.experiment import LocalRunner
from garage.exploration_strategies import OUStrategy
from garage.misc import ext
from garage.misc import logger as garage_logger
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

# Hyperparams for baselines and garage
params = {
    "policy_lr": 1e-3,
    "qf_lr": 1e-3,
    "policy_hidden_sizes": [400, 300],
    "qf_hidden_sizes": [400, 300],
    "n_epochs": 200,
    "n_epoch_cycles": 20,
    "n_rollout_steps": 250,
    "n_train_steps": 50,
    "discount": 0.99,
    "tau": 0.005,
    "replay_buffer_size": int(1e6),
    "sigma": 0.1,
    "smooth_return": False,
    "buffer_batch_size": 100,
}


class TestBenchmarkTD3(unittest.TestCase):
    def test_benchmark_td3(self):
        """
        Test garage TD3 benchmarks.

        :return:
        """
        # Load Mujoco1M tasks, you can check other benchmarks here
        # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py # noqa: E501
        mujoco1m = benchmarks.get_benchmark("Mujoco1M")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./benchmark_td3/%s/" % timestamp

        for task in mujoco1m["tasks"]:
            env_id = task["env_id"]
            env = gym.make(env_id)
            seeds = random.sample(range(100), task["trials"])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                "{}_benchmark.png".format(env_id))
            garage_csvs = []

            for trail in range(task["trials"]):
                env.reset()
                seed = seeds[trail]

                trail_dir = task_dir + "/trail_%d_seed_%d" % (trail + 1, seed)
                garage_dir = trail_dir + "/garage"

                # Run garage algorithms
                garage_csv = run_garage(env, seed, garage_dir)
                garage_csvs.append(garage_csv)

            plot(
                g_csvs=garage_csvs,
                g_x="Epoch",
                g_y="AverageReturn",
                trails=task["trials"],
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id)

    test_benchmark_td3.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the td3 with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    ext.set_seed(seed)

    with tf.Graph().as_default():
        with LocalRunner() as runner:
            env = TfEnv(env)
            # Set up params for TD3
            action_noise = OUStrategy(env.spec, sigma=params["sigma"])

            policy = ContinuousMLPPolicy(
                env_spec=env.spec,
                name="Policy",
                hidden_sizes=params["policy_hidden_sizes"],
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh)

            qf = ContinuousMLPQFunction(
                name="ContinuousMLPQFunction",
                env_spec=env.spec,
                hidden_sizes=params["qf_hidden_sizes"],
                hidden_nonlinearity=tf.nn.relu)

            qf2 = ContinuousMLPQFunction(
                name="ContinuousMLPQFunction2",
                env_spec=env.spec,
                hidden_sizes=params["qf_hidden_sizes"],
                hidden_nonlinearity=tf.nn.relu)

            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=params["replay_buffer_size"],
                time_horizon=params["n_rollout_steps"])

            td3 = TD3(
                env,
                policy=policy,
                qf=qf,
                qf2=qf2,
                replay_buffer=replay_buffer,
                policy_lr=params["policy_lr"],
                qf_lr=params["qf_lr"],
                target_update_tau=params["tau"],
                n_epoch_cycles=params["n_epoch_cycles"],
                max_path_length=params["n_rollout_steps"],
                n_train_steps=params["n_train_steps"],
                discount=params["discount"],
                smooth_return=params["smooth_return"],
                min_buffer_size=int(1e4),
                buffer_batch_size=params["buffer_batch_size"],
                exploration_strategy=action_noise,
                policy_optimizer=tf.train.AdamOptimizer,
                qf_optimizer=tf.train.AdamOptimizer)

            # Set up logger since we are not using run_experiment
            tabular_log_file = osp.join(log_dir, "progress.csv")
            tensorboard_log_dir = osp.join(log_dir)
            garage_logger.add_tabular_output(tabular_log_file)
            garage_logger.set_tensorboard_dir(tensorboard_log_dir)

            runner.setup(td3, env)
            runner.train(
                n_epochs=params['n_epochs'],
                n_epoch_cycles=params['n_epoch_cycles'])

            garage_logger.remove_tabular_output(tabular_log_file)

            return tabular_log_file


def plot(g_csvs, g_x, g_y, trails, seeds, plt_file, env_id):
    """
    Plot benchmark from csv files of garage.

    :param g_csvs: A list contains all csv files in the task.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param trails: Number of trails in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    for trail in range(trails):
        seed = seeds[trail]

        df_g = pd.read_csv(g_csvs[trail])

        plt.plot(
            df_g[g_x],
            df_g[g_y],
            label="garage_trail%d_seed%d" % (trail + 1, seed))

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("AverageReturn")
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()
