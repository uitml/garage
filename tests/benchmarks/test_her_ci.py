"""This script creates a regression test over garage-HER and baselines-HER."""
import datetime
import os.path as osp
import random
import unittest
import json
import shutil
import os

from baselines.bench import benchmarks
from baselines.her.experiment.config import CACHED_ENVS
from baselines.her.experiment.config import DEFAULT_PARAMS as BASELINES_PARAMS
from baselines.her.experiment.train import launch
import gym
import pandas as pd
import tensorflow as tf

from garage.misc import ext
from garage.misc import logger as garage_logger
from garage.replay_buffer import HerReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

# Hyperparams for baselines and garage
params = {
    "policy_lr": 1e-3,
    "qf_lr": 1e-3,
    "policy_hidden_sizes": [256, 256, 256],
    "qf_hidden_sizes": [256, 256, 256],
    "n_epochs": 30,
    "n_epoch_cycles": 20,
    "n_rollout_steps": 100,
    "n_train_steps": 40,
    "discount": 0.9,
    "tau": 0.05,
    "replay_buffer_size": int(1e6),
    "sigma": 0.2,
}

BASELINES_PARAMS["rollout_batch_size"] = 1


class TestBenchmarkHER(unittest.TestCase):
    def test_benchmark_her(self):
        """
        Compare benchmarks between garage and baselines.
        :return:
        """
        mujoco1m = benchmarks.get_benchmark("HerDdpg")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./data/local/benchmark_her/%s/" % timestamp

        latest_dir = "./latest_results"
        result_json = {}
        result_json["time_start"] = timestamp

        if osp.exists(latest_dir):
            shutil.rmtree(latest_dir)
        os.makedirs(latest_dir)

        for task in mujoco1m["tasks"]:
            env_id = task["env_id"]
            env = gym.make(env_id)
            seeds = random.sample(range(100), task["trials"])

            task_dir = osp.join(benchmark_dir, env_id)
            baselines_csvs = []
            garage_csvs = []

            for trail in range(task["trials"]):
                env.reset()
                seed = seeds[trail]

                trail_dir = task_dir + "/trail_%d_seed_%d" % (trail + 1, seed)
                garage_dir = trail_dir + "/garage"
                baselines_dir = trail_dir + "/baselines"

                garage_csv = run_garage(env, seed, garage_dir)

                CACHED_ENVS.clear()
                baselines_csv = run_baselines(env_id, seed, baselines_dir)

                env.close()

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            result_json[env_id] = create_json(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                seeds=seeds,
                trails=task["trials"],
                g_x="Epoch",
                g_y="AverageReturn",
                b_x="total/epochs",
                b_y="rollout/return",
                factor=6400)

        write_file(json.dumps(result_json), "DDPG")

    test_benchmark_her.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.
    Replace the ppo with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    ext.set_seed(seed)

    with tf.Graph().as_default():
        env = TfEnv(env)

        action_noise = OUStrategy(env.spec, sigma=params["sigma"])

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            name="Policy",
            hidden_sizes=params["policy_hidden_sizes"],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            input_include_goal=True,
        )

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            name="QFunction",
            hidden_sizes=params["qf_hidden_sizes"],
            hidden_nonlinearity=tf.nn.relu,
            input_include_goal=True,
        )

        replay_buffer = HerReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params["replay_buffer_size"],
            time_horizon=params["n_rollout_steps"],
            replay_k=0.4,
            reward_fun=env.compute_reward,
        )

        algo = DDPG(
            env,
            policy=policy,
            qf=qf,
            replay_buffer=replay_buffer,
            policy_lr=params["policy_lr"],
            qf_lr=params["qf_lr"],
            plot=False,
            target_update_tau=params["tau"],
            n_epochs=params["n_epochs"],
            n_epoch_cycles=params["n_epoch_cycles"],
            max_path_length=params["n_rollout_steps"],
            n_train_steps=params["n_train_steps"],
            discount=params["discount"],
            exploration_strategy=action_noise,
            policy_optimizer=tf.train.AdamOptimizer,
            qf_optimizer=tf.train.AdamOptimizer,
            buffer_batch_size=256,
            input_include_goal=True,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(log_dir)

        algo.train()

        garage_logger.remove_tabular_output(tabular_log_file)

        return tabular_log_file


def run_baselines(env_id, seed, log_dir):
    """
    Create baselines model and training.

    Replace the ppo and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """
    with tf.Session():
        launch_params = {
            "env": env_id,
            "logdir": log_dir,
            "n_epochs": params["n_epochs"],
            "num_cpu":
            1,  # For FetchReachEnv, the performance is not relevant to num_cpu
            "seed": seed,
            "policy_save_interval": 0,
            "replay_strategy": "future",
            "clip_return": 1,
        }
        launch(**launch_params)

    return osp.join(log_dir, "progress.csv")


def write_file(result_json, algo):
    latest_dir = "./latest_results"
    latest_result = latest_dir + "/progress.json"
    res = {}
    if osp.exists(latest_result):
        res = json.loads(open(latest_result, 'r').read())
    elif not osp.exists(latest_dir):
        os.makedirs(latest_dir)
    res[algo] = result_json
    result_file = open(latest_result, "w")
    result_file.write(json.dumps(res))


def create_json(b_csvs, g_csvs, trails, seeds, b_x, b_y, g_x, g_y, factor):
    task_result = {}
    for trail in range(trails):
        g_res, b_res = {}, {}
        trail_seed = "trail_%d" % (trail + 1)
        task_result["seed"] = seeds[trail]
        task_result[trail_seed] = {}
        df_g = json.loads(pd.read_csv(g_csvs[trail]).to_json())
        df_b = json.loads(pd.read_csv(b_csvs[trail]).to_json())

        g_res["time_steps"] = list(map(lambda x: float(x) * factor, df_g[g_x]))
        g_res["return"] = df_g[g_y]

        b_res["time_steps"] = list(map(lambda x: float(x) * factor, df_b[b_x]))
        b_res["return"] = df_b[b_y]

        task_result[trail_seed]["garage"] = g_res
        task_result[trail_seed]["baselines"] = b_res
    return task_result
