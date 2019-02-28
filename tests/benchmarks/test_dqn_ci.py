"""
This script creates a regression test over garage-DQN and baselines-DQN.

It get Atari10M benchmarks from baselines benchmark, and test each task in
its trail times on garage model and baselines model. For each task, there will
be `trail` times with different random seeds. For each trail, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
"""
import datetime
import os.path as osp
import os
import random
import shutil
import unittest
import json

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines import deepq
from baselines.common.misc_util import set_global_seeds
from baselines.logger import configure
from baselines.common.atari_wrappers import make_atari

import gym
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf

from garage.envs import normalize
from garage.misc import ext
from garage.misc import logger as garage_logger
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.envs import TfEnv
from garage.replay_buffer.queue_replay_buffer import QueueReplayBuffer
from garage.tf.algos import DQN
from garage.tf.exploration_strategies import EpsilonGreedyStrategy
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteCNNQFunction
from baselines.common.atari_wrappers import make_atari
from baselines import deepq


# Hyperparams for baselines and garage
params = {
    "lr": 1e-4,
    "num_timesteps": int(1e6),
    "train_freq": 1,
    "discount": 0.99,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01,
    "learning_starts": 10000,
    "target_network_update_freq": 1000,
    "dueling": False,
    "buffer_size": int(1e5),
    "batch_size": 32
}


class TestJson(unittest.TestCase):
    def test_json(self):
        """
        Compare benchmarks between garage and baselines.

        :return:
        """
        # Load Atari10M tasks, you can check other benchmarks here
        # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py
        atart_envs = benchmarks.get_benchmark("Atari10M")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./benchmark_dqn/%s/" % timestamp

        result_json = {}
        result_json["time_start"] = timestamp
        for task in atart_envs["tasks"]:
            env_id = task["env_id"]
            env = make_atari(env_id)
            env = deepq.wrap_atari_dqn(env)

            seeds = random.sample(range(100), task["trials"])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                "{}_benchmark.png".format(env_id))
            baselines_csvs = []
            garage_csvs = []

            for trail in range(task["trials"]):
                env.reset()
                seed = seeds[trail]

                trail_dir = task_dir + "/trail_%d_seed_%d" % (trail + 1, seed)
                garage_dir = trail_dir + "/garage"
                baselines_dir = trail_dir + "/baselines"

                # Run garage algorithms
                garage_csv = run_garage(env, seed, garage_dir)

                # Run baselines algorithms
                baselines_csv = run_baselines(env, seed, baselines_dir)

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            result_json[env_id] = create_json(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                seeds=seeds,
                trails=task["trials"],
                g_x="Iteration",
                g_y="AverageReturn",
                b_x="steps",
                b_y="mean 100 episode reward",
                factor=1) 

        write_file(result_json, "DQN")

    test_json.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the ddpg with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    ext.set_seed(seed)

    with tf.Graph().as_default():
        env = TfEnv(normalize(env))

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['buffer_size'],
            time_horizon=1)

        qf = DiscreteCNNQFunction(
            env_spec=env.spec, filter_dims=(8, 4, 3), num_filters=(32, 64, 64), strides=(4, 2, 1), dueling=params['dueling'])

        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_step=params['num_timesteps'],
            max_epsilon=1.0,
            min_epsilon=params['exploration_final_eps'],
            decay_ratio=params['exploration_fraction'])

        algo = DQN(
            env=env,
            policy=policy,
            qf=qf,
            exploration_strategy=epilson_greedy_strategy,
            replay_buffer=replay_buffer,
            max_path_length=1,
            n_epochs=params['num_timesteps'],
            qf_lr=params['lr'],
            discount=params['discount'],
            grad_norm_clipping=10,
            double_q=True,  # baseline use True internally
            min_buffer_size=params['learning_starts'],
            n_train_steps=params['train_freq'],
            smooth_return=False,
            target_network_update_freq=params['target_network_update_freq'],
            buffer_batch_size=params['batch_size'],
            use_atari_wrappers=True)


        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        tensorboard_log_dir = osp.join(log_dir, "progress")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(tensorboard_log_dir)

        algo.train()

        garage_logger.remove_tabular_output(tabular_log_file)
        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """
    Create baselines model and training.

    Replace the ddpg and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """
    with tf.Session():
        rank = MPI.COMM_WORLD.Get_rank()
        seed = seed + 1000000 * rank
        set_global_seeds(seed)
        env.seed(seed)

        # Set up logger for baselines
        configure(dir=log_dir)
        baselines_logger.info('rank {}: seed={}, logdir={}'.format(
            rank, seed, baselines_logger.get_dir()))

        deepq.learn(
            env,
            "conv_only",
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=params['dueling'],
            lr=params['lr'],
            total_timesteps=params['num_timesteps'],
            buffer_size=params['buffer_size'],
            exploration_fraction=params['exploration_fraction'],
            exploration_final_eps=params['exploration_final_eps'],
            train_freq=params['train_freq'],
            learning_starts=params['learning_starts'],
            target_network_update_freq=params['target_network_update_freq'],
            gamma=params['discount'],
            batch_size=params['batch_size'],
            print_freq=1
        )

        return osp.join(log_dir, "progress.csv")


def write_file(result_json, algo):
    #if results file does not exist, create it.
    #else: load file and append to it.
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
        #convert epochs vs AverageReturn into time_steps vs AverageReturn
        g_res, b_res = {}, {}
        trail_seed = "trail_%d" % (trail + 1)
        task_result["seed"] = seeds[trail]
        task_result[trail_seed] = {}
        df_g = json.loads(pd.read_csv(g_csvs[trail]).to_json())
        df_b = json.loads(pd.read_csv(b_csvs[trail]).to_json())

        g_res["time_steps"] = list(map( lambda x: float(x)*factor , df_g[g_x] ))
        g_res["return"] =  df_g[g_y] 

        b_res["time_steps"] = list(map( lambda x: float(x)*factor , df_b[b_x] ))
        b_res["return"] =  df_b[b_y]

        task_result[trail_seed]["garage"] = g_res
        task_result[trail_seed]["baselines"] = b_res
    return task_result
