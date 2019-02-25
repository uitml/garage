"""
This script creates a regression test over garage-TRPO and baselines-TRPO.
Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So you need to change the baselines source code to make it
stops at length 100. You also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import os
import random
import unittest
import json

from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common.tf_util import _PLACEHOLDER_CACHE
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import gym
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from garage.envs import normalize
from garage.misc import ext
from garage.misc import logger as garage_logger
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.helpers import AutoStopEnv


class TestBenchmarkPPO(unittest.TestCase):
    def test_benchmark_trpo(self):
        """
        Compare benchmarks between garage and baselines.
        :return:
        """

        mujoco1m = benchmarks.get_benchmark("Mujoco1M")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        benchmark_dir = "./data/local/benchmark_trpo/%s/" % timestamp
        result_json = {}
        result_json["time_start"] = timestamp
        for task in mujoco1m["tasks"]:
            env_id = task["env_id"]
            env = gym.make(env_id)
            baseline_env = AutoStopEnv(env_name=env_id)
            seeds = random.sample(range(100), task["trials"])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                "{}_benchmark.png".format(env_id))
            baselines_csvs = []
            garage_csvs = []

            for trail in range(task["trials"]):
                _PLACEHOLDER_CACHE.clear()
                env.reset()
                seed = seeds[trail]

                trail_dir = task_dir + "/trail_%d_seed_%d" % (trail + 1, seed)
                garage_dir = trail_dir + "/garage"
                baselines_dir = trail_dir + "/baselines"

                baseline_env.reset()
                baselines_csv = run_baselines(baseline_env, seed,
                                              baselines_dir)

                # Run garage algorithms
                env.reset()
                garage_csv = run_garage(env, seed, garage_dir)

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            result_json[env_id] = create_json(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                seeds=seeds,
                trails=task["trials"],
                g_x="Iteration",
                g_y="AverageReturn",
                b_x="Iter",
                b_y="EpRewMean",
                factor=2048)
            
        write_file(result_json, "TRPO")
    test_benchmark_trpo.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.
    Replace the trpo with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:import baselines.common.tf_util as U
    """
    ext.set_seed(seed)

    with tf.Graph().as_default():
        env = TfEnv(normalize(env))

        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(32, 32),
                use_trust_region=True,
            ),
        )

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1024,
            max_path_length=100,
            n_itr=976,
            discount=0.99,
            gae_lambda=0.98,
            clip_range=0.1,
            policy_ent_coeff=0.0,
            plot=False,
        )

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, "progress.csv")
        garage_logger.add_tabular_output(tabular_log_file)
        garage_logger.set_tensorboard_dir(log_dir)

        algo.train()

        garage_logger.remove_tabular_output(tabular_log_file)
        garage_logger.reset()

        return tabular_log_file


def run_baselines(env, seed, log_dir):
    """
    Create baselines model and training.
    Replace the trpo and its training with the algorithm you want to run.
    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """

    with tf.Session().as_default():
        baselines_logger.configure(log_dir)

        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=32,
                num_hid_layers=2)

        # env = make_mujoco_env(env_id, seed)
        trpo_mpi.learn(
            env,
            policy_fn,
            timesteps_per_batch=1024,
            max_kl=0.01,
            cg_iters=10,
            cg_damping=0.1,
            max_timesteps=int(1e6),
            gamma=0.99,
            lam=0.98,
            vf_iters=5,
            vf_stepsize=1e-3)
        env.close()

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
