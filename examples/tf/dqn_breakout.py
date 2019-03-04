"""
This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Breakout. And uses a DQN with
1M steps.
"""
import gym

from garage.envs import normalize
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.clipped_reward import ClippedReward
from garage.experiment import run_experiment
from garage.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.replay_buffer.queue_replay_buffer import QueueReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteCNNQFunction
import tensorflow as tf

# def run_task(*_):
"""Run task."""
with tf.Session() as sess:
    max_path_length = 1
    n_epochs = 1000000

    env = gym.make("PongNoFrameskip-v4")
    env = EpisodicLife(env)
    env = Noop(env, noop_max=10)
    env = MaxAndSkip(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    env = ClippedReward(env)
    env = StackFrames(env, 4)

    env = TfEnv(normalize(env))

    # replay_buffer = QueueReplayBuffer(capacity=int(2e5))
        
    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=int(5e4),
        time_horizon=max_path_length)

    qf = DiscreteCNNQFunction(
        env_spec=env.spec, filter_dims=(8, 4, 3), num_filters=(32, 64, 64), strides=(4, 2, 1),
        hidden_sizes=(256,))

    policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

    epilson_greedy_strategy = EpsilonGreedyStrategy(
        env_spec=env.spec,
        total_timesteps=n_epochs,
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
        qf_lr=1e-3,
        discount=0.99,
        grad_norm_clipping=10,
        double_q=False,
        min_buffer_size=1e4,
        n_train_steps=1,
        smooth_return=False,
        target_network_update_freq=2000,
        buffer_batch_size=32)

    algo.train(sess)


# run_experiment(
#     run_task,
#     n_parallel=1,
#     snapshot_mode="last",
#     seed=1,
#     plot=False,
# )
