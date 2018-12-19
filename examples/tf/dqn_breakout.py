"""
This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Breakout. And uses a DQN with
1M steps.
"""
import gym

from garage.envs import normalize
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.repeat_action import RepeatAction
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.experiment import run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import GreedyPolicy
from garage.tf.q_functions import DiscreteCNNQFunction


def run_task(*_):
    """Run task."""
    max_path_length = 100
    n_epochs = 500

    env = TfEnv(
        normalize(
            StackFrames(
                RepeatAction(
                    Resize(
                        Grayscale(gym.make("BreakoutNoFrameskip-v4")),
                        width=84,
                        height=84),
                    n_frame_to_repeat=3),
                n_frames=3)))

    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=int(1e4),
        time_horizon=max_path_length,
        dtype="uint8")

    policy = GreedyPolicy(
        env_spec=env.spec,
        max_epsilon=1.0,
        min_epsilon=0.02,
        total_step=max_path_length * n_epochs,
        decay_ratio=0.1)

    qf = DiscreteCNNQFunction(
        env_spec=env.spec, filter_dims=(8, 4, 3), num_filters=(16, 32, 32))

    algo = DQN(
        env=env,
        policy=policy,
        qf=qf,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        qf_lr=0.001,
        n_epochs=n_epochs,
        discount=0.99,
        min_buffer_size=1e3,
        n_train_steps=500,
        smooth_return=False,
        target_network_update_freq=5,
        buffer_batch_size=32,
        dueling=False)

    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
