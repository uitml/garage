"""
This module implements a replay buffer memory.

Replay buffer is an important technique in reinforcement learning. It
stores transitions in a memory buffer of fixed size. When the buffer is
full, oldest memory will be discarded. At each step, a batch of memories
will be sampled from the buffer to update the agent's parameters. In a
word, replay buffer breaks temporal correlations and thus benefits RL
algorithms.
"""
import numpy as np


class QueueReplayBuffer:
    """Abstract class for Replay Buffer."""

    def __init__(self,
                 capacity):
        """
        Initialize the data used in ReplayBuffer.

        :param buffer_shapes: shape of values for each key in the buffer
        :param size_in_transitions: total size of transitions in the buffer
        :param time_horizon: time horizon of rollout
        """
        self._size = 0
        self.currnet_ptr = 0
        self.capacity = capacity
        self._obs_buffer = np.empty([capacity], dtype=object)
        self._next_obs_buffer = np.empty([capacity], dtype=object)
        self._action_buffer = np.empty([capacity], dtype=np.int32)
        self._reward_buffer = np.empty([capacity])
        self._done_buffer = np.empty([capacity])

    def sample(self, batch_size):
        assert self.n_transitions_stored >= batch_size

        indices = np.random.choice(self.n_transitions_stored, batch_size, replace=False)

        result = {}
        result['observation'] = np.stack(self._obs_buffer[indices])
        result['reward'] = self._reward_buffer[indices]
        result['action'] = self._action_buffer[indices]
        result['next_observation'] = np.stack(self._next_obs_buffer[indices])
        result['terminal'] = self._done_buffer[indices]

        return result

    def add_transition(self, **kwargs):
        """Add one transition into the replay buffer."""
        self._obs_buffer[self.currnet_ptr] = kwargs['observation']
        self._reward_buffer[self.currnet_ptr] = kwargs['reward']
        self._action_buffer[self.currnet_ptr] = kwargs['action']
        self._next_obs_buffer[self.currnet_ptr] = kwargs['next_observation']
        self._done_buffer[self.currnet_ptr] = kwargs['terminal']

        if self._size < self.capacity:
            self._size += 1

        self.currnet_ptr += 1
        if self.currnet_ptr == self.capacity:
            self.currnet_ptr = 0

    @property
    def n_transitions_stored(self):
        """
        Return the size of the replay buffer.

        Returns:
            self._size: Size of the current replay buffer.

        """
        return self._size
