"""Repeat action wrapper for gym.Env."""
import gym
import numpy as np


class ClippedReward(gym.Wrapper):
    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        return obs, np.sign(reward), done, info