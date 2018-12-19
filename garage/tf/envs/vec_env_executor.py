import numpy as np

from garage.tf.misc import tensor_utils


class VecEnvExecutor:
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length
        self.lives = None

    def step(self, action_n):
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(
            map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)

        if "ale.lives" in env_infos[0]:
            if self.lives is None:
                self.lives = [env_infos[0]["ale.lives"]] * len(obs)
            else:
                for i in range(len(self.lives)):
                    if self.lives[i] > env_infos[i]["ale.lives"]:
                        rewards[i] = -1
                        self.lives[i] = env_infos[i]["ale.lives"]

        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(
            env_infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        pass
