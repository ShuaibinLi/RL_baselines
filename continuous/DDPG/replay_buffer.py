import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        if act_dim == 0:  # Discrete control environment
            self.action = np.zeros((max_size, ), dtype='int32')
        else:  # Continuous control environment
            self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size, ), dtype='float32')
        self.terminal = np.zeros((max_size, ), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')

        self.cur_size = 0
        self.cur_pos = 0

    def store(self, obs, act, reward, next_obs, terminal):
        if self.cur_size < self.max_size:
            self.cur_size += 1
        self.obs[self.cur_pos] = obs
        self.action[self.cur_pos] = act
        self.reward[self.cur_pos] = reward
        self.next_obs[self.cur_pos] = next_obs
        self.terminal[self.cur_pos] = terminal
        self.cur_pos = (self.cur_pos + 1) % self.max_size

    def sample(self, batch_size):
        batch_idx = np.random.randint(self.cur_size, size=batch_size)

        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal
