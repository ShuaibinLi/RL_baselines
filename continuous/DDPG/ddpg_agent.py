import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent(object):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        self.alg = algorithm
        self.act_dim = act_dim
        self.expl_noise = expl_noise

    def sample(self, obs):
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action = self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.FloatTensor(obs).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        terminal = torch.FloatTensor(terminal).to(device)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss
