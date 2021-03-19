import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class DDPGModel(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super(DDPGModel, self).__init__()
        self.actor_model = Actor(obs_dim, act_dim)
        self.critic_model = Critic(obs_dim, act_dim)

        self.max_action = max_action

    def actor(self, obs):
        act = self.actor_model(obs)
        act = act * self.max_action
        return act

    def critic(self, obs, act):
        q = self.critic_model(obs, act)
        return q


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, act_dim)

    def forward(self, obs):
        act = F.relu(self.l1(obs))
        act = F.relu(self.l2(act))
        act = torch.tanh(self.l3(act))
        return act


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400 + act_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, obs, act):
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(torch.cat([q, act], 1)))
        q = self.l3(q)
        return q
