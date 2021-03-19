import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(nn.Module):
    def __init__(self,
                 model,
                 gamma=None,
                 decay=None,
                 actor_lr=None,
                 critic_lr=None):
        self.gamma = gamma
        self.decay = decay
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_model.parameters(), lr=critic_lr)

    def predict(self, obs):
        return self.model.actor(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        # Compute the target Q value
        with torch.no_grad():
            target_Q = self.target_model.critic(
                next_obs, self.target_model.actor(next_obs))
            target_Q = reward + ((1. - terminal) * self.gamma * target_Q)

        # Get current Q estimate
        current_Q = self.model.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        # Compute actor loss and Update the frozen target models
        actor_loss = -self.model.critic(
            obs, self.model.actor(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self):
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_(self.decay * param.data +
                                    (1. - self.decay) * target_param.data)
