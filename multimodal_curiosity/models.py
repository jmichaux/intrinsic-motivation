import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from distributions import DiagGaussian
from utils import init

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(ActorCritic, self).__init__()
        self.base = MLP(num_inputs, hidden_size)
        self.dist = DiagGaussian(hidden_size, num_outputs)

    def forward(self):
        raise NotImplentedError

    def select_action(self, obs):
        value, actor_features = self.base(obs)
        dist = self.dist(actor_features)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        return value, action, action_log_probs

    def evaluate_action(self, obs, action):
        value, actor_features = self.base(obs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return value, action_log_probs, entropy

    def get_value(self, obs):
        value, _ = self.base(obs)
        return value

class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(MLP, self).__init__()
        # weight initialization
        init_ = lambda m: init(m,nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               5.0/3)

        # actor
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # critic
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.critic_head = init_(nn.Linear(hidden_size, 1))

    def forward(self, x):
        #TODO: need to modify if using
        # recurrent policy
        actor_features = self.actor(x)
        value = self.critic_head(self.critic(x))
        return value, actor_features
