import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

from distributions import DiagGaussian
from utils import *

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(ActorCritic, self).__init__()
        self.policy = Policy(num_inputs, hidden_size, num_outputs)
        self.value_fn = ValueFn(num_inputs, hidden_size)

    def forward(self):
        raise NotImplementedError

    def select_action(self, obs):
        value, actor_features = self.value_fn(obs), self.policy(obs)
        dist = self.policy.dist(actor_features)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs

    def evaluate_action(self, obs, action):
        value, actor_features = self.value_fn(obs), self.policy(obs)

        dist = self.policy.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        entropy = dist.entropy().mean()

        return value, action_log_probs, entropy

    def get_value(self, obs):
        return self.value_fn(obs)

class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(Policy, self).__init__()

        self.base = nn.Sequential(
            init_relu(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_tanh(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.dist = DiagGaussian(hidden_size, num_outputs)

    def forward(self, x):
        return self.base(x)

class ValueFn(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(ValueFn, self).__init__()

        self.base = nn.Sequential(
            init_relu(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_relu(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.head = init_(nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.head(self.base(x))

class FwdDyn(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(FwdDyn, self).__init__()
        self.base = nn.Sequential(
            init_relu(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_relu(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_tanh(nn.Linear(hidden_size, num_outputs)), nn.Tanh())

    def forward(self, state, action):
        feature = torch.cat((state, action), -1)
        return self.base(feature)
