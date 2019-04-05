import torch
import torch.nn as nn
import torch.distributions as D

class Normal(D.Normal):
    """
    Wrapper for Gaussian distribution
    """
    def mode(self):
        return self.mean

    def log_prob(self, action):
        return super().log_prob(action).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

class DiagGaussian(nn.Module):
    """
    Diagonal Gaussian with fixed standard deviation
    """
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.mu = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.mu.weight.data, gain=1.0)
        nn.init.constant_(self.mu.bias.data, 0)
        self.log_std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        mean = self.mu(x)
        return Normal(mean, self.log_std.exp())
