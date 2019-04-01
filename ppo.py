import numpy as np

import torch

from utils import *

class PPO():
    def __init__(self,
                 logdir,
                 env_id,
                 env_fn,
                 num_envs,
                 optimizer,
                 #policy,
                 batch_size,
                 rollout_steps,
                 max_grad_norm=None,
                 ent_coef,
                 val_coef,
                 clip_param,
                 gamma,
                 tau,
                 use_gae=True,
                 norm_obs=True,
                 norm_adv=True,
                 use_gpu=True):

        if use_gpu:
            if torch.cuda.is_available():
                device = 'cuda':
            else:
                print('Cuda not available, using CPU instead')
                device = 'cpu'
        else:
            device = 'cpu'
        self.device=device

        self.logdir = logdir

        # environment
        self.env_id = env_id
        self.num_envs = num_envs
        self.env = env_fn(env_id, num_envs)

        # hyperparams
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.ent_coef = ent_coef
        self.val_coef = val_coef
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae
        self.norm_obs = norm_obs
        self.norm_adv = norm_adv
        self.use_gpu = use_gpu

        # rollout storage
        self.rollouts = RolloutStorage(rollout_steps,
                                       num_envs,
                                       self.env.observation_space.shape,
                                       self.env.action_space,
                                       self.device)

        # network and optimizer
        self.optimizer = optimizer(self.model)

    def act(self):
        pass

    def update(self):
        pass

    def compute_loss(self):
        pass

    def train(self):
        for frame in range(1, max_frames + 1):
            self.act()

            self.update()

    def select_action(self):
        pass

    def log(self):
        pass
