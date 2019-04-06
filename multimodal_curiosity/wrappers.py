"""
Adapted from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
from collections import deque
import numpy as np
import gym
from gym.spaces.box import Box
import copy
import cv2
import torch
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
cv2.ocl.setUseOpenCL(False)


def _make_envs(env_id,
               seed,
               rank,
               log_dir=None,
               wrap_pytorch=False):
    def _thunk():
        env = gym.make(env_id)
        if wrap_pytorch:
            pass
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

def make_fetch_env(env_id, num_processes, seed, log_dir=None,
                 wrap_pytorch=False, device=None):
    envs = [_make_envs(env_id, seed, rank, log_dir, wrap_pytorch) for rank in range(num_processes)]
    if num_processes == 1:
        envs = DummyVecEnv(envs)
    else:
        # envs = SubprocVecEnv(envs)
        envs = ShmemVecEnv(envs)
    if wrap_pytorch:
        return VecEnvPyTorch(envs, device)
    return envs

class TorchTensor(gym.ObservationWrapper):
    """
    Convert observations to torch tensors
    """
    def __init__(self, env=None, device=None):
        super(TorchTensor, self).__init__(env)
        self.device = device

    def observation(self, obs):
        if self.device is None or self.device == 'cpu':
            return torch.from_numpy(obs).float().to('cpu')
        return torch.from_numpy(obs).float().to('cuda')


class VecEnvPyTorch(VecEnvWrapper):
    def __init__(self, venv, device=None):
        self.venv = venv
        if device is None or device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device=torch.device('cuda')
        observation_space = self.venv.observation_space
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def close(self):
        self.venv.close()

if __name__ == '__main__':
    env = gym.make('FetchReachDense-v2')
