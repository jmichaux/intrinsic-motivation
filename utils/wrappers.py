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
cv2.ocl.setUseOpenCL(False)


# TODO: IT MIGHT MAKE SENSE TO MAKE THE WRAPPERS GIN CONFIGURABLE AS WELL
def make_env(env_id,
             stack_frames=True,
             episodic_life=True,
             clip_rewards=False,
             scale=False,
             transpose=None,
             torch_tensor=False,
             device=None):
    env = gym.make(env_id)
    if episodic_life:
        env = EpisodicLifeEnv(env)

    # To do: make deep mind wrapper
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)
    if stack_frames:
        env = FrameStack(env, 4)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if transpose is not None:
        env = Transpose(env, transpose)
    if torch_tensor:
        env = TorchTensor(env, device)
    return env

def _make_envs(env_id,
               seed,
               rank,
               log_dir=None,
               stack_frames=True,
               episodic_life=True,
               clip_rewards=False,
               scale=False,
               wrap_pytorch=True):
    def _thunk():
        if wrap_pytorch:
            env = make_env(env_id, stack_frames, episodic_life, clip_rewards, scale, transpose=(2, 0, 1))
        else:
            env = make_env(env_id, stack_frames, episodic_life, clip_rewards, scale)

        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

def make_a2c_env(env_id, num_agents, seed, log_dir=None,
                  stack_frames=True, episodic_life=True,
                  clip_rewards=False, scale=False, wrap_pytorch=True, device=None):
    envs = [_make_envs(env_id, seed, rank, log_dir, stack_frames, episodic_life, clip_rewards, scale, wrap_pytorch) for rank in range(num_agents)]
    if num_agents == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)
    if wrap_pytorch:
        return VecEnvPyTorch(envs, device)
    return envs


class Transpose(gym.ObservationWrapper):
    """
    Transpose image channels
    """
    def __init__(self, env=None, transpose=(2,0,1)):
        super(Transpose, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]], # (2,0,1) or (2,1,0)?
            dtype=self.observation_space.dtype)
        self.transpose = transpose

    def observation(self, obs):
        obs = np.array(obs)
        obs = obs.transpose(self.transpose)
        return obs

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
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def close(self):
        self.venv.close()


class RewardScaler(gym.RewardWrapper):
    def reward(self, reward):
        return reward * 0.1


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs
