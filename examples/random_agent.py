import numpy as np

import gym
import multimodal_curiosity

import matplotlib.pyplot as plt
plt.ion()
def run_episode(env_id=None, env=None, num_episodes=1, max_steps=100000, render=False):
    if env is None:
        env = gym.make(env_id)
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.
        for step in range(1, max_steps + 1):
            if render:
                env.render()
            action = env.action_space.sample() * 0.
            action += np.array([0., -1.0, .0, 0.0])
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        print('{}: Finished episode {} in {} steps with reward {}'.format(env_id, episode, step, total_reward))
    env.close()
    return obs


if __name__ == '__main__':
    env_id = 'FetchPickAndPlaceDense-v2'
    env = gym.make(env_id)
    obs = run_episode(env_id, num_episodes=1, render=False)
    plt.imshow(obs['image'])
