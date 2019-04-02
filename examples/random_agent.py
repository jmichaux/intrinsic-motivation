import argparse
import numpy as np

import gym
import multimodal_curiosity

import matplotlib.pyplot as plt
plt.ion()

parser = argparse.ArgumentParser(description='Random Agent')
parser.add_argument('--env-id', default='FetchPushDense-v2')
parser.add_argument('--render', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save', action='store_true')

def run_episode(env_id=None, env=None, num_episodes=1, max_steps=100000, render=False, verbose=False):
    all_obs = []
    if env is None:
        env = gym.make(env_id)
    for episode in range(num_episodes):
        ep_obs = []
        obs = env.reset()
        total_reward = 0.
        for step in range(1, max_steps + 1):
            if render:
                env.render()
            # action = env.action_space.sample() * 0.
            action = np.array([0.1, 0.1, .1, 0.0])
            obs, reward, done, info = env.step(action)
            total_reward += reward
            ep_obs.append(obs)
            print('Desired goal: {}'.format(obs['desired_goal']))

            if done:
                all_obs.append(ep_obs)
                break
        if verbose:
            print('{}: Finished episode {} in {} steps with reward {}'.format(env_id, episode, step, total_reward))
    env.close()
    return all_obs


if __name__ == '__main__':
    args = parser.parse_args()
    env_id = args.env_id
    verbose = args.verbose
    render = args.render
    save = args.save
    env = gym.make(env_id)
    obs = run_episode(env_id, num_episodes=1, render=render)
    if verbose:
        print('Desired goal: {}'.format(obs['desired_goal']))
    plt.imshow(obs[0][-1]['image'])
    if save:
        plt.savefig('{}_render_{}'.format(env_id, render))
