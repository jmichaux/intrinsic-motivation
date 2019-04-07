import argparse
import random
import time

import numpy as np
import gym
gym.logger.set_level(40)

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import ActorCritic
from rollouts import Rollouts
from distributions import DiagGaussian
import multimodal_curiosity
from multimodal_curiosity.wrappers import make_fetch_env
from utils import *
import logger

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--env_id', type=str, default="FetchPushDense-v1",
                    help='Name of atari environment')
parser.add_argument('--num_processes', type=int, default=8,
                    help='Number of agents')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--num_steps', type=int, default=50,
                    help='Number of rollout steps')
parser.add_argument('--num_updates', type=int, default=20e6,
                    help='Number of updates')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device for training (default: cuda)')
parser.add_argument('--use_gae', action='store_true',
                    help='Calculate Policy Gradient using Generalized Advantage Estimates')
parser.add_argument('--tau', type=float, default=0.95, help='GAE parameter')
parser.add_argument('--eval', action='store_true',
                    help='Evaluate Model')
parser.add_argument('--eval-freq', type=int, default=100,
                    help="Evaluate model every 'eval_freq' steps")
parser.add_argument('--lr', '--learning-rate', default=7e-4, type=float,
                    metavar='LR', help='learning rate')
parser.add_argument('--debug', action='store_true')

def compute_loss(sample, value_coeff, entropy_coeff, eps=0.01):
    obs_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_target = sample

    # evaluate actions
    values, action_log_probs, entropy = actor_critic.evaluate_action(obs_batch, actions_batch)

    # compute policy loss
    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
    sur1 = ratio * adv_target
    sur2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_target
    policy_loss = -torch.min(sur1, sur2).mean()

    # compute value loss
    value_loss = 0.5 * (return_batch - values).pow(2).mean()

    loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    return loss, policy_loss, value_loss, entropy


def update(optimizer, rollouts, n_epochs, num_mini_batch, grad_norm_max=0.5):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    value_loss_epoch = 0
    policy_loss_epoch = 0
    entropy_epoch = 0

    for e in range(n_epochs):
        data_generator = rollouts.feed_forward_generator(advantages, num_mini_batch)

        for sample in data_generator:
            loss, policy_loss, value_loss, entropy = compute_loss(
                sample, value_coeff=0.5, entropy_coeff=0.01, eps=0.01)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), grad_norm_max)
            optimizer.step()

            value_loss_epoch += value_loss.item()
            policy_loss_epoch += policy_loss.item()
            entropy_epoch += entropy.item()

    value_loss_epoch /= (n_epochs * num_mini_batch)
    policy_loss_epoch /= (n_epochs * num_mini_batch)
    entropy_epoch /= (n_epochs * num_mini_batch)
    total_loss = value_loss_epoch + policy_loss_epoch + entropy_epoch

    return policy_loss_epoch, value_loss_epoch, entropy_epoch


def train(envs, optimizer, max_frames, num_steps, num_processes, gamma=0.99, seed=1, evaluate=False, use_gae=False):

    # initialize rewards
    episode_rewards = np.zeros(num_processes, dtype=np.float)
    final_rewards = np.zeros(num_processes, dtype=np.float)

    # add initial observation to rollouts
    obs = envs.reset()
    rollouts.obs[0].copy_(obs['observation'])

    start = time.time()

    print("Starting training...")

    for frame_idx in range(1, max_frames + 1):
        for step in range(num_steps):
            # choose action
            with torch.no_grad():
                value, action, action_log_probs = actor_critic.select_action(rollouts.obs[step].to(device))
            cpu_actions = action.cpu().numpy()

            # take environment step
            obs, reward, done, info = envs.step(cpu_actions)

            # calculate mask
            masks = 1.0 - done.astype(np.float32)

            # compute rewards
            episode_rewards += reward
            final_rewards *= masks
            final_rewards += (1. - masks) * episode_rewards
            episode_rewards *= masks

            # convert to torch tensors and store
            rewards = torch.from_numpy(reward).float().view(-1, 1).to(device)
            masks = torch.from_numpy(masks).float().view(-1, 1).to(device)
            action = action.to(device)
            action_log_probs = action_log_probs.view(-1, 1).to(device)
            obs = obs.to(device)

            # save rollouts
            rollouts.insert(obs,
                            action,
                            action_log_probs,
                            value,
                            rewards,
                            masks)

        # Get next value and compute returns
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1])
        rollouts.compute_returns(next_value, gamma)

        # update model
        policy_loss, value_loss, entropy = update(
             optimizer, rollouts, n_epochs=3, num_mini_batch=32, grad_norm_max=0.5)

        # update rollout storage
        rollouts.after_update()

        if frame_idx % 100 == 0:
            print("Updates: {}".format(frame_idx))
            print("Mean/Median Reward: {}/{}".format(np.mean(final_rewards), np.median(final_rewards)))
            print("Min/Max Reward: {}/{}".format(np.min(final_rewards), np.max(final_rewards)))
            print("Entropy: {}\t Value Loss: {}\t Policy_loss: {}".format(
                entropy, value_loss, policy_loss))
            print("=================================================")
    envs.close()
    print("Training complete in {}".format(time.time() - start))
    return

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    env_id = args.env_id
    seed = args.seed
    num_processes = args.num_processes
    num_steps = args.num_steps
    device = args.device
    lr = args.lr
    if args.use_gae:
        use_gae = args.use_gae
        tau = args.tau

    evaluate = args.eval
    evaluate_freq = args.eval_freq
    debug = args.debug
    # setup
    use_cuda = torch.cuda.is_available()
    set_random_seed(seed, use_cuda)
    device = "cpu"
    # torch.set_num_threads(1)

    # create training environments
    envs = make_fetch_env(env_id, num_processes, seed,
                          wrap_pytorch=True, device=device)
    obs = envs.reset()

    # create rollout storage
    rollouts = Rollouts(num_steps,
                        num_processes,
                        envs.observation_space.spaces['observation'].shape,
                        envs.action_space,
                        device)
    rollouts.to(device)

    # create model
    actor_critic = ActorCritic(
        num_inputs=envs.observation_space.spaces['observation'].shape[0],
        hidden_size=64,
        num_outputs=envs.action_space.shape[0])
    actor_critic.to(device)

    # setup optimizer
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    if debug:
        checkpoint = torch.load('ppo_debug')
        actor_critic.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])

    # train
    max_frames = int(20e6)
    train(envs, optimizer, max_frames, num_steps, num_processes)
