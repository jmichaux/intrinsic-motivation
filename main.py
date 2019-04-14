import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np
import gym
import torch
import torch.optim as optim

from models import ActorCritic, FwdDyn
from ppo import PPO
import multimodal_envs
from multimodal_envs.wrappers import make_vec_envs
import utils
import logger

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--env-id', type=str, default='FetchReach-v1')
parser.add_argument('--add-intrinsic-reward', action='store_true')
parser.add_argument('--share-optim', action='store_true')
parser.add_argument('--log-dir', type=str, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--clean-dir', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-processes', type=int, default=1)
parser.add_argument('--num-steps', type=int, default=2048)
parser.add_argument('--num-updates', type=int, default=int(1e4))
parser.add_argument('--ppo-epochs', type=int, default=10)
parser.add_argument('--dyn-epochs', type=int, default=5)
parser.add_argument('--num-mini-batch', type=int, default=32)
parser.add_argument('--clip-param', type=float, default=0.2)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--grad-norm-max', type=float, default=0.5)
parser.add_argument('--dyn-grad-norm-max', type=float, default=5)
parser.add_argument('--use-clipped-value-loss', action='store_true')
parser.add_argument('--use-tensorboard', action='store_true')
parser.add_argument('--dyn-lr', type=float, default=1e-3)
parser.add_argument('--pi-lr', type=float, default=1e-4)
parser.add_argument('--v-lr', type=float, default=1e-3)
parser.add_argument('--hidden-size', type=int, default=64)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--use-gae', action='store_true')
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--eval-interval', type=int, default=100)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # setup logging
    if args.log_dir is None:
        log_dir = utils.create_log_dirs(args.env_id, force_clean=args.clean_dir)
    else:
        log_dir = args.log_dir
    logger.configure(log_dir, ['stdout', 'log'], tbX=args.use_tensorboard)

    # set device and random seeds
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    utils.set_random_seeds(args.seed, args.cuda, args.debug)

    # setup environment
    envs = make_vec_envs(args.env_id, args.seed, args.num_processes,
                         args.gamma, log_dir, device, False)

    # create agent
    agent = PPO(envs.observation_space,
                envs.action_space,
                actor_critic=ActorCritic,
                dynamics_model=FwdDyn,
                optimizer=optim.Adam,
                hidden_size=args.hidden_size,
                num_steps=args.num_steps,
                num_processes=args.num_processes,
                ppo_epochs=args.ppo_epochs,
                num_mini_batch=args.num_mini_batch,
                pi_lr=args.pi_lr,
                v_lr=args.v_lr,
                dyn_lr=args.dyn_lr,
                clip_param=args.clip_param,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                grad_norm_max=args.grad_norm_max,
                use_clipped_value_loss=True,
                use_tensorboard=args.use_tensorboard,
                add_intrinsic_reward=args.add_intrinsic_reward,
                device=device,
                share_optim=args.share_optim,
                debug=args.debug)

    # reset envs and initialize rollouts
    obs = envs.reset()
    agent.rollouts.obs[0].copy_(obs[1])
    agent.rollouts.to(device)

    # start training
    start = time.time()
    episode_rewards = deque(maxlen=10)

    for update in range(args.num_updates):
        for step in range(args.num_steps):
            # select action
            value, action, action_log_probs = agent.select_action(step)

            # take a step in the environment
            obs, reward, done, infos = envs.step(action)

            # get intrinsic reward
            if args.add_intrinsic:
                intrinsic_reward = agent.compute_intrinsic_reward(step)
            else:
                intrinsic_reward = 0

            # get episode reward
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # store experience
            agent.store_rollout(obs[1], action, action_log_probs,
                                value, reward, intrinsic_reward,
                                done, infos)

        # compute returns
        agent.compute_returns(args.gamma, args.use_gae,
                              args.gae_lambda, args.add_intrinsic)

        # update policy and value_fn, reset rollout storage
        tot_loss, pi_loss, v_loss, dyn_loss, entropy, kl, delta_p, delta_v =  agent.update()

        # log data
        current = time.time()
        delta_t = current - start
        total_steps = (update + 1) * args.num_processes * args.num_steps
        fps =int(total_steps / (current - start))

        logger.logkv('Time/Updates', update)
        logger.logkv('Time/Total Steps', total_steps)
        logger.logkv('Time/FPS', fps)
        logger.logkv('Time/Current', current)
        logger.logkv('Time/Elapsed', delta_t)
        logger.logkv('Reward/Mean', np.mean(episode_rewards))
        logger.logkv('Reward/Median', np.median(episode_rewards))
        logger.logkv('Reward/Min', np.min(episode_rewards))
        logger.logkv('Reward/Max', np.max(episode_rewards))
        logger.logkv('Loss/Total', tot_loss)
        logger.logkv('Loss/Policy', pi_loss)
        logger.logkv('Loss/Value', v_loss)
        logger.logkv('Loss/Entropy', entropy)
        logger.logkv('Loss/KL', kl)
        logger.logkv('Loss/DeltaPi', delta_p)
        logger.logkv('Loss/DeltaV', delta_v)
        logger.logkv('Loss/Dynamics', dyn_loss)

        if args.use_tensorboard:
            logger.add_scalar('reward/mean', np.mean(episode_rewards), total_steps, delta_t)
            logger.add_scalar('reward/median', np.median(episode_rewards), total_steps, delta_t)
            logger.add_scalar('reward/min', np.min(episode_rewards), total_steps, delta_t)
            logger.add_scalar('reward/max', np.max(episode_rewards), total_steps, delta_t)
            logger.add_scalar('loss/total', tot_loss, total_steps, delta_t)
            logger.add_scalar('loss/policy', pi_loss, total_steps, delta_t)
            logger.add_scalar('loss/value', v_loss, total_steps, delta_t)
            logger.add_scalar('loss/entropy', entropy, total_steps, delta_t)
            logger.add_scalar('loss/kl', kl, total_steps, delta_t)
            logger.add_scalar('loss/delta_p', delta_p, total_steps, delta_t)
            logger.add_scalar('loss/delta_v', delta_v, total_steps, delta_t)
            logger.add_scalar('loss/dynamics', dyn_loss, total_steps, delta_t)

            if args.debug:
                pass

        if update % args.print_freq == 0:
            logger.dumpkvs()
