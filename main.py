import argparse
from collections import deque
import os
import random
import sys
import time
import yaml

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
parser.add_argument('--experiment-name', type=str, default='RandomAgent')
parser.add_argument('--env-id', type=str, default='FetchReach-v1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-dir', type=str, default=None)
parser.add_argument('--clean-dir', action='store_true')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--checkpoint-interval', type=int, default=20)
parser.add_argument('--eval-interval', type=int, default=100)
parser.add_argument('--add-intrinsic-reward', action='store_true')
parser.add_argument('--num-env-steps', type=int, default=int(1e6))
parser.add_argument('--num-processes', type=int, default=4)
parser.add_argument('--num-steps', type=int, default=2048)
parser.add_argument('--ppo-epochs', type=int, default=10)
parser.add_argument('--dyn-epochs', type=int, default=5)
parser.add_argument('--num-mini-batch', type=int, default=32)
parser.add_argument('--pi-lr', type=float, default=7e-4)
parser.add_argument('--v-lr', type=float, default=3e-3)
parser.add_argument('--dyn-lr', type=float, default=1e-3)
parser.add_argument('--hidden-size', type=int, default=64)
parser.add_argument('--clip-param', type=float, default=0.2)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--grad-norm-max', type=float, default=0.5)
parser.add_argument('--dyn-grad-norm-max', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--use-gae', action='store_true')
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--share-optim', action='store_true')
parser.add_argument('--use-clipped-value-loss', action='store_true')
parser.add_argument('--use-tensorboard', action='store_true')
parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--debug', action='store_true')

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # setup logging
    if args.log_dir is None:
        log_dir = utils.create_log_dirs("{}/{}".format(args.env_id, args.experiment_name),
                                        force_clean=args.clean_dir)
    else:
        log_dir = args.log_dir
    logger.configure(log_dir, ['stdout', 'log', 'csv'], tbX=args.use_tensorboard)

    # save parameters
    with open(os.path.join(log_dir,'params.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f, default_flow_style=False)

    # set device and random seeds
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    utils.set_random_seeds(args.seed, args.cuda, args.debug)

    # setup environment
    envs = make_vec_envs(args.env_id, args.seed, args.num_processes,
                         args.gamma, log_dir, device, False)

    # create agent
    agent = PPO(log_dir,
                envs.observation_space,
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
    agent.train()
    start = time.time()
    episode_rewards = deque(maxlen=100)
    solved_episodes = deque(maxlen=100)

    num_updates = int(args.num_env_steps // args.num_processes // args.num_steps)

    for update in range(num_updates):
        for step in range(args.num_steps):
            # select action
            value, action, action_log_probs = agent.select_action(step)

            # take a step in the environment
            obs, reward, done, infos = envs.step(action)

            # get intrinsic reward
            if args.add_intrinsic_reward:
                intrinsic_reward = agent.compute_intrinsic_reward(step)
            else:
                intrinsic_reward = torch.tensor(0).view(1, 1)

            # get episode reward
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    solved_episodes.append(info['is_success'])

            # store experience
            agent.store_rollout(obs[1], action, action_log_probs,
                                value, reward, intrinsic_reward,
                                done)

        # compute returns
        agent.compute_returns(args.gamma, args.use_gae, args.gae_lambda)

        # update policy and value_fn, reset rollout storage
        tot_loss, pi_loss, v_loss, dyn_loss, entropy, kl, delta_p, delta_v =  agent.update()

        # log data
        if update % args.log_interval == 0:
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
            logger.logkv('Reward/Solved', np.mean(solved_episodes))
            logger.logkv('Loss/Total', tot_loss)
            logger.logkv('Loss/Policy', pi_loss)
            logger.logkv('Loss/Value', v_loss)
            logger.logkv('Loss/Entropy', entropy)
            logger.logkv('Loss/KL', kl)
            logger.logkv('Loss/DeltaPi', delta_p)
            logger.logkv('Loss/DeltaV', delta_v)
            logger.logkv('Loss/Dynamics', dyn_loss)
            logger.logkv('Value/Mean', np.mean(agent.rollouts.value_preds.cpu().data.numpy()))
            logger.logkv('Value/Median', np.median(agent.rollouts.value_preds.cpu().data.numpy()))
            logger.logkv('Value/Min', np.min(agent.rollouts.value_preds.cpu().data.numpy()))
            logger.logkv('Value/Max', np.max(agent.rollouts.value_preds.cpu().data.numpy()))

            if args.use_tensorboard:
                logger.add_scalar('reward/mean', np.mean(episode_rewards), total_steps, delta_t)
                logger.add_scalar('reward/median', np.median(episode_rewards), total_steps, delta_t)
                logger.add_scalar('reward/min', np.min(episode_rewards), total_steps, delta_t)
                logger.add_scalar('reward/max', np.max(episode_rewards), total_steps, delta_t)
                logger.add_scalar('reward/solved', np.max(solved_episodes), total_steps, delta_t)
                logger.add_scalar('loss/total', tot_loss, total_steps, delta_t)
                logger.add_scalar('loss/policy', pi_loss, total_steps, delta_t)
                logger.add_scalar('loss/value', v_loss, total_steps, delta_t)
                logger.add_scalar('loss/entropy', entropy, total_steps, delta_t)
                logger.add_scalar('loss/kl', kl, total_steps, delta_t)
                logger.add_scalar('loss/delta_p', delta_p, total_steps, delta_t)
                logger.add_scalar('loss/delta_v', delta_v, total_steps, delta_t)
                logger.add_scalar('loss/dynamics', dyn_loss, total_steps, delta_t)
                logger.add_scalar('value/mean', np.mean(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, delta_t)
                logger.add_scalar('value/median', np.median(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, delta_t)
                logger.add_scalar('value/min', np.min(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, delta_t)
                logger.add_scalar('value/max', np.max(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, delta_t)

                if args.debug:
                    logger.add_histogram('debug/actions', agent.rollouts.actions.cpu().data.numpy(), total_steps)
                    logger.add_histogram('debug/observations', agent.rollouts.obs.cpu().data.numpy(), total_steps)

                    total_grad_norm = 0
                    total_weight_norm = 0
                    for name, param in agent.actor_critic.named_parameters():
                        logger.add_histogram('debug/param/{}'.format(name), param.cpu().data.numpy(), total_steps)
                        grad_norm = param.grad.data.norm(2)
                        weight_norm = param.data.norm(2)
                        total_grad_norm += grad_norm.item() ** 2
                        total_weight_norm += weight_norm.item() ** 2

                    total_grad_norm = total_grad_norm ** (1. / 2)
                    total_weight_norm = total_weight_norm ** (1. / 2)
                    logger.add_scalar('debug/param/grad_norm', total_grad_norm, total_steps, delta_t)
                    logger.add_scalar('debug/param/weight_norm', total_weight_norm, total_steps, delta_t)

            logger.dumpkvs()

            # checkpoint model
            if (update + 1) % args.checkpoint_interval == 0:
                agent.save_checkpoint()
