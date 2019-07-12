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
import gym_fetch
from gym_fetch.wrappers import make_vec_envs
import utils
import logger

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--experiment-name', type=str, default='RandomAgent')
parser.add_argument('--env-id', type=str, default='FetchPush-v1')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-dir', type=str, default=None)
parser.add_argument('--clean-dir', action='store_true')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--checkpoint-interval', type=int, default=20)
parser.add_argument('--eval-interval', type=int, default=100)
parser.add_argument('--add-intrinsic-reward', action='store_true')
parser.add_argument('--intrinsic-coef', type=float, default=1.0)
parser.add_argument('--max-intrinsic-reward', type=float, default=None)
parser.add_argument('--num-env-steps', type=int, default=int(1e7))
parser.add_argument('--num-processes', type=int, default=4)
parser.add_argument('--num-steps', type=int, default=2048)
parser.add_argument('--ppo-epochs', type=int, default=10)
parser.add_argument('--dyn-epochs', type=int, default=5)
parser.add_argument('--num-mini-batch', type=int, default=32)
parser.add_argument('--pi-lr', type=float, default=1e-4)
parser.add_argument('--v-lr', type=float, default=1e-3)
parser.add_argument('--dyn-lr', type=float, default=1e-3)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--clip-param', type=float, default=0.3)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--grad-norm-max', type=float, default=5.0)
parser.add_argument('--dyn-grad-norm-max', type=float, default=5)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--use-gae', action='store_true')
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--share-optim', action='store_true')
parser.add_argument('--predict-delta-obs', action='store_true')
parser.add_argument('--use-linear-lr-decay', action='store_true')
parser.add_argument('--use-clipped-value-loss', action='store_true')
parser.add_argument('--use-tensorboard', action='store_true')
parser.add_argument('--cuda', action='store_false', default=True, help='enables CUDA training')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--render', action='store_true')

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # setup logging
    if args.log_dir is None:
        log_dir = utils.create_log_dirs("{}/{}".format(args.env_id, args.experiment_name),
                                        force_clean=args.clean_dir)
        args.__dict__['log_dir'] = log_dir
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
    envs = make_vec_envs(env_id=args.env_id,
                         seed=args.seed,
                         num_processes=args.num_processes,
                         gamma=None,
                         log_dir=log_dir,
                         device=device,
                         obs_keys=['observation', 'desired_goal'],
                         allow_early_resets=False)

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
                predict_delta_obs=args.predict_delta_obs,
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

    num_updates = int(args.num_env_steps // args.num_processes // args.num_steps)

    for update in range(num_updates):
        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            if args.share_optim:
                utils.update_linear_schedule(optimizer=agent.optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.pi_lr)
            else:
                utils.update_linear_schedule(optimizer=agent.policy_optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.pi_lr)

                utils.update_linear_schedule(optimizer=agent.value_fn_optimizer,
                                             update=update,
                                             total_num_updates=num_updates,
                                             initial_lr=args.v_lr)

        extrinsic_rewards = []
        episode_length = []
        intrinsic_rewards = []
        solved_episodes = []

        for step in range(args.num_steps):
            # render
            if args.render:
                envs.render()

            # select action
            value, action, action_log_probs = agent.select_action(step)

            # take a step in the environment
            obs, reward, done, infos = envs.step(action)

            # calculate intrinsic reward
            if args.add_intrinsic_reward:
                intrinsic_reward = args.intrinsic_coef * agent.compute_intrinsic_reward(step)
                if args.max_intrinsic_reward is not None:
                    intrinsic_reward = torch.clamp(agent.compute_intrinsic_reward(step), 0.0, args.max_intrinsic_reward)
            else:
                intrinsic_reward = torch.tensor(0).view(1, 1)
            intrinsic_rewards.extend(list(intrinsic_reward.numpy().reshape(-1)))

            # store experience
            agent.store_rollout(obs[1], action, action_log_probs,
                                value, reward, intrinsic_reward,
                                done)

            # get final episode rewards
            for info in infos:
                if 'episode' in info.keys():
                    extrinsic_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])
                    solved_episodes.append(info['is_success'])

        # compute returns
        agent.compute_returns(args.gamma, args.use_gae, args.gae_lambda)

        # update policy and value_fn, reset rollout storage
        tot_loss, pi_loss, v_loss, dyn_loss, entropy, kl, delta_p, delta_v =  agent.update(obs_mean=obs[2], obs_var=obs[3])

        # log data
        if update % args.log_interval == 0:
            current = time.time()
            elapsed = current - start
            total_steps = (update + 1) * args.num_processes * args.num_steps
            fps =int(total_steps / (current - start))

            logger.logkv('Time/Updates', update)
            logger.logkv('Time/Total Steps', total_steps)
            logger.logkv('Time/FPS', fps)
            logger.logkv('Time/Current', current)
            logger.logkv('Time/Elapsed', elapsed)
            logger.logkv('Time/Epoch', elapsed)
            logger.logkv('Extrinsic/Mean', np.mean(extrinsic_rewards))
            logger.logkv('Extrinsic/Median', np.median(extrinsic_rewards))
            logger.logkv('Extrinsic/Min', np.min(extrinsic_rewards))
            logger.logkv('Extrinsic/Max', np.max(extrinsic_rewards))
            logger.logkv('Episodes/Solved', np.mean(solved_episodes))
            logger.logkv('Episodes/Length', np.mean(episode_length))
            logger.logkv('Intrinsic/Mean', np.mean(intrinsic_rewards))
            logger.logkv('Intrinsic/Median', np.median(intrinsic_rewards))
            logger.logkv('Intrinsic/Min', np.min(intrinsic_rewards))
            logger.logkv('Intrinsic/Max', np.max(intrinsic_rewards))
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
                logger.add_scalar('reward/mean', np.mean(extrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('reward/median', np.median(extrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('reward/min', np.min(extrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('reward/max', np.max(extrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('episode/solved', np.mean(solved_episodes), total_steps, elapsed)
                logger.add_scalar('episode/length', np.mean(episode_length), total_steps, elapsed)
                logger.add_scalar('intrinsic/mean', np.mean(intrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('intrinsic/median', np.median(intrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('intrinsic/min', np.min(intrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('intrinsic/max', np.max(intrinsic_rewards), total_steps, elapsed)
                logger.add_scalar('loss/total', tot_loss, total_steps, elapsed)
                logger.add_scalar('loss/policy', pi_loss, total_steps, elapsed)
                logger.add_scalar('loss/value', v_loss, total_steps, elapsed)
                logger.add_scalar('loss/entropy', entropy, total_steps, elapsed)
                logger.add_scalar('loss/kl', kl, total_steps, elapsed)
                logger.add_scalar('loss/delta_p', delta_p, total_steps, elapsed)
                logger.add_scalar('loss/delta_v', delta_v, total_steps, elapsed)
                logger.add_scalar('loss/dynamics', dyn_loss, total_steps, elapsed)
                logger.add_scalar('value/mean', np.mean(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                logger.add_scalar('value/median', np.median(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                logger.add_scalar('value/min', np.min(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)
                logger.add_scalar('value/max', np.max(agent.rollouts.value_preds.cpu().data.numpy()), total_steps, elapsed)

                if args.debug:
                    logger.add_histogram('debug/actions', agent.rollouts.actions.cpu().data.numpy(), total_steps)
                    logger.add_histogram('debug/observations', agent.rollouts.obs.cpu().data.numpy(), total_steps)
                    logger.logkv('Action/Mean', np.mean(agent.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Median', np.median(agent.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Min', np.min(agent.rollouts.actions.cpu().data.numpy()))
                    logger.logkv('Action/Max', np.max(agent.rollouts.actions.cpu().data.numpy()))

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
                    logger.add_scalar('debug/param/grad_norm', total_grad_norm, total_steps, elapsed)
                    logger.add_scalar('debug/param/weight_norm', total_weight_norm, total_steps, elapsed)

            logger.dumpkvs()

            # checkpoint model
            if (update + 1) % args.checkpoint_interval == 0:
                agent.save_checkpoint()
