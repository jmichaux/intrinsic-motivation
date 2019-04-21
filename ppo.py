import os
import numpy as np
import torch
import torch.optim as optim

from rollouts import Rollouts

class PPO():
    def __init__(self,
                 log_dir,
                 observation_space,
                 action_space,
                 actor_critic,
                 dynamics_model,
                 optimizer=optim.Adam,
                 hidden_size=64,
                 num_steps=2048,
                 num_processes=1,
                 ppo_epochs=10,
                 num_mini_batch=32,
                 pi_lr=3e-4,
                 v_lr=1e-3,
                 dyn_lr=1e-3,
                 clip_param=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 dyn_coef=0.5,
                 intrinsic_coef = 0.01,
                 grad_norm_max=0.5,
                 use_clipped_value_loss=True,
                 use_tensorboard=True,
                 add_intrinsic_reward=False,
                 device='cpu',
                 share_optim=False,
                 debug=False):

        # setup logging
        self.checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')

        # ppo hyperparameters
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batch = num_mini_batch

        # loss hyperparameters
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.dyn_coef = dyn_coef
        self.intrinsic_coef = intrinsic_coef
        # clip values
        self.grad_norm_max = grad_norm_max
        self.use_clipped_value_loss = use_clipped_value_loss
        self.add_intrinsic_reward = add_intrinsic_reward

        # setup actor critic
        self.actor_critic = actor_critic(
            num_inputs=observation_space.shape[0],
            hidden_size=hidden_size,
            num_outputs=action_space.shape[0])
        self.actor_critic.to(device)

        # setup dynamics model
        if self.add_intrinsic_reward:
            dynamics_dim = observation_space.shape[0] + action_space.shape[0]
            self.dynamics_model = dynamics_model(num_inputs=dynamics_dim,
                                                 hidden_size=hidden_size,
                                                 num_outputs=observation_space.shape[0])
            self.dynamics_model.to(device)

        # setup optimizers
        self.share_optim = share_optim
        if self.share_optim:
            if self.add_intrinsic_reward:
                self.optimizer = optimizer(list(self.actor_critic.parameters()) + list(self.dynamics_model.parameters()), lr=pi_lr)
            else:
                self.optimizer = optimizer(self.actor_critic.parameters(), lr=pi_lr)
        else:
            self.policy_optimizer = optimizer(self.actor_critic.policy.parameters(), lr=pi_lr)
            self.value_fn_optimizer = optimizer(self.actor_critic.value_fn.parameters(), lr=v_lr)
            if self.add_intrinsic_reward:
                self.dynamics_optimizer = optimizer(self.dynamics_model.parameters(), lr=dyn_lr)

        # create rollout storage
        self.rollouts = Rollouts(num_steps, num_processes,
                                 observation_space.shape,
                                 action_space,
                                 device)

    def train(self):
        self.actor_critic.train()
        if self.add_intrinsic_reward:
            self.dynamics_model.train()

    def eval(self):
        self.actor_critic.eval()
        if self.add_intrinsic_reward:
            self.dynamics_model.eval()

    def select_action(self, step):
        with torch.no_grad():
            return self.actor_critic.select_action(self.rollouts.obs[step])

    def evaluate_action(self, obs, action):
        return self.actor_critic.evaluate_action(obs, action)

    def get_value(self, obs):
        with torch.no_grad():
            return self.actor_critic.get_value(obs)

    def store_rollout(self, obs, action, action_log_probs, value, reward, intrinsic_reward, done):
        masks = torch.tensor(1.0 - done.astype(np.float32)).view(-1, 1)
        self.rollouts.insert(obs, action, action_log_probs, value, reward, intrinsic_reward, masks)

    def compute_returns(self, gamma, use_gae=True, gae_lambda=0.95):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.rollouts.obs[-1]).detach()
        if self.add_intrinsic_reward:
            self.rollouts.rewards += self.rollouts.intrinsic_rewards
        self.rollouts.compute_returns(next_value, gamma, use_gae, gae_lambda)

    def compute_intrinsic_reward(self, step):
        with torch.no_grad():
            obs = self.rollouts.obs[step]
            action = self.rollouts.actions[step]
            next_obs = self.rollouts.obs[step + 1]
            next_obs_preds = self.dynamics_model(obs, action)
            # return 0.5 * self.eta * torch.norm(next_obs - next_obs_preds, p=2, dim=-1).pow(2).unsqueeze(-1)
            return 0.5 * self.intrinsic_coef * (next_obs_preds - next_obs).pow(2).sum(-1).unsqueeze(-1)

    def update(self):
        tot_loss, pi_loss, v_loss, dyn_loss, ent, kl, delta_p, delta_v = self._update()

        self.rollouts.after_update()
        return tot_loss, pi_loss, v_loss, dyn_loss, ent, kl, delta_p, delta_v

    def compute_loss(self, sample):
        # get sample batch
        obs_batch, actions_batch, next_obs_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_target = sample

        # evaluate actions
        values, action_log_probs, entropy = self.actor_critic.evaluate_action(obs_batch, actions_batch)

        # compute policy loss
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        sur1 = ratio * adv_target
        sur2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_target
        policy_loss = -torch.min(sur1, sur2).mean()

        # compute value loss
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (return_batch - values).pow(2).mean()
            value_losses_clipped = (return_batch - value_pred_clipped).pow(2).mean()
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        # compute dynamics loss
        if self.add_intrinsic_reward:
            dynamics_loss = self.compute_dynamics_loss(obs_batch, actions_batch, next_obs_batch)
        else:
            dynamics_loss = 0

        # compute total loss
        total_loss =  self.value_coef * value_loss + self.dyn_coef * dynamics_loss \
                    + (policy_loss - self.entropy_coef * entropy)

        # compute kl divergence
        kl = (old_action_log_probs_batch - action_log_probs).mean().detach()

        return total_loss, policy_loss, value_loss, dynamics_loss, entropy, kl

    def compute_dynamics_loss(self, obs, action, next_obs):
        next_obs_preds = self.dynamics_model(obs, action)
        return 0.5 * (next_obs_preds - next_obs).pow(2).sum(-1).unsqueeze(-1).mean()
        # return 0.5 * torch.norm(next_obs - next_obs_preds, p=2, dim=-1).unsqueeze(-1).mean()

    def _update(self):
        # compute and normalize advantages
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # policy and value losses before gradient update
        with torch.no_grad():
            # Get whole batch of data
            update_generator = self.rollouts.feed_forward_generator(advantages, num_mini_batch=1)
            for update_sample in update_generator:
                _, policy_loss_old, value_loss_old, _, _, _ = self.compute_loss(update_sample)

        total_loss_epoch = 0
        policy_loss_epoch = 0
        value_loss_epoch = 0
        dynamics_loss_epoch = 0
        entropy_epoch = 0
        kl_epoch = 0

        for epoch in range(self.ppo_epochs):
            data_generator = self.rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                total_loss, policy_loss, value_loss, dynamics_loss, entropy, kl = self.compute_loss(sample)

                if self.share_optim:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.grad_norm_max)
                    self.optimizer.step()

                    if not self.add_intrinsic_reward:
                        dynamics_loss = torch.tensor(0).view(1, 1)
                else:
                    self.policy_optimizer.zero_grad()
                    (policy_loss - self.entropy_coef * entropy).backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.policy.parameters(), self.grad_norm_max)
                    self.policy_optimizer.step()

                    self.value_fn_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.value_fn.parameters(), self.grad_norm_max)
                    self.value_fn_optimizer.step()

                    if self.add_intrinsic_reward:
                        self.dynamics_optimizer.zero_grad()
                        dynamics_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.grad_norm_max)
                        self.dynamics_optimizer.step()
                    else:
                        dynamics_loss = torch.tensor(0).view(1, 1)

                total_loss_epoch += total_loss.item()
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                dynamics_loss_epoch += dynamics_loss.item()
                entropy_epoch += entropy.item()
                kl_epoch += kl.item()

        num_updates = (self.ppo_epochs + 1) * self.num_mini_batch
        total_loss_epoch /= num_updates
        policy_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        dynamics_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        kl_epoch /= num_updates


        # policy and value losses after gradient update
        with torch.no_grad():
            _, policy_loss_new, value_loss_new, _, _, _ = self.compute_loss(update_sample)
            delta_p = policy_loss_new - policy_loss_old
            delta_v = value_loss_new - value_loss_old

        return total_loss_epoch, policy_loss_epoch, value_loss_epoch, dynamics_loss_epoch, entropy_epoch, kl_epoch, delta_p.item(), delta_v.item()

    def save_checkpoint(self, path=None):
        # create checkpoint dict
        checkpoint = {
            'share_optim': self.share_optim,
            'add_intrinsic_reward': self.add_intrinsic_reward}

        # save models
        checkpoint['actor_critic'] =  self.actor_critic.state_dict()
        if self.add_intrinsic_reward:
            checkpoint['dynamics_model'] = self.dynamics_model.state_dict()

        # save optimizer(s)
        if self.share_optim:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        else:
            checkpoint['policy_optimizer'] = self.policy_optimizer.state_dict()
            checkpoint['value_fn_optimizer'] = self.value_fn_optimizer.state_dict()
            if self.add_intrinsic_reward:
                checkpoint['dynamics_model'] = self.dynamics_model.state_dict()
                checkpoint['dynamics_optimizer'] = self.dynamics_optimizer.state_dict()

        if path is None:
            torch.save(checkpoint, self.checkpoint_path)
        else:
            torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        # load checkpoint
        checkpoint = torch.load(path)
        self.share_optim = checkpoint['share_optim']
        self.add_intrinsic_reward = checkpoint['add_intrinsic_reward']

        # load models
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        if self.add_intrinsic_reward:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])

        # load optimizer(s)
        if self.share_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_fn_optimizer.load_state_dict(checkpoint['value_fn_optimizer'])
            if self.add_intrinsic_reward:
                self.dynamics_optimizer.load_state_dict(checkpoint['dynamics_optimizer'])

    def load_models(self, path):
        checkpoint = torch.load(path)
        # load models
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        if self.add_intrinsic_reward:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])
        del checkpoint
