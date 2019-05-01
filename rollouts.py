"""
Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/storage.py
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Rollouts(object):
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_shape,
                 action_space,
                 contact_shape,
                 device=None,
                 use_gae=False):

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.contacts = torch.zeros(num_steps + 1, num_processes, *contact_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.intrinsic_rewards = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'
        self.to(self.device)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device=None):
        if device is None:
            device = self.device
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.intrinsic_rewards = self.intrinsic_rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, contacts, actions, action_log_probs, value_preds, rewards, intrinsic_rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.contacts[self.step + 1].copy_(contacts)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.intrinsic_rewards[self.step].copy_(intrinsic_rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        After updating move the last observation and mask
        to the begining of the rollout storage
        """
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.contacts[0].copy_(self.contacts[-1])

    def compute_returns(self, next_value, gamma=0.99, use_gae=True, gae_lambda=0.95):

        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step +
                                           1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step +
                                                              1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]


    def feed_forward_generator(self, advantages, num_mini_batch):
        # get number of steps and number of processes
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_steps * num_processes
        # make sure the size of the batch is greater than the number of mini batches
        assert batch_size >= num_mini_batch
        # size of minibatch is size of big batch / number of minibatches
        mini_batch_size = batch_size // num_mini_batch
        # This will randomly partition indices will keep the last partition even
        # if it isn't the same size as mini_batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            contacts_batch = self.contacts[:-1].view(-1, *self.contacts.size()[2:])[indices]
            actions_batch = self.actions.view(-1, *self.actions.size()[2:])[indices]
            next_obs_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_target = advantages.view(-1, 1)[indices]

            yield obs_batch, contacts_batch, actions_batch, next_obs_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_target

    def curiosity_generator(self, num_mini_batch):
        # get number of steps and number of processes
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_steps * num_processes
        # make sure the size of the batch is greater than the number of mini batches
        assert batch_size >= num_mini_batch
        # size of minibatch is size of big batch / number of minibatches
        mini_batch_size = batch_size // num_mini_batch
        # This will randomly partition indices will keep the last partition even
        # if it isn't the same size as mini_batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            next_obs_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, *self.actions.size()[2:])[indices]

            yield obs_batch, actions_batch, next_obs_batch
