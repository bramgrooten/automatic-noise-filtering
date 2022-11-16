import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from utils.target_network import soft_update, hard_update
from utils.core_anf_sac import GaussianPolicy, QNetwork, DeterministicPolicy
import utils.sparse_utils as sp


class Static_SAC(object):
    def __init__(self, state_dim, action_space, args, device):
        self.device = device

        self.gamma = args.discount
        self.tau = args.tau
        self.alpha = args.temperature

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.total_it = 0
        self.setZeta = args.ann_setZeta
        self.ascTopologyChangePeriod = args.ann_ascTopologyChangePeriod
        self.lastTopologyChangeCritic = False
        self.lastTopologyChangePolicy = False
        self.ascStatsPolicy = []
        self.ascStatsCritic = []
        self.ascStatsValue = []

        self.dim_state_with_fake = int(np.ceil(state_dim / (1 - args.fake_features)))
        self.prev_permutations = []

        self.critic = QNetwork(state_dim, action_space.shape[0], args,
                               self.dim_state_with_fake, self.device).to(device=self.device)
        self.critic_target = QNetwork(state_dim, action_space.shape[0], args,
                                      self.dim_state_with_fake, self.device).to(self.device)
        hard_update(self.critic_target, self.critic)

        if args.sac_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.policy = GaussianPolicy(state_dim, action_space.shape[0], args,
                                         self.dim_state_with_fake, self.device, action_space).to(self.device)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_dim, action_space.shape[0], args,
                                              self.dim_state_with_fake, self.device, action_space).to(self.device)

        if args.optimizer in ['adam', 'maskadam']:  # for all-dense networks: adam == maskadam
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr, weight_decay=0.0002)
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr, weight_decay=0.0002)
        elif args.optimizer == 'sgd':
            self.policy_optim = SGD(self.policy.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
            self.critic_optim = SGD(self.critic.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
        else:
            raise ValueError(f'Unknown optimizer {args.optimizer} given')

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        self.total_it += 1

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + done_batch * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Maintain the same sparse connectivity for critic
        self.apply_masks_critic()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Maintain the same sparse connectivity for actor
        self.apply_masks_actor()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        loss_info = {'q1_loss': qf1_loss.item(),
                     'q2_loss': qf2_loss.item(),
                     'actor_loss': policy_loss.item(),
                     'alpha_loss': alpha_loss.item(),
                     'alpha_val': alpha_tlogs.item()}
        return loss_info

    def apply_masks_critic(self):
        if not self.critic.dense_layers[0]:
            self.critic.linear1.weight.data.mul_(self.critic.torchMask1)
            self.critic.linear4.weight.data.mul_(self.critic.torchMask4)
        if not self.critic.dense_layers[1]:
            self.critic.linear2.weight.data.mul_(self.critic.torchMask2)
            self.critic.linear5.weight.data.mul_(self.critic.torchMask5)

    def apply_masks_actor(self):
        if not self.policy.dense_layers[0]:
            self.policy.linear1.weight.data.mul_(self.policy.torchMask1)
        if not self.policy.dense_layers[1]:
            self.policy.linear2.weight.data.mul_(self.policy.torchMask2)

    def print_sparsity(self):
        return sp.print_sparsities(self.critic.parameters(), self.critic_target.parameters(), self.policy.parameters())

    def set_new_permutation(self):
        # sample a new permutation until it is not a duplicate
        duplicate = True
        while duplicate:
            permutation = torch.randperm(self.dim_state_with_fake)
            for p in self.prev_permutations:
                if torch.equal(p, permutation):
                    break
            else:
                duplicate = False
        print(f'\nEnvironment change: new permutation of input features.')
        self.prev_permutations.append(permutation)
        self.policy.set_new_permutation(permutation)
        self.critic.set_new_permutation(permutation)
        self.critic_target.set_new_permutation(permutation)

    # Save model parameters
    def save(self, filename):
        checkpoint = {
            'actor': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.policy_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Saved current model in: {filename}")

    # Load model parameters
    def load(self, filename, load_device=None):
        if load_device is None:
            load_device = self.device
        loaded_checkpoint = torch.load(filename, map_location=load_device)
        self.policy.load_state_dict(loaded_checkpoint["actor"])
        self.policy_optim.load_state_dict(loaded_checkpoint["actor_optim"])
        self.critic.load_state_dict(loaded_checkpoint["critic"])
        self.critic_target.load_state_dict(loaded_checkpoint["critic_target"])
        self.critic_optim.load_state_dict(loaded_checkpoint["critic_optim"])
        print(f"Loaded model from: {filename}")

