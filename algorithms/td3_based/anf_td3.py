import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
from utils import sparse_utils as sp
from utils.mask_adam import MaskAdam
from utils.core import SparseBaseAgent
from utils.activations import setup_activation_funcs_list


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, activation, act_func_args, global_sparsity,
                 sparsity_distribution_method, input_layer_dense, output_layer_dense, device,
                 dim_state_with_fake, args, num_hid_layers=2, num_hid_neurons=256):
        super().__init__()
        assert num_hid_layers >= 1
        self.num_hid_layers = num_hid_layers
        all_connection_layers = num_hid_layers-1 + 2
        # -1 for neuron layers to connection layers, +2 for input and output layer
        self.device = device
        self.permutation = None  # first env is without permutation
        self.num_fake_features = dim_state_with_fake - state_dim
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        sparsities = sp.compute_sparsity_per_layer(
            global_sparsity=global_sparsity,
            neuron_layers=[dim_state_with_fake] + [num_hid_neurons for _ in range(num_hid_layers)] + [action_dim],
            keep_dense=[input_layer_dense] + [False for _ in range(num_hid_layers-1)] + [output_layer_dense],
            method=sparsity_distribution_method,
            input_layer_sparsity=args.input_layer_sparsity)
        self.dense_layers = [True if sparsity == 0 else False for sparsity in sparsities]

        # First define the dense network
        self.input_layer = nn.Linear(dim_state_with_fake, num_hid_neurons)
        self.hid_layers = nn.ModuleList()
        for hid_connection_layer in range(num_hid_layers - 1):
            self.hid_layers.append(nn.Linear(num_hid_neurons, num_hid_neurons))
        self.output_layer = nn.Linear(num_hid_neurons, action_dim)

        # Now make masks for the sparse layers
        self.num_parm_in_layer = [dim_state_with_fake * num_hid_layers] + \
                                 [num_hid_neurons**2 for _ in range(num_hid_layers-1)] + \
                                 [num_hid_neurons * action_dim]
        self.masks = [None for _ in range(all_connection_layers)]
        self.torch_masks = [None for _ in range(all_connection_layers)]
        for layer in range(all_connection_layers):
            if not self.dense_layers[layer]:
                if layer == 0:
                    self.num_parm_in_layer[layer], self.masks[layer] = sp.initialize_mask(
                        f"actor input layer", sparsities[layer], dim_state_with_fake, num_hid_neurons)
                    self.torch_masks[layer] = torch.from_numpy(self.masks[layer]).float().to(device)
                    self.input_layer.weight.data.mul_(torch.from_numpy(self.masks[layer]).float())
                elif layer == all_connection_layers - 1:
                    self.num_parm_in_layer[layer], self.masks[layer] = sp.initialize_mask(
                        f"actor output layer", sparsities[layer], num_hid_neurons, action_dim)
                    self.torch_masks[layer] = torch.from_numpy(self.masks[layer]).float().to(device)
                    self.output_layer.weight.data.mul_(torch.from_numpy(self.masks[layer]).float())
                else:
                    self.num_parm_in_layer[layer], self.masks[layer] = sp.initialize_mask(
                        f"actor hid layer {layer}", sparsities[layer], num_hid_neurons, num_hid_neurons)
                    self.torch_masks[layer] = torch.from_numpy(self.masks[layer]).float().to(device)
                    self.hid_layers[layer - 1].weight.data.mul_(torch.from_numpy(self.masks[layer]).float())
                    # weights are put .to(device) later on, whole network at once

        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, num_hid_layers, num_hid_neurons)
        self.output_activation = nn.Tanh()
        self.max_action = max_action

    def forward(self, state):
        state = utils.add_fake_features(state, self.num_fake_features, self.device,
                                        self.fake_noise_std, self.fake_noise_generator)
        state = utils.permute_features(state, self.permutation)
        a = self.activation_funcs[0](self.input_layer(state))
        for hid_layer in range(self.num_hid_layers - 1):
            a = self.activation_funcs[hid_layer + 1](self.hid_layers[hid_layer](a))
        return self.max_action * self.output_activation(self.output_layer(a))

    def set_new_permutation(self, permutation):
        self.permutation = permutation


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, activation, act_func_args, global_sparsity,
                 sparsity_distribution_method, input_layer_dense, output_layer_dense, device,
                 dim_state_with_fake, args, num_hid_layers=2, num_hid_neurons=256):
        super().__init__()
        assert num_hid_layers >= 1
        self.num_hid_layers = num_hid_layers
        all_connection_layers = num_hid_layers-1 + 2
        self.device = device
        self.permutation = None
        self.num_fake_features = dim_state_with_fake - state_dim
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        sparsities = sp.compute_sparsity_per_layer(
            global_sparsity=global_sparsity,
            neuron_layers=[dim_state_with_fake + action_dim] + [num_hid_neurons for _ in range(num_hid_layers)] + [1],
            keep_dense=[input_layer_dense] + [False for _ in range(num_hid_layers-1)] + [True],
            method=sparsity_distribution_method,
            input_layer_sparsity=args.input_layer_sparsity)
        self.dense_layers = [True if sparsity == 0 else False for sparsity in sparsities]

        # Q1 dense architecture
        self.q1_input_layer = nn.Linear(dim_state_with_fake + action_dim, num_hid_neurons)
        self.q1_hid_layers = nn.ModuleList()
        for hid_connection_layer in range(num_hid_layers - 1):
            self.q1_hid_layers.append(nn.Linear(num_hid_neurons, num_hid_neurons))
        self.q1_output_layer = nn.Linear(num_hid_neurons, 1)

        # Q2 dense architecture
        self.q2_input_layer = nn.Linear(dim_state_with_fake + action_dim, num_hid_neurons)
        self.q2_hid_layers = nn.ModuleList()
        for hid_connection_layer in range(num_hid_layers - 1):
            self.q2_hid_layers.append(nn.Linear(num_hid_neurons, num_hid_neurons))
        self.q2_output_layer = nn.Linear(num_hid_neurons, 1)

        # Setup masks for Q1
        self.q1_num_parm_in_layer = [(dim_state_with_fake+action_dim) * num_hid_layers] + \
                                    [num_hid_neurons**2 for _ in range(num_hid_layers-1)] + \
                                    [num_hid_neurons]
        self.q1_masks = [None for _ in range(all_connection_layers)]
        self.q1_torch_masks = [None for _ in range(all_connection_layers)]
        for layer in range(all_connection_layers):
            if not self.dense_layers[layer]:
                if layer == 0:
                    self.q1_num_parm_in_layer[layer], self.q1_masks[layer] = sp.initialize_mask(
                        f"critic Q1 input layer", sparsities[layer], dim_state_with_fake+action_dim, num_hid_neurons)
                    self.q1_torch_masks[layer] = torch.from_numpy(self.q1_masks[layer]).float().to(device)
                    self.q1_input_layer.weight.data.mul_(torch.from_numpy(self.q1_masks[layer]).float())
                elif layer == all_connection_layers - 1:
                    self.q1_num_parm_in_layer[layer], self.q1_masks[layer] = sp.initialize_mask(
                        f"critic Q1 output layer", sparsities[layer], num_hid_neurons, 1)
                    self.q1_torch_masks[layer] = torch.from_numpy(self.q1_masks[layer]).float().to(device)
                    self.q1_output_layer.weight.data.mul_(torch.from_numpy(self.q1_masks[layer]).float())
                else:
                    self.q1_num_parm_in_layer[layer], self.q1_masks[layer] = sp.initialize_mask(
                        f"critic Q1 hid layer {layer}", sparsities[layer], num_hid_neurons, num_hid_neurons)
                    self.q1_torch_masks[layer] = torch.from_numpy(self.q1_masks[layer]).float().to(device)
                    self.q1_hid_layers[layer-1].weight.data.mul_(torch.from_numpy(self.q1_masks[layer]).float())

        # Setup masks for Q2
        self.q2_num_parm_in_layer = [(dim_state_with_fake+action_dim) * num_hid_layers] + \
                                    [num_hid_neurons**2 for _ in range(num_hid_layers-1)] + \
                                    [num_hid_neurons]
        self.q2_masks = [None for _ in range(all_connection_layers)]
        self.q2_torch_masks = [None for _ in range(all_connection_layers)]
        for layer in range(all_connection_layers):
            if not self.dense_layers[layer]:
                if layer == 0:
                    self.q2_num_parm_in_layer[layer], self.q2_masks[layer] = sp.initialize_mask(
                        f"critic Q2 input layer", sparsities[layer], dim_state_with_fake+action_dim, num_hid_neurons)
                    self.q2_torch_masks[layer] = torch.from_numpy(self.q2_masks[layer]).float().to(device)
                    self.q2_input_layer.weight.data.mul_(torch.from_numpy(self.q2_masks[layer]).float())
                elif layer == all_connection_layers - 1:
                    self.q2_num_parm_in_layer[layer], self.q2_masks[layer] = sp.initialize_mask(
                        f"critic Q2 output layer", sparsities[layer], num_hid_neurons, 1)
                    self.q2_torch_masks[layer] = torch.from_numpy(self.q2_masks[layer]).float().to(device)
                    self.q2_output_layer.weight.data.mul_(torch.from_numpy(self.q2_masks[layer]).float())
                else:
                    self.q2_num_parm_in_layer[layer], self.q2_masks[layer] = sp.initialize_mask(
                        f"critic Q2 hid layer {layer}", sparsities[layer], num_hid_neurons, num_hid_neurons)
                    self.q2_torch_masks[layer] = torch.from_numpy(self.q2_masks[layer]).float().to(device)
                    self.q2_hid_layers[layer-1].weight.data.mul_(torch.from_numpy(self.q2_masks[layer]).float())

        # Activation functions
        self.q1_activation_funcs = setup_activation_funcs_list(activation, act_func_args, num_hid_layers, num_hid_neurons)
        self.q2_activation_funcs = setup_activation_funcs_list(activation, act_func_args, num_hid_layers, num_hid_neurons)

    def forward(self, state, action):
        state = utils.add_fake_features(state, self.num_fake_features, self.device,
                                        self.fake_noise_std, self.fake_noise_generator)
        state = utils.permute_features(state, self.permutation)
        sa = torch.cat([state, action], 1)

        q1 = self.q1_activation_funcs[0](self.q1_input_layer(sa))
        for hid_layer in range(self.num_hid_layers - 1):
            q1 = self.q1_activation_funcs[hid_layer + 1](self.q1_hid_layers[hid_layer](q1))
        q1 = self.q1_output_layer(q1)

        q2 = self.q2_activation_funcs[0](self.q2_input_layer(sa))
        for hid_layer in range(self.num_hid_layers - 1):
            q2 = self.q2_activation_funcs[hid_layer + 1](self.q2_hid_layers[hid_layer](q2))
        q2 = self.q2_output_layer(q2)

        return q1, q2

    def Q1(self, state, action):
        state = utils.add_fake_features(state, self.num_fake_features, self.device,
                                        self.fake_noise_std, self.fake_noise_generator)
        state = utils.permute_features(state, self.permutation)
        sa = torch.cat([state, action], 1)
        q1 = self.q1_activation_funcs[0](self.q1_input_layer(sa))
        for hid_layer in range(self.num_hid_layers - 1):
            q1 = self.q1_activation_funcs[hid_layer + 1](self.q1_hid_layers[hid_layer](q1))
        return self.q1_output_layer(q1)

    def set_new_permutation(self, permutation):
        self.permutation = permutation


class ANF_TD3(SparseBaseAgent):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            args,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            num_hid_layers=2,
            num_hid_neurons=256,
            activation='relu',
            act_func_args=(None, False),
            optimizer='adam',
            lr=0.001,
            global_sparsity=0.5,
            sparsity_distribution_method='ER',
            input_layer_dense=False,
            output_layer_dense=True,
            setZeta=0.05,
            init_new_weights_method='zero',
            ascTopologyChangePeriod=1000,
            earlyStopTopologyChangeIteration=1e9,  # kind of never
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            fake_features=0.0,
    ):
        super().__init__()
        self.dim_state_with_fake = int(np.ceil(state_dim / (1 - fake_features)))
        self.prev_permutations = []

        self.actor = Actor(state_dim, action_dim, max_action, activation, act_func_args, global_sparsity,
                           sparsity_distribution_method, input_layer_dense, output_layer_dense, device,
                           self.dim_state_with_fake, args, num_hid_layers, num_hid_neurons).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, activation, act_func_args, global_sparsity,
                             sparsity_distribution_method, input_layer_dense, output_layer_dense, device,
                             self.dim_state_with_fake, args, num_hid_layers, num_hid_neurons).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.optimizer_name = optimizer
        if optimizer == 'adam':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=0.0002)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=0.0002)
        elif optimizer == 'sgd':
            self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
        elif optimizer == 'maskadam':
            self.actor_optimizer = MaskAdam(self.actor.parameters(), lr=lr, weight_decay=0.0002)
            self.critic_optimizer = MaskAdam(self.critic.parameters(), lr=lr, weight_decay=0.0002)
        else:
            raise ValueError(f'Unknown optimizer {optimizer} given')

        self.device = device
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.setZeta = setZeta
        self.init_new_weights_method = init_new_weights_method
        self.ascTopologyChangePeriod = ascTopologyChangePeriod
        self.earlyStopTopologyChangeIteration = earlyStopTopologyChangeIteration
        self.lastTopologyChangeCritic = False
        self.lastTopologyChangeActor = False
        self.ascStatsActor = []
        self.ascStatsCritic = []
        self.total_it = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.optimizer_name == 'maskadam':
            self.critic_optimizer.step(masks=self.critic.q1_torch_masks + self.critic.q2_torch_masks)
        else:
            self.critic_optimizer.step()
        # Maintain the same sparse connectivity for critic
        self.apply_masks_critic()

        # Adapt the sparse connectivity
        if not self.lastTopologyChangeCritic and self.total_it % self.ascTopologyChangePeriod == 2:
            if self.total_it > self.earlyStopTopologyChangeIteration:
                self.lastTopologyChangeCritic = True
            if self.init_new_weights_method != 'zero':
                q1_old_masks, q2_old_masks = copy.deepcopy(self.critic.q1_masks), copy.deepcopy(self.critic.q2_masks)

            self.update_topology_critic()

            if self.init_new_weights_method != 'zero':
                sp.critic_give_new_connections_init_values(self.critic, q1_old_masks, q2_old_masks,
                                                           self.init_new_weights_method, self.device)
            self.apply_masks_critic()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.optimizer_name == 'maskadam':
                self.actor_optimizer.step(masks=self.actor.torch_masks)
            else:
                self.actor_optimizer.step()
            # Maintain the same sparse connectivity for actor
            self.apply_masks_actor()

            # Adapt the sparse connectivity of the actor
            if not self.lastTopologyChangeActor and self.total_it % self.ascTopologyChangePeriod == 2:
                if self.total_it > self.earlyStopTopologyChangeIteration:
                    self.lastTopologyChangeActor = True
                if self.init_new_weights_method != 'zero':
                    old_masks = copy.deepcopy(self.actor.masks)

                self.update_topology_actor()

                if self.init_new_weights_method != 'zero':
                    sp.actor_give_new_connections_init_values(self.actor, old_masks,
                                                              self.init_new_weights_method, self.device)
                self.apply_masks_actor()

            # Update the frozen target models
            self.update_target_networks()

    def update_topology_critic(self):
        for layer in range(self.critic.num_hid_layers + 1):
            if not self.critic.dense_layers[layer]:
                if layer == 0:
                    self.critic.q1_masks[layer] = sp.adjust_connectivity_set(
                        self.critic.q1_input_layer.weight.data.cpu().numpy(), self.critic.q1_num_parm_in_layer[layer],
                        self.setZeta, self.critic.q1_masks[layer])
                    self.critic.q2_masks[layer] = sp.adjust_connectivity_set(
                        self.critic.q2_input_layer.weight.data.cpu().numpy(), self.critic.q2_num_parm_in_layer[layer],
                        self.setZeta, self.critic.q2_masks[layer])
                elif layer == self.critic.num_hid_layers:
                    self.critic.q1_masks[layer] = sp.adjust_connectivity_set(
                        self.critic.q1_output_layer.weight.data.cpu().numpy(), self.critic.q1_num_parm_in_layer[layer],
                        self.setZeta, self.critic.q1_masks[layer])
                    self.critic.q2_masks[layer] = sp.adjust_connectivity_set(
                        self.critic.q2_output_layer.weight.data.cpu().numpy(), self.critic.q2_num_parm_in_layer[layer],
                        self.setZeta, self.critic.q2_masks[layer])
                else:
                    self.critic.q1_masks[layer] = sp.adjust_connectivity_set(
                        self.critic.q1_hid_layers[layer - 1].weight.data.cpu().numpy(),
                        self.critic.q1_num_parm_in_layer[layer], self.setZeta, self.critic.q1_masks[layer])
                    self.critic.q2_masks[layer] = sp.adjust_connectivity_set(
                        self.critic.q2_hid_layers[layer - 1].weight.data.cpu().numpy(),
                        self.critic.q2_num_parm_in_layer[layer], self.setZeta, self.critic.q2_masks[layer])
                self.critic.q1_torch_masks[layer] = torch.from_numpy(self.critic.q1_masks[layer]).float().to(
                    self.device)
                self.critic.q2_torch_masks[layer] = torch.from_numpy(self.critic.q2_masks[layer]).float().to(
                    self.device)

    def update_topology_actor(self):
        for layer in range(self.actor.num_hid_layers + 1):
            if not self.actor.dense_layers[layer]:
                if layer == 0:
                    self.actor.masks[layer] = sp.adjust_connectivity_set(
                        self.actor.input_layer.weight.data.cpu().numpy(),
                        self.actor.num_parm_in_layer[layer], self.setZeta, self.actor.masks[layer])
                elif layer == self.actor.num_hid_layers:
                    self.actor.masks[layer] = sp.adjust_connectivity_set(
                        self.actor.output_layer.weight.data.cpu().numpy(),
                        self.actor.num_parm_in_layer[layer], self.setZeta, self.actor.masks[layer])
                else:
                    self.actor.masks[layer] = sp.adjust_connectivity_set(
                        self.actor.hid_layers[layer - 1].weight.data.cpu().numpy(),
                        self.actor.num_parm_in_layer[layer], self.setZeta, self.actor.masks[layer])
                self.actor.torch_masks[layer] = torch.from_numpy(self.actor.masks[layer]).float().to(self.device)

    def update_target_networks(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if len(param.shape) > 1:
                self.maintain_sparsity_target_networks(param, target_param, self.device)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if len(param.shape) > 1:
                self.maintain_sparsity_target_networks(param, target_param, self.device)

    def maintain_sparsity_target_networks(self, param, target_param, device):
        current_density = (param != 0).sum()
        target_density = (target_param != 0).sum()  # torch.count_nonzero(target_param.data)
        difference = target_density - current_density
        # constrain the sparsity by removing the extra elements (smallest values)
        if difference > 0:
            count_rmv = difference
            tmp = copy.deepcopy(abs(target_param.data))
            tmp[tmp == 0] = 10000000
            unraveled = self.unravel_index(torch.argsort(tmp.view(1, -1)[0]), tmp.shape)
            rmv_indicies = torch.stack(unraveled, dim=1)
            rmv_values_smaller_than = tmp[rmv_indicies[count_rmv][0], rmv_indicies[count_rmv][1]]
            target_param.data[tmp < rmv_values_smaller_than] = 0

    def unravel_index(self, index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def apply_masks_critic(self):
        for layer in range(self.critic.num_hid_layers + 1):
            if not self.critic.dense_layers[layer]:
                if layer == 0:
                    self.critic.q1_input_layer.weight.data.mul_(self.critic.q1_torch_masks[layer])
                    self.critic.q2_input_layer.weight.data.mul_(self.critic.q2_torch_masks[layer])
                elif layer == self.critic.num_hid_layers:
                    self.critic.q1_output_layer.weight.data.mul_(self.critic.q1_torch_masks[layer])
                    self.critic.q2_output_layer.weight.data.mul_(self.critic.q2_torch_masks[layer])
                else:
                    self.critic.q1_hid_layers[layer - 1].weight.data.mul_(self.critic.q1_torch_masks[layer])
                    self.critic.q2_hid_layers[layer - 1].weight.data.mul_(self.critic.q2_torch_masks[layer])

    def apply_masks_actor(self):
        for layer in range(self.actor.num_hid_layers + 1):
            if not self.actor.dense_layers[layer]:
                if layer == 0:
                    self.actor.input_layer.weight.data.mul_(self.actor.torch_masks[layer])
                elif layer == self.actor.num_hid_layers:
                    self.actor.output_layer.weight.data.mul_(self.actor.torch_masks[layer])
                else:
                    self.actor.hid_layers[layer - 1].weight.data.mul_(self.actor.torch_masks[layer])

    def print_sparsity(self):
        return sp.print_sparsities(self.critic.parameters(), self.critic_target.parameters(),
                                   self.actor.parameters(), self.actor_target.parameters())

    def saveAscStats(self, filename):
        np.savez(filename + "_ASC_stats.npz", ascStatsActor=self.ascStatsActor, ascStatsCritic=self.ascStatsCritic)

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
        self.actor.set_new_permutation(permutation)
        self.critic.set_new_permutation(permutation)
        self.actor_target.set_new_permutation(permutation)
        self.critic_target.set_new_permutation(permutation)
