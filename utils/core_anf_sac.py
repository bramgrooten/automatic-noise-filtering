import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils.sparse_utils as sp
from utils import utils
from utils.activations import setup_activation_funcs_list

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, args, dim_state_with_fake, device):
        super(QNetwork, self).__init__()
        hidden_dim = args.num_hid_neurons
        self.device = device
        self.num_fake_features = dim_state_with_fake - state_dim
        self.permutation = None
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        sparsities = sp.compute_sparsity_per_layer(
            global_sparsity=args.global_sparsity,
            neuron_layers=[dim_state_with_fake + action_dim, hidden_dim, hidden_dim, 1],
            keep_dense=[(args.input_layer_sparsity == 0), False, True],  # sparse output layer not implemented yet
            method=args.sparsity_distribution_method,
            input_layer_sparsity=args.input_layer_sparsity)
        self.dense_layers = [True if sparsity == 0 else False for sparsity in sparsities]

        # Q1 architecture
        self.linear1 = nn.Linear(dim_state_with_fake + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(dim_state_with_fake + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        activation = args.activation
        act_func_args = (args.act_func_args, args.act_func_per_neuron)
        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, args.num_hid_layers, args.num_hid_neurons)

        if not self.dense_layers[0]:
            self.noPar1, self.mask1 = sp.initialize_mask(
                'critic Q1 first layer', sparsities[0], dim_state_with_fake + action_dim, hidden_dim)
            self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
            self.linear1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        if not self.dense_layers[1]:
            self.noPar2, self.mask2 = sp.initialize_mask(
                'critic Q1 second layer', sparsities[1], hidden_dim, hidden_dim)
            self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
            self.linear2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        if not self.dense_layers[0]:
            self.noPar4, self.mask4 = sp.initialize_mask(
                'critic Q2 first layer', sparsities[0], dim_state_with_fake + action_dim, hidden_dim)
            self.torchMask4 = torch.from_numpy(self.mask4).float().to(device)
            self.linear4.weight.data.mul_(torch.from_numpy(self.mask4).float())

        if not self.dense_layers[1]:
            self.noPar5, self.mask5 = sp.initialize_mask(
                'critic Q2 second layer', sparsities[1], hidden_dim, hidden_dim)
            self.torchMask5 = torch.from_numpy(self.mask5).float().to(device)
            self.linear5.weight.data.mul_(torch.from_numpy(self.mask5).float())

    def forward(self, state, action):
        state = utils.add_fake_features(state, self.num_fake_features, self.device,
                                        self.fake_noise_std, self.fake_noise_generator)
        state = utils.permute_features(state, self.permutation)
        xu = torch.cat([state, action], 1)

        x1 = self.activation_funcs[0](self.linear1(xu))
        x1 = self.activation_funcs[1](self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = self.activation_funcs[0](self.linear4(xu))
        x2 = self.activation_funcs[1](self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def set_new_permutation(self, permutation):
        self.permutation = permutation


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, args, dim_state_with_fake, device, action_space=None):
        super(GaussianPolicy, self).__init__()
        hidden_dim = args.num_hid_neurons
        self.device = device
        self.num_fake_features = dim_state_with_fake - state_dim
        self.permutation = None
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        sparsities = sp.compute_sparsity_per_layer(
            global_sparsity=args.global_sparsity,
            neuron_layers=[dim_state_with_fake, hidden_dim, hidden_dim, 2 * action_dim],  # *2 for two heads: mean and log_std
            keep_dense=[(args.input_layer_sparsity == 0), False, True],  # sparse output layer not implemented yet
            method=args.sparsity_distribution_method,
            input_layer_sparsity=args.input_layer_sparsity)
        self.dense_layers = [True if sparsity == 0 else False for sparsity in sparsities]

        self.linear1 = nn.Linear(dim_state_with_fake, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        activation = args.activation
        act_func_args = (args.act_func_args, args.act_func_per_neuron)
        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, args.num_hid_layers, args.num_hid_neurons)

        if not self.dense_layers[0]:
            self.noPar1, self.mask1 = sp.initialize_mask(
                'Gaussian actor input layer', sparsities[0], dim_state_with_fake, hidden_dim)
            self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
            self.linear1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        if not self.dense_layers[1]:
            self.noPar2, self.mask2 = sp.initialize_mask(
                'Gaussian actor hidden layer', sparsities[1], hidden_dim, hidden_dim)
            self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
            self.linear2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        state = utils.add_fake_features(state, self.num_fake_features, self.device,
                                        self.fake_noise_std, self.fake_noise_generator)
        state = utils.permute_features(state, self.permutation)
        x = self.activation_funcs[0](self.linear1(state))
        x = self.activation_funcs[1](self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    def set_new_permutation(self, permutation):
        self.permutation = permutation


class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, args, dim_state_with_fake, device, action_space=None):
        super(DeterministicPolicy, self).__init__()
        hidden_dim = args.num_hid_neurons
        self.device = device
        self.num_fake_features = dim_state_with_fake - state_dim
        self.permutation = None
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        sparsities = sp.compute_sparsity_per_layer(
            global_sparsity=args.global_sparsity,
            neuron_layers=[dim_state_with_fake, hidden_dim, hidden_dim, action_dim],
            keep_dense=[(args.input_layer_sparsity == 0), False, True],  # sparse output layer not implemented yet
            method=args.sparsity_distribution_method,
            input_layer_sparsity=args.input_layer_sparsity)
        self.dense_layers = [True if sparsity == 0 else False for sparsity in sparsities]

        self.linear1 = nn.Linear(dim_state_with_fake, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.noise = torch.Tensor(action_dim)

        self.apply(weights_init_)

        activation = args.activation
        act_func_args = (args.act_func_args, args.act_func_per_neuron)
        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, args.num_hid_layers, args.num_hid_neurons)

        if not self.dense_layers[0]:
            self.noPar1, self.mask1 = sp.initialize_mask(
                'Gaussian actor input layer', sparsities[0], dim_state_with_fake, hidden_dim)
            self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
            self.linear1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        if not self.dense_layers[1]:
            self.noPar2, self.mask2 = sp.initialize_mask(
                'Gaussian actor hidden layer', sparsities[1], hidden_dim, hidden_dim)
            self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
            self.linear2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        state = utils.add_fake_features(state, self.num_fake_features, self.device,
                                        self.fake_noise_std, self.fake_noise_generator)
        state = utils.permute_features(state, self.permutation)
        x = self.activation_funcs[0](self.linear1(state))
        x = self.activation_funcs[1](self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

    def set_new_permutation(self, permutation):
        self.permutation = permutation

