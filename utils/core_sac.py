import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import utils
from utils.activations import setup_activation_funcs_list
# was called model.py in the original code

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(ValueNetwork, self).__init__()
#
#         self.linear1 = nn.Linear(state_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, 1)
#
#         self.apply(weights_init_)
#
#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, args, dim_state_with_fake, device):
        super(QNetwork, self).__init__()
        hidden_dim = args.num_hid_neurons
        self.device = device
        self.num_fake_features = dim_state_with_fake - state_dim
        self.permutation = None
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        # Q1 architecture
        self.linear1 = nn.Linear(dim_state_with_fake + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(dim_state_with_fake + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        activation = args.activation
        act_func_args = (args.act_func_args, args.act_func_per_neuron)
        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, args.num_hid_layers, args.num_hid_neurons)

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
    def __init__(self, state_dim, num_actions, args, dim_state_with_fake, device, action_space=None):
        super(GaussianPolicy, self).__init__()
        hidden_dim = args.num_hid_neurons
        self.device = device
        self.num_fake_features = dim_state_with_fake - state_dim
        self.permutation = None
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        self.linear1 = nn.Linear(dim_state_with_fake, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        activation = args.activation
        act_func_args = (args.act_func_args, args.act_func_per_neuron)
        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, args.num_hid_layers, args.num_hid_neurons)

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
    def __init__(self, state_dim, num_actions, args, dim_state_with_fake, device, action_space=None):
        super(DeterministicPolicy, self).__init__()
        hidden_dim = args.num_hid_neurons
        self.device = device
        self.num_fake_features = dim_state_with_fake - state_dim
        self.permutation = None
        self.fake_noise_std = args.fake_noise_std
        self.fake_noise_generator = utils.setup_noise_generator(args.load_noise_distribution)

        self.linear1 = nn.Linear(dim_state_with_fake, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        activation = args.activation
        act_func_args = (args.act_func_args, args.act_func_per_neuron)
        self.activation_funcs = setup_activation_funcs_list(activation, act_func_args, args.num_hid_layers, args.num_hid_neurons)

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

