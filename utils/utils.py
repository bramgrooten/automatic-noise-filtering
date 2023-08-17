import os
import random
import numpy as np
import torch
import gym
import wandb
import datetime
from utils.load_feats_distr import RealFeatureDistribution
from algorithms.td3_based import td3, anf_td3, ss_td3
from algorithms.sac_based import sac, anf_sac, ss_sac


# Runs policy for X episodes and returns average reward
def eval_policy(policy, env, seed=2, print_comments=True, eval_episodes=10):
    # env.reset(seed=seed + 100)  # env is being reset in for loop anyway
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), evaluate=True)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    if print_comments:
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episode(s): {avg_reward:.3f}")
        print("---------------------------------------")
    return avg_reward


def eval_policy_updated_actor(policy, env, seed, print_comments=True, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        while not done:
            action = policy.eval_select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    if print_comments:
        print("---------------------------------------")
        print(f"Evaluation of updated actor over {eval_episodes} episode(s): {avg_reward:.3f}")
        print("---------------------------------------")
    return avg_reward


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.array(random.sample(range(self.size), batch_size))
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def empty_buffer(self):
        self.state = np.zeros((self.max_size, self.state.shape[1]))
        self.action = np.zeros((self.max_size, self.action.shape[1]))
        self.next_state = np.zeros((self.max_size, self.next_state.shape[1]))
        self.reward = np.zeros((self.max_size, self.reward.shape[1]))
        self.not_done = np.zeros((self.max_size, self.not_done.shape[1]))
        self.size = 0
        self.ptr = 0

    def save_buffer(self, filename):
        np.savez(filename,
                 state=self.state,
                 action=self.action,
                 next_state=self.next_state,
                 reward=self.reward,
                 not_done=self.not_done)

    def load_buffer(self, filename):
        npzfile = np.load(filename + '.npz')
        self.state = npzfile['state']
        self.action = npzfile['action']
        self.next_state = npzfile['next_state']
        self.reward = npzfile['reward']
        self.not_done = npzfile['not_done']


def print_all_args(args):
    print("\nAll settings used in this call to main.py: ")
    for arg_name, arg_value in sorted(vars(args).items()):
        num_tabs = 1 if len(str(arg_value)) >= 8 else 2
        tabs = num_tabs * '\t'
        print(f"{arg_value}{tabs}{arg_name}")


def add_arguments(parser):
    parser.add_argument("--policy", default="ANF-SAC",
                        help='Policy name (ANF-SAC, ANF-TD3, SAC, TD3, Static-SAC, or Static-TD3). Default: ANF-SAC')
    parser.add_argument("--sac_type", default="Gaussian",
                        help='SAC type (Gaussian, Deterministic). Default: Gaussian')
    parser.add_argument("--env", default="HalfCheetah-v3",
                        help='Environment name. Options from OpenAI gym: '
                             'HalfCheetah-v3, Hopper-v3, Walker2d-v3, Humanoid-v3, Ant-v3.')
    parser.add_argument("--adjust_env_period", default=-1, type=int,
                        help='Period after which the environment is adjusted (Can be used for continual-envs, '
                             'but also the MuJoCo Gym envs, which will permute their real & fake input features '
                             'differently after every env change). '
                             'Default: -1 (means no env changes at all). Good value to use is 500_000.')
    parser.add_argument("--env_init_friction", default=1.5, type=float,
                        help='Initial friction value (only for SlipperyAnt). Default: 1.5')
    parser.add_argument("--empty_buffer_on_env_change", action='store_true',
                        help='To empty the replay buffer when the environment changes. Default: False. '
                             'When arg is used, this is set to True.')
    parser.add_argument('--buffer_size', type=int, default=1_000_000,
                        help='Max capacity of the experience replay buffer (default: 1e6)')

    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"],
                        help='Wandb mode (online, offline, disabled). Default: online')
    parser.add_argument("--seed", default=42, type=int,
                        help='Random seed. Sets PyTorch and Numpy seeds')
    parser.add_argument("--start_timesteps", default=25e3, type=int,
                        help='Number of time steps where initial random policy is used to fill ReplayBuffer')
    parser.add_argument("--refill_timesteps", default=25e3, type=int,
                        help='Number of time steps used to refill ReplayBuffer after env change')
    parser.add_argument("--refill_mode", default="random",
                        help="Method used to refill ReplayBuffer after env change. Options: random, current. "
                             "(current uses the current policy on the new env.) Default: random.")
    parser.add_argument("--eval_freq", default=5e3, type=int,
                        help='How often (time steps) we evaluate')
    parser.add_argument("--eval_episodes", default=5, type=int,
                        help='Number of episodes to evaluate for')
    parser.add_argument("--max_timesteps", default=1e6, type=int,
                        help='Max time steps to run environment')
    parser.add_argument("--batch_size", default=100, type=int,
                        help='Batch size for both actor and critic')
    parser.add_argument("--optimizer", default="adam",
                        help="Optimizer. Options: adam, maskadam, sgd. Default: adam "
                             "(for dense/static TD3 and SAC maskadam is the same as adam).")
    parser.add_argument("--lr", default=0.001, type=float,
                        help='Learning rate. Default: 1e-3 (0.001)')
    parser.add_argument("--discount", default=0.99,
                        help='Discount factor. Default: 0.99')
    parser.add_argument("--tau", default=0.005,
                        help='Target network update rate. Default: 0.005')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='For SAC: temperature parameter determines the relative importance '
                             'of the entropy term against the reward (default: 0.2)')
    parser.add_argument("--automatic_entropy_tuning", action='store_true',
                        help='For SAC: To automatically learn the temperature hyperparameter. Default: False. '
                             'When arg is used, this is set to True.')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='For SAC: number of model updates per env step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        help='For SAC: Period of env steps to wait before updating the target networks (default: 1)')

    parser.add_argument("--policy_noise", default=0.2,
                        help='For TD3: Noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.5,
                        help='For TD3: Range to clip target policy noise')
    parser.add_argument("--policy_freq", default=2, type=int,
                        help='For TD3: Frequency of delayed policy updates')
    parser.add_argument("--expl_noise", default=0.1,
                        help='For TD3: Std.dev. of Gaussian exploration noise')

    parser.add_argument('--print_comments', dest='print_comments', action='store_true',
                        help='Print all comments while running')
    parser.add_argument('--not_print_comments', dest='print_comments', action='store_false')
    parser.set_defaults(print_comments=True)

    parser.add_argument('--save_results', dest='save_results', action='store_true',
                        help='Save results of the experiment')
    parser.add_argument('--not_save_results', dest='save_results', action='store_false')
    parser.set_defaults(save_results=True)
    parser.add_argument('--save_model', dest='save_model', action='store_true',
                        help='Save model and optimizer parameters')
    parser.add_argument('--not_save_model', dest='save_model', action='store_false')
    parser.set_defaults(save_model=True)
    parser.add_argument("--load_model", default="",
                        help='Model load file name. Does not load anything by default.')
    parser.add_argument("--save_model_period", default=250_000, type=int,
                        help='Save model and optimizer parameters after the set number of iterations')
    parser.add_argument("--outname", default="default_output_script.txt",
                        help='Output script name, to save in wandb, passed by HPC scripts.')

    parser.add_argument("--global_sparsity", default=0.0, type=float,
                        help='Set the global sparsity of the network. Specific layer sparsities are computed.'
                             'Default: 0.0 (will set a minimum sparsity if input_layer_sparsity is used).')
    parser.add_argument("--input_layer_sparsity", default=-1, type=float,
                        help='Set the sparsity of the input layer, value between 0 (fully dense) and '
                             '1 (no connections at all). Sparsity levels of other layers are computed. '
                             'Default: -1 (means input layer sparsity is computed with global_sparsity and the'
                             'sparsity_distribution_method).')
    parser.add_argument("--sparsity_distribution_method", default='uniform',
                        help="Which method to use to set sparsity levels for each layer. "
                             "Options: new, ER, uniform. Default: uniform.")
    parser.add_argument("--init_new_weights_method", default='zero',
                        help="Which method to use to reinitialize new weights. "
                             "Options: zero, unif, xavier. Default: zero.")
    parser.add_argument("--output_layer_sparse", action='store_true',
                        help='To use a sparse output layer, instead of dense. (Only for the actor, the '
                             'critic output layer is always kept dense, as it only has one output neuron.)'
                             'Default: False. When arg is used, this is set to True.')

    parser.add_argument("--num_hid_layers", default=2, type=int,
                        help='Number of hidden layers (of neurons)')
    parser.add_argument("--num_hid_neurons", default=256, type=int,
                        help='Number of neurons in each hidden layer')
    parser.add_argument("--ann_setZeta", default=0.05, type=float,
                        help='Proportion of connections to prune and regrow at topology change')
    parser.add_argument("--ann_ascTopologyChangePeriod", default=1e3, type=int,
                        help='Number of iterations to wait until doing a topology change')
    parser.add_argument("--ann_earlyStopTopologyChange", default=1e9, type=int,
                        help='The total policy iteration when to stop with adjusting the topology. '
                             'Default: 1e9 (kind of never, such that you can continue training). '
                             'Recommended when last training session, use: your_total - 5e4.')

    parser.add_argument("--activation", default="relu",
                        help='Activation function for hidden layers. Options: '
                             'relu, tanh, sigmoid, elu, fixedsrelu, srelu, leakyrelu, allrelu, swish, selu'
                             'symsqrt, symsqrt1, nonlex, lex. Default: relu.')
    parser.add_argument("--act_func_args", nargs="+", type=float,
                        help='Provide (float) arguments for the activation function in the right order, '
                             'with spaces in between.'
                             'For srelu the args are: threshold_right, slope_right, threshold_left, slope_left.'
                             'There is no default defined (None) for this argument. For srelu it is: 0.4 0.2 -0.4 0.2. '
                             'For allrelu you should provide the slope_left as a positive number. Recommended: 0.6')
    parser.add_argument("--act_func_per_neuron", action='store_true',
                        help='To learn activation function parameters per neuron instead of per layer. Default: False.'
                             'When arg is used, this is set to True.')

    parser.add_argument("--fake_features", default=0.0, type=float,
                        help='Proportion of fake features used for input of the NNs. Default: 0. '
                             'Must be in the interval [0,1). Cannot be 1.')
    parser.add_argument("--fake_noise_std", default=1.0, type=float,
                        help='Standard deviation of the fake features. Default: 1.0')
    parser.add_argument("--load_noise_distribution", default='',
                        help='Load a noise distribution from a file. '
                             'Give the file name, example: real_feats_distr_HalfCheetah.npy '
                             'Default: "" (no loading, will use Gaussian noise).')


def fill_initial_replay_buffer(replay_buffer, env, args):
    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0
    episode_start_time = datetime.datetime.now()
    for t in range(int(args.start_timesteps)):
        episode_timesteps += 1
        action = env.action_space.sample()
        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        if done:
            if args.print_comments:
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                      f"Reward: {episode_reward:.3f} Time: {datetime.datetime.now() - episode_start_time}")
            episode_start_time = datetime.datetime.now()
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


def refill_replay_buffer(replay_buffer, env, policy, args):
    print("Refilling replay buffer...")
    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0
    episode_start_time = datetime.datetime.now()
    max_action = float(env.action_space.high[0])
    for t in range(int(args.refill_timesteps)):
        episode_timesteps += 1
        if args.refill_mode == 'random':
            action = env.action_space.sample()
        elif args.refill_mode == 'current':
            action = policy.select_action(np.array(state)).clip(-max_action, max_action)
        else:
            raise ValueError('Invalid refill_mode. Options: random, current.')
        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        if done:
            if args.print_comments:
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                      f"Reward: {episode_reward:.3f} Time: {datetime.datetime.now() - episode_start_time}")
            episode_start_time = datetime.datetime.now()
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


def set_file_name(args):
    # File name configurations
    act_funcs_with_params = ['srelu', 'fixedsrelu', 'elu', 'leakyrelu', 'nonlex', 'lex', 'allrelu']
    act_func_type = "PerNeuron" if args.act_func_per_neuron and args.activation in ['srelu', 'lex'] else ""
    act_func_params = "_".join([str(n) for n in args.act_func_args]) if args.act_func_args is not None and \
                                                                        args.activation in act_funcs_with_params else ""
    # Checking for unusual input/output layers (default is in=sparse and out=dense)
    in_layer = 'in-lay-dense_' if (args.input_layer_sparsity == 0) and args.policy != 'TD3' else ''
    out_layer = 'out-lay-sparse_' if args.output_layer_sparse and args.policy != 'TD3' else ''
    # Sparsity info only for policies that use sparsity
    sparse_info = '' if args.policy in ['TD3', 'SAC'] \
        else f'sparsity{args.global_sparsity}_{args.sparsity_distribution_method}_'
    inlay_sparse = f'inlayspars{args.input_layer_sparsity}_' if \
        args.input_layer_sparsity > 0 and args.policy not in ['TD3', 'SAC'] else ''
    reinit_weights = f'reinit-{args.init_new_weights_method}_' if args.init_new_weights_method != 'zero' else ''
    zeta = '' if args.ann_setZeta == 0.05 else f'zeta{args.ann_setZeta}_'
    # Optimizer
    optim = f'{args.optimizer}_' if args.optimizer != 'adam' and 'ANF' in args.policy else ''
    lr = f'lr{args.lr:.0e}_' if args.lr != 1e-3 else ''
    empty_buf = f'emptybuf-refill{args.refill_mode}_' if args.empty_buffer_on_env_change else ''
    # For continual/changing envs
    adjust_period = f'-adjust{args.adjust_env_period}' if args.adjust_env_period != -1 else ''
    # For SAC types
    sac_type = f'-{args.sac_type}' if args.sac_type != 'Gaussian' else ''
    # Noise features
    fake_feats = f'fakefeats{args.fake_features}_' if args.fake_features != 0.0 else ''
    fake_noise_std = f'noise-std{args.fake_noise_std}_' if args.fake_noise_std != 1.0 else ''
    load_noise = f'noise-distr-{args.load_noise_distribution[:-4]}_' if args.load_noise_distribution != '' else ''

    file_name = f"{args.policy}{sac_type}_{args.env}{adjust_period}_{args.activation}{act_func_type}{act_func_params}_" \
                f"{sparse_info}{zeta}{reinit_weights}{inlay_sparse}{in_layer}{out_layer}hid-lay{args.num_hid_layers}_" \
                f"{empty_buf}{optim}{lr}{fake_feats}{fake_noise_std}{load_noise}seed{args.seed}"
    print("---------------------------------------")
    print(f"File name: {file_name}")
    print("---------------------------------------")
    return file_name


def set_policy_kwargs(state_dim, action_dim, max_action, args, device):
    kwargs = {
        "device": device,
        "args": args,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "num_hid_layers": args.num_hid_layers,   # comment out if using old version of code
        "num_hid_neurons": args.num_hid_neurons,
        "activation": args.activation,
        "act_func_args": (args.act_func_args, args.act_func_per_neuron),
        "optimizer": args.optimizer,
        "lr": args.lr,
        "fake_features": args.fake_features,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq

    # Initialize policy
    if args.policy == "TD3":
        policy = td3.TD3(**kwargs)
    elif args.policy == "Static-TD3":
        kwargs["global_sparsity"] = args.global_sparsity
        kwargs["sparsity_distribution_method"] = args.sparsity_distribution_method
        kwargs["input_layer_dense"] = (args.input_layer_sparsity == 0)
        kwargs["output_layer_dense"] = not args.output_layer_sparse
        policy = ss_td3.StaticSparseTD3(**kwargs)
    elif args.policy == "ANF-TD3":
        kwargs["global_sparsity"] = args.global_sparsity
        kwargs["sparsity_distribution_method"] = args.sparsity_distribution_method
        kwargs["init_new_weights_method"] = args.init_new_weights_method
        kwargs["input_layer_dense"] = (args.input_layer_sparsity == 0)
        kwargs["output_layer_dense"] = not args.output_layer_sparse
        kwargs["setZeta"] = args.ann_setZeta
        kwargs["ascTopologyChangePeriod"] = args.ann_ascTopologyChangePeriod
        kwargs["earlyStopTopologyChangeIteration"] = args.ann_earlyStopTopologyChange
        policy = anf_td3.ANF_TD3(**kwargs)
    else:
        raise ValueError("Unknown policy name. Run \'python main.py -h\' to see options.")
    return policy


def setup_sac_based_agent(args, env, device):
    if args.policy == "SAC":
        agent = sac.SAC(env.observation_space.shape[0], env.action_space, args, device)
    elif args.policy == "ANF-SAC":
        agent = anf_sac.ANF_SAC(env.observation_space.shape[0], env.action_space, args, device)
    elif args.policy == "Static-SAC":
        agent = ss_sac.Static_SAC(env.observation_space.shape[0], env.action_space, args, device)
    else:
        raise ValueError("Invalid algorithm name. Choose from SAC, ANF-SAC, Static-SAC")
    return agent


def initialize_environments(args):
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    if args.adjust_env_period <= 0:
        adjust_env_period = float('inf')  # don't change the environment over time
    else:
        adjust_env_period = args.adjust_env_period
    next_env_change = adjust_env_period
    env_num = 0
    return env, eval_env, next_env_change, adjust_env_period, env_num


def make_folders():
    if not os.path.exists("./output/results"):
        os.makedirs("./output/results")
    if not os.path.exists("./output/models"):
        os.makedirs("./output/models")
    if not os.path.exists("./output/replay_buffers"):
        os.makedirs("./output/replay_buffers")
    if not os.path.exists("./output/connectivity"):
        os.makedirs("./output/connectivity")


def add_fake_features(state, num_fake_features, device, noise_std=1.0, noise_generator=None):
    """ Adds a certain number of fake features to the state, by concatenating them at the end.
    The fake features are generated by a standard normal distribution N(0,1). """
    if noise_generator is None:
        fake_feats = torch.randn(state.shape[0], num_fake_features).to(device) * noise_std
        # state.shape[0] gives the batch size (could be 1 for evals & making new moves in env)
    else:
        fake_feats = noise_generator.sample(num_fake_features, batch_size=state.shape[0]).to(device)
    return torch.cat((state, fake_feats), dim=1)
    # concatenate along dimension 1 (the feature dim, not the batch dim (0) as we want to keep that constant)


def permute_features(state_with_fake, permutation):
    """ Permutes the features of the state. """
    if permutation is None:
        return state_with_fake
    else:
        return state_with_fake[:, permutation]
        # the colon (:) skips the batch dim, which is dim 0


def setup_noise_generator(distr_file=''):
    if distr_file == '':
        return None
    else:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "utils/noise_distributions", distr_file)
        generator = RealFeatureDistribution(path)
        return generator


def count_weights(model, args):
    """ Counts the number of active weights per input neuron in the input layer
    of the three main (non-target) networks in the agent: actor, critic1, critic2.
    Then computes the avg number of active weights (connections) for the real & fake features. """
    if "SAC" in args.policy:
        weights_actor = model.policy.linear1.weight.data.cpu().numpy()
        weights_q1 = model.critic.linear1.weight.data.cpu().numpy()
        weights_q2 = model.critic.linear4.weight.data.cpu().numpy()
    elif "TD3" in args.policy:
        weights_actor = model.actor.input_layer.weight.data.cpu().numpy()
        weights_q1 = model.critic.q1_input_layer.weight.data.cpu().numpy()
        weights_q2 = model.critic.q2_input_layer.weight.data.cpu().numpy()
    else:
        raise ValueError("Invalid algorithm name. Should contain SAC or TD3 as a substring.")

    env_dims = {'HalfCheetah-v3': (17, 6),
                'Walker2d-v3': (17, 6),
                'Hopper-v3': (11, 3),
                'Humanoid-v3': (376, 17)}
    state_dim, _ = env_dims[args.env]

    num_conn_per_in_neuron_actor = np.count_nonzero(weights_actor, axis=0)
    num_conn_per_in_neuron_q1 = np.count_nonzero(weights_q1, axis=0)
    num_conn_per_in_neuron_q2 = np.count_nonzero(weights_q2, axis=0)

    permutation = model.critic.permutation
    if permutation is None:
        # compute avg number of connections per input neuron
        avg_num_conn_actor_real = np.sum(num_conn_per_in_neuron_actor[:state_dim]) / state_dim
        avg_num_conn_q1_real = np.sum(num_conn_per_in_neuron_q1[:state_dim]) / state_dim
        avg_num_conn_q2_real = np.sum(num_conn_per_in_neuron_q2[:state_dim]) / state_dim
        if args.fake_features > 0:
            avg_num_conn_actor_fake = np.sum(num_conn_per_in_neuron_actor[state_dim:]) / (weights_actor.shape[1] - state_dim)
            avg_num_conn_q1_fake = np.sum(num_conn_per_in_neuron_q1[state_dim:]) / (weights_q1.shape[1] - state_dim)
            avg_num_conn_q2_fake = np.sum(num_conn_per_in_neuron_q2[state_dim:]) / (weights_q2.shape[1] - state_dim)
        else:
            avg_num_conn_actor_fake = 0
            avg_num_conn_q1_fake = 0
            avg_num_conn_q2_fake = 0
    else:
        permu = permutation.cpu().numpy()
        new_locs_real_feats = np.where(permu < state_dim)[0]
        new_locs_fake_feats = np.where(permu >= state_dim)[0]
        avg_num_conn_actor_real = np.sum(num_conn_per_in_neuron_actor[new_locs_real_feats]) / state_dim
        avg_num_conn_q1_real = np.sum(num_conn_per_in_neuron_q1[new_locs_real_feats]) / state_dim
        avg_num_conn_q2_real = np.sum(num_conn_per_in_neuron_q2[new_locs_real_feats]) / state_dim
        if args.fake_features > 0:
            avg_num_conn_actor_fake = np.sum(num_conn_per_in_neuron_actor[new_locs_fake_feats]) / (weights_actor.shape[1] - state_dim)
            avg_num_conn_q1_fake = np.sum(num_conn_per_in_neuron_q1[new_locs_fake_feats]) / (weights_q1.shape[1] - state_dim)
            avg_num_conn_q2_fake = np.sum(num_conn_per_in_neuron_q2[new_locs_fake_feats]) / (weights_q2.shape[1] - state_dim)
        else:
            avg_num_conn_actor_fake = 0
            avg_num_conn_q1_fake = 0
            avg_num_conn_q2_fake = 0

    return [avg_num_conn_actor_real, avg_num_conn_actor_fake,
            avg_num_conn_q1_real, avg_num_conn_q1_fake,
            avg_num_conn_q2_real, avg_num_conn_q2_fake]


