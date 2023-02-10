import torch
import gym
import numpy as np
import argparse
import time
from utils import utils


def show_policy(config):
    policy = load_policy(config)
    env = gym.make(config['env'])
    state, done = env.reset(), False
    while not done:
        env.render()
        action = policy.select_action(np.array(state), evaluate=True)
        state, reward, done, _ = env.step(action)


def show_random_actions(config):
    env = gym.make(config['env'])
    env.reset()
    done, step_num = False, 0
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)

        time.sleep(0.01)
        print(f"step {step_num}")
        step_num += 1
    env.close()


def format_optimizer_arg(config):
    if config['policy'] in ['TD3', 'SAC', 'Static-TD3', 'Static-SAC']:
        optimizer = ''  # for dense/static it's all the same (adam=maskadam), we use default (Adam)
    elif config['policy'] in ['ANF-TD3', 'ANF-SAC']:
        optimizer = 'maskadam_'
    else:
        raise ValueError('unknown policy')
    return optimizer


def format_sparsity(exp):
    if exp['policy'] in ['TD3', 'SAC']:
        return ''
    else:
        return f'sparsity0.0_uniform_inlayspars0.8_'


def format_env(config):
    if config['noise_fraction'] == 0.95:
        return f'{config["env"]}-adjust1000000'
    else:
        return config['env']


def load_policy(config):
    env = gym.make(config['env'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    parser = argparse.ArgumentParser()
    utils.add_arguments(parser)
    seed = 3101
    args = parser.parse_args(["--policy", config['policy'],
                              "--env", config['env'],
                              "--fake_features", str(config.get('noise_fraction', 0)),
                              "--fake_noise_std", str(config.get('noise_amplitude', 1)),
                              "--global_sparsity", str(0),
                              "--sparsity_distribution_method", 'uniform',
                              "--input_layer_sparsity", str(0.8),
                              "--seed", str(int(seed)),
                              ])
    utils.print_all_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'TD3' in config['policy']:
        policy = utils.set_policy_kwargs(state_dim, action_dim, max_action, args, device)
    elif 'SAC' in config['policy']:
        policy = utils.setup_sac_based_agent(args, env, device)
    else:
        raise ValueError('Unknown policy name')

    sparsity_info = format_sparsity(config)
    optim = format_optimizer_arg(config)
    noisefeats = f'fakefeats{config["noise_fraction"]}_' if config["noise_fraction"] != 0 else ''
    noiseamp = config.get('noise_amplitude', 1)
    noise_ampl = f'noise-std{noiseamp}.0_' if noiseamp != 1 else ''
    folder = './utils/pretrained_models/'
    file_name = f"{config['policy']}_{format_env(config)}_relu_" \
                f"{sparsity_info}hid-lay2_{optim}{noisefeats}{noise_ampl}seed{seed}_best"
    file_path = f'{folder}{file_name}'
    policy.load(file_path)
    return policy


if __name__ == '__main__':
    # PRESS >>>TAB<<< TO SWITCH CAMERA TO TRACK THE AGENT :)
    # script to see an agent in action!
    # run this script from the terminal with:
    # python view_mujoco.py
    
    # possible environments: HalfCheetah-v3, Hopper-v3, Walker2d-v3, Humanoid-v3
    # possible policies: ANF-SAC, ANF-TD3, SAC, TD3
    # possible noise_fractions: 0, 0.8, 0.9, 0.95, 0.98, 0.99
    # possible noise_amplitudes: 1 (for all) or 2, 4, 8, 16 (for noise_fraction=0.9, env=HalfCheetah-v3)
    config = {
        'env': 'HalfCheetah-v3',
        'policy': 'ANF-SAC',
        'noise_fraction': 0.9,
        'noise_amplitude': 1,
    }
    show_policy(config)

    # if you want to run this file with other config settings,
    # first download the pretrained models from: 
    # https://www.dropbox.com/s/qr1l7bscnnd8non/pretrained_models.zip?dl=0 
    # extract the zip, and put the model files in the folder './utils/pretrained_models/'
