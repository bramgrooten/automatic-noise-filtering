import random
import numpy as np
import datetime


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def empty_buffer(self):
        self.buffer = []
        self.position = 0


def fill_initial_replay_memory(replay_buffer, env, args):
    print("Filling initial replay memory...")
    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0
    episode_start_time = datetime.datetime.now()
    for t in range(int(args.start_timesteps)):
        episode_timesteps += 1
        action = env.action_space.sample()
        # Perform action
        next_state, reward, done, _ = env.step(action)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # see https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py#L304
        not_done = 1 if episode_timesteps == env._max_episode_steps else float(not done)
        replay_buffer.push(state, action, reward, next_state, not_done)  # Append transition to memory

        state = next_state
        episode_reward += reward
        if done:
            if args.print_comments:
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
                      f"Reward: {episode_reward:.3f} Time: {datetime.datetime.now() - episode_start_time}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_start_time = datetime.datetime.now()


def refill_replay_buffer(replay_buffer, env, policy, args):
    print("Refilling replay buffer...")
    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num = 0, 0, 0
    episode_start_time = datetime.datetime.now()

    for t in range(int(args.refill_timesteps)):
        episode_timesteps += 1
        if args.refill_mode == 'random':
            action = env.action_space.sample()
        elif args.refill_mode == 'current':
            action = policy.select_action(state)
        else:
            raise ValueError('Invalid refill_mode. Options: random, current.')
        # Perform action
        next_state, reward, done, _ = env.step(action)

        not_done = 1 if episode_timesteps == env._max_episode_steps else float(not done)
        replay_buffer.push(state, action, reward, next_state, not_done)  # Append transition to memory

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

