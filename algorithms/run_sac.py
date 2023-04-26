import numpy as np
import wandb
from utils.replay_memory_sac import ReplayMemory, fill_initial_replay_memory, refill_replay_buffer
from utils import utils


def run(args, file_name, device):
    env, eval_env, next_env_change, adjust_env_period, env_num = utils.initialize_environments(args)
    agent = utils.setup_sac_based_agent(args, env, device)
    memory = ReplayMemory(args.buffer_size, args.seed)
    avg_return0 = utils.eval_policy(agent, eval_env, args.seed, args.print_comments, args.eval_episodes)
    evaluations = [avg_return0]
    wandb.log({'eval_return': avg_return0}, step=0)

    num_connections0 = utils.count_weights(agent, args)
    connections = [num_connections0]
    wandb.log({'num_connections': num_connections0}, step=0)
    wandb.watch((agent.policy, agent.critic), log="all", log_freq=5000)
    if args.save_model:
        agent.save(f"./output/models/{file_name}_iter_0")

    fill_initial_replay_memory(memory, env, args)

    updates = 0
    state, done = env.reset(), False
    episode_reward, episode_steps, episode_num = 0, 0, 0
    max_eval_return = float('-inf')
    loss_info_dict = {}

    # Training Loop
    print(f"\nNow the training starts")
    for t in range(int(args.max_timesteps)):
        action = agent.select_action(state)  # Sample action from policy
        next_state, reward, done, _ = env.step(action)  # Perform action
        episode_steps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # see https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py#L304
        not_done = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(state, action, reward, next_state, not_done)  # Append transition to memory
        state = next_state

        # Number of updates per step in environment
        for i in range(args.updates_per_step):
            # Train the agent
            loss_info_dict = agent.update_parameters(memory, args.batch_size, updates)
            updates += 1
        # wandb.log(loss_info_dict, step=t+1)

        # Evaluate the policy
        if (t + 1) % args.eval_freq == 0:
            avg_return = utils.eval_policy(agent, eval_env, args.seed, args.print_comments, args.eval_episodes)
            wandb.log({'eval_return': avg_return}, step=t+1)
            num_connections = utils.count_weights(agent, args)
            # wandb.log({'num_connections': num_connections}, step=t+1)
            wandb.log({'actor_real_connections': num_connections[0]}, step=t+1)
            wandb.log({'actor_fake_connections': num_connections[1]}, step=t+1)
            if args.save_results:
                evaluations.append(avg_return)
                np.save(f"./output/results/{file_name}", evaluations)
                connections.append(num_connections)
                np.save(f"./output/connectivity/{file_name}", connections)
            if t > 0.8 * int(args.max_timesteps) and avg_return > max_eval_return:
                max_eval_return = avg_return
                if args.save_model:
                    agent.save(f"./output/models/{file_name}_best")

        if done:
            if args.print_comments:
                print(f"Total T: {t+1} Episode Num: {episode_num} "
                      f"Episode T: {episode_steps}, Reward: {round(episode_reward, 2)}")
            if t > next_env_change:
                agent.set_new_permutation()
                next_env_change += adjust_env_period
                if args.empty_buffer_on_env_change:
                    memory.empty_buffer()
                    refill_replay_buffer(memory, env, agent, args)
            # Reset environment
            state, done = env.reset(), False
            episode_reward, episode_steps = 0, 0
            episode_num += 1

        # Save current policy
        if args.save_model and (t + 1) % args.save_model_period == 0:
            agent.save(f"./output/models/{file_name}_iter_{t + 1}")

        # Tracking the sparsity
        if args.policy in ['ANF-SAC', 'Static-SAC'] and t % 7_100 == 0:
            wandb.log(agent.print_sparsity(), step=t)

    wandb.log({'max_eval_return': max_eval_return})
    print(f"Maximum evaluation return value was {max_eval_return} (only measured after 80% of training steps onwards)")
