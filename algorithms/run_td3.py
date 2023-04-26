import numpy as np
import datetime
import wandb
from utils import utils


def run(args, file_name, device):
    env, eval_env, next_env_change, adjust_env_period, env_num = utils.initialize_environments(args)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = utils.set_policy_kwargs(state_dim, action_dim, max_action, args, device)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

    if args.load_model != "":  # Loading previously trained model
        agent.load(f"./output/models/{args.load_model}")
        current_iter = agent.total_it
        num_eval_to_keep = int(current_iter / args.eval_freq) + 1
        evaluations = list(np.load(f"./output/results/{file_name}.npy"))[:num_eval_to_keep]
        # replay_buffer.load_buffer(f"./output/replay_buffers/{file_name}")
    else:  # No model loaded, training from scratch
        if args.save_model:  # Save untrained policy
            agent.save(f"./output/models/{file_name}_iter0")
        # Evaluate untrained policy
        avg_return0 = utils.eval_policy(agent, eval_env, args.seed, args.print_comments, args.eval_episodes)
        evaluations = [avg_return0]
        wandb.log({'eval_return': avg_return0}, step=0)

        num_connections0 = utils.count_weights(agent, args)
        connections = [num_connections0]
        wandb.log({'num_connections': num_connections0}, step=0)
        if args.print_comments:
            print(f"\nFirst running {args.start_timesteps} steps with random policy to fill ReplayBuffer")
        utils.fill_initial_replay_buffer(replay_buffer, env, args)

    state, done = env.reset(), False
    episode_reward, episode_steps, episode_num = 0, 0, 0
    max_eval_return = float('-inf')
    episode_start_time = datetime.datetime.now()
    wandb.watch((agent.actor, agent.critic), log="all", log_freq=5000)

    print(f"\nNow the training starts")
    for t in range(int(args.max_timesteps)):
        # Select action according to policy, then add some noise
        action = (agent.select_action(np.array(state))
                  + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                  ).clip(-max_action, max_action)
        # Perform action
        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        episode_reward += reward

        done_bool = float(done) if episode_steps < env._max_episode_steps else 0
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        # Train the agent
        agent.train(replay_buffer, args.batch_size)

        # Evaluate the agent
        if (t + 1) % args.eval_freq == 0:
            avg_return = utils.eval_policy(agent, eval_env, args.seed, args.print_comments, args.eval_episodes)
            wandb.log({'eval_return': avg_return}, step=agent.total_it)
            num_connections = utils.count_weights(agent, args)
            # wandb.log({'num_connections': num_connections}, step=agent.total_it)
            wandb.log({'actor_real_connections': num_connections[0]}, step=agent.total_it)
            wandb.log({'actor_fake_connections': num_connections[1]}, step=agent.total_it)
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
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_steps} "
                      f"Reward: {episode_reward:.3f} Time: {datetime.datetime.now() - episode_start_time}")
            if t > next_env_change:
                agent.set_new_permutation()
                next_env_change += adjust_env_period
                if args.empty_buffer_on_env_change:
                    replay_buffer.empty_buffer()
                    utils.refill_replay_buffer(replay_buffer, env, agent, args)
            # Reset environment
            state, done = env.reset(), False
            episode_reward, episode_steps = 0, 0
            episode_num += 1
            episode_start_time = datetime.datetime.now()

        # Save current policy
        if args.save_model and (t + 1) % args.save_model_period == 0:
            agent.save(f"./output/models/{file_name}_iter{agent.total_it}")
            # replay_buffer.save_buffer(f"./output/replay_buffers/{file_name}")

        # Tracking the sparsity
        if args.policy in ['ANF-TD3', 'Static-TD3'] and t % 7_100 == 0:
            wandb.log(agent.print_sparsity(), step=t)

    wandb.log({'max_eval_return': max_eval_return})
    print(f"Maximum evaluation return value was {max_eval_return} (only measured after 80% of training steps onwards)")
