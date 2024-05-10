import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym
import time

from A2C_agent import Agent
from helpers import *



def train(agent, actor_optimizer, critic_optimizer, batch):

    gamma_ = agent.gamma
    critic_loss = []
    actor_loss = []

    for i in range(agent.k):
        states, actions, log_probs, rewards, next_states, terminated = zip(*batch[i])
        # Convert lists to PyTorch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(agent.device)
        next_states = torch.stack(next_states)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(agent.device)

        # Get the V values from critic network
        current_V_values = agent.critic(states)
        next_V_values = agent.critic(next_states)
        
        # Compute the n, (n-1), ...-step targets
        targets = []
        for t in range(agent.n): # 0 to 5
            target = 0
            for i in range(t+1): # 0, 0 to 1, 0 to 2, 0 to 3, 0 to 4, 0 to 5
                target += (gamma_**i)*rewards[i]
            next_V_value = next_V_values[i].squeeze()
            target += next_V_value*(1-terminated[i])*gamma_**(i+1)
            targets.append(target)
        targets = torch.stack(targets)
        # compute the advantage for worker 
        advantage = targets.detach() - current_V_values.squeeze()
        # compute and store losses for worker
        critic_loss.append(advantage.pow(2).mean())
        actor_loss.append(- (log_probs * advantage.detach()).mean())

    critic_loss = torch.stack(critic_loss)
    actor_loss = torch.stack(actor_loss)

    # Gradient descent for the critic
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Gradient descent for the actor
    actor_loss = actor_loss.mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), critic_loss.item()


def reset_env(env, device = 'cpu'):
    reset_seed = np.random.randint(0, 1000000) # Random seed for resetting the environment, fixed sequence because of set_seed() call above
    state, _ = env.reset(seed=reset_seed)
    state = tensor(state).to(device)  # Convert state to a tensor
    return state



def training_loop(k, n, continuous, seeds, lr_actor=1e-5, lr_critic=1e-3, total_steps_budget=500000, stochastic_rewards=False, device="cpu"):
    
    if continuous: 
        environment = "InvertedPendulum-v4"
        output_size_actor = 1
    else: 
        environment = "CartPole-v1"
        output_size_actor = 2

    
    # default hyoerparameters
    gamma_ = 0.99

    # neural network structure
    input_size = 4
    hidden_size = 64
    output_size_critic = 1

    # initializing logging dicts
    logging_agent = {}
    all_critic_losses = {}
    all_actor_losses = {}
    all_episode_rewards = {}
    all_evaluation_reward_means = {}
    all_evaluation_reward_stds = {}
    all_evaluation_value_trajectories = {}

    # training loop over 3 seeds
    for i in range(len(seeds)):
        start_time = time.time()
        set_seed(seeds[i])

        # Initialize environment, agent and optimizers
        envs = [gym.make(environment) for _ in range(k)]
        agent = Agent(continuous, k, n, input_size, hidden_size, \
                        output_size_actor, output_size_critic, \
                        gamma_, lr_actor, lr_critic, \
                        device=device)
        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr_actor)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr_critic)

        # Initialize recording lists and assign them to dictionary key
        all_critic_losses[i] = []
        all_actor_losses[i] = []
        all_episode_rewards[i] = []
        all_evaluation_reward_means[i] = []
        all_evaluation_reward_stds[i] = []
        all_evaluation_value_trajectories[i] = []

        critic_losses = []
        actor_losses = []
        episode_rewards = [[] for _ in range(k)]

        batch = [[] for _ in range(k)] # batches of experiences for each worker

        # reset flags
        reached_train_budget = False
        episode = 0
        dones = [False for _ in range(k)]

        # initial setting
        states = []
        k_rewards = []
        for env in envs:
            states.append(reset_env(env, device))
            k_rewards.append(0)

        # Training loop
        while not reached_train_budget:            
            for env_idx, env in enumerate(envs):
                if dones[env_idx]:            
                    states[env_idx] = reset_env(envs[env_idx], device)
                    # episode += 1
                    episode_rewards[env_idx].append(k_rewards[env_idx])
                    # if episode % 100 == 0:
                    #     print(f"-------- Episode {episode} ended with reward {k_rewards[env_idx]:.2f} for model {i} --------")
                    #     print(f"Actor loss: {actor_losses[-1]:.4f}, Critic loss: {critic_losses[-1]:.4f}")
                    #     print(f"Total steps taken during training: {agent.num_steps}")
                    k_rewards[env_idx] = 0
                    dones[env_idx] = False

                action, log_probs = agent.select_action(states[env_idx], mode="learning")
                next_state, reward, terminated, truncated, _ = envs[env_idx].step(action.detach()) if continuous else envs[env_idx].step(action.detach().item())
                
                next_state = tensor(next_state).to(device)
                dones[env_idx] = terminated or truncated
                k_rewards[env_idx] += reward
                agent.num_steps += 1 # If k > 1, num_steps would be a total steps of K-workers.

                # apply stochastic mask on reward (if stochastic_rewards=True)
                reward = 0 if stochastic_rewards and np.random.rand() < 0.9 else reward # with probability 0.9 cancel out reward passed to learner                

                # Add the experience to the batch
                experience = (states[env_idx], action, log_probs, reward, next_state, terminated)
                batch[env_idx].append(experience)
                states[env_idx] = next_state
            
            # Train the agent when batches are full
            if all(len(batch[i]) == agent.n for i in range(k)): 
                actor_loss, critic_loss = train(agent, actor_optimizer, critic_optimizer, batch)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                batch = [[] for _ in range(k)] # empty batches
            
            # logging procedures
            if agent.num_steps >= 20000 * (len(all_evaluation_reward_means[i])+1): 
                print(f"---- Proceeding to evaluate model {i} ... ----")
                mean_reward, std_reward, value_trajectories = agent.evaluate_agent(num_episodes=10)
                all_evaluation_reward_means[i].append(mean_reward)
                all_evaluation_reward_stds[i].append(std_reward)
                all_evaluation_value_trajectories[i].append(value_trajectories[0])
                print(f" Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, total steps: {agent.num_steps}")
                print("----     Evaluation finished        ----")
            
            if agent.num_steps >= 1000 * (len(all_episode_rewards[i])+1):
                average_reward = sum(rewards[-1] for rewards in episode_rewards) / len(episode_rewards)
                all_episode_rewards[i].append(average_reward)
                all_actor_losses[i].append(actor_losses[-1])
                all_critic_losses[i].append(critic_losses[-1])

            if (agent.num_steps >= total_steps_budget): 
                reached_train_budget = True
                break
                 
        if reached_train_budget:
            print(f"Reached total training budget of {total_steps_budget} steps ----> Stopping training at episode {episode}")
        else:
            print('Training loop was terminated by unexpected reason.')
        
        logging_agent[i] = agent # record the agent
        for env in envs:
            env.close()
        end_time = time.time()
        print(f'Experiment took {end_time-start_time:.2f}s in total.')
    return logging_agent, all_critic_losses, all_actor_losses, all_episode_rewards, all_evaluation_reward_means, all_evaluation_reward_stds, all_evaluation_value_trajectories


# old train function for vanilla A2C

# def train(agent, actor_optimizer, critic_optimizer, experience):

#     state, action, reward, next_state, terminated = experience
#     reward = torch.tensor(reward, dtype=torch.float).to(agent.device)
#     terminated = torch.tensor(terminated, dtype=torch.float).to(agent.device)

#     # Get the V values from critic network
#     current_V_value = agent.critic(state)
#     next_V_value = agent.critic(next_state)

#     # Get the log policy for taken action from actor
#     logits = agent.actor.forward(state)
#     probs = Categorical(logits=logits)
#     log_prob = probs.log_prob(action)

#     # Compute the advantage
#     advantage = reward + (1-terminated)*agent.gamma*next_V_value.detach() - current_V_value

#     # Gradient descent for the critic
#     critic_loss = advantage.pow(2).mean()
#     critic_optimizer.zero_grad()
#     critic_loss.backward()
#     critic_optimizer.step()

#     # Gradient descent for the actor
#     actor_loss = -(log_prob * advantage.detach())
#     actor_optimizer.zero_grad()
#     actor_loss.backward()
#     actor_optimizer.step()


#     return actor_loss.item(), critic_loss.item()