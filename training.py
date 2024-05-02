import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym

from A2C_agent import Agent
from helpers import *



def train(agent, actor_optimizer, critic_optimizer, batch):

    n = len(batch)
    gamma_ = agent.gamma
    states, actions, log_probs, rewards, next_states, terminated = zip(*batch)
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
    # Get the log policy for taken action from actor
    # logits = agent.actor.forward(states)
    # probs = Categorical(logits=logits)
    # log_probs = probs.log_prob(actions)
    
    # Compute the n, (n-1), ...-step targets
    targets = []
    for t in range(n): # 0 to 5
        target = 0
        for i in range(t+1): # 0, 0 to 1, 0 to 2, 0 to 3, 0 to 4, 0 to 5
            target += (gamma_**i)*rewards[i]
        next_V_value = next_V_values[i].squeeze()
        target += next_V_value*(1-terminated[i])*gamma_**(i+1)
        targets.append(target)
    targets = torch.stack(targets)

    # compute the advantage
    advantage = targets.detach() - current_V_values.squeeze()
    
    # Gradient descent for the critic
    critic_loss = advantage.pow(2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Gradient descent for the actor
    actor_loss = - (log_probs * advantage.detach()).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), critic_loss.item()



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
    agent3 = {}
    all_critic_losses = {}
    all_actor_losses = {}
    all_episode_rewards = {}
    all_evaluation_reward_means = {}
    all_evaluation_reward_stds = {}
    all_evaluation_value_trajectories = {}

    # training loop over 3 seeds
    for i in range(len(seeds)):
        set_seed(seeds[i])

        # Initialize environment, agent and optimizers
        env = gym.make(environment)
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
        episode_rewards = []

        batch = []

        # reset flags
        reached_train_budget = False
        episode = 0
        done = False

        # Training loop
        while not reached_train_budget:
            reset_seed = np.random.randint(0, 1000000) # Random seed for resetting the environment, fixed sequence because of set_seed() call above
            state, _ = env.reset(seed=reset_seed)
            state = tensor(state).to(device)  # Convert state to a tensor
            episode_reward = 0

            # Run an episode
            while not done:
                action, log_probs = agent.select_action(state, mode="learning")
                next_state, reward, terminated, truncated, _ = env.step(np.array([action.detach().item()]))
                agent.num_steps += 1
                next_state = tensor(next_state).to(device)  # Convert next_state to a tensor
                done = terminated or truncated
                episode_reward += reward

                # apply stochastic mask on reward (if stochastic_rewards=True)
                mask = 1
                if stochastic_rewards and np.random.rand() < 0.9: mask = 0 # with probability 0.9 cancel out reward passed to learner
                reward = reward * mask

                # Add the experience to the batch
                experience = (state, action, log_probs, reward, next_state, terminated)
                batch.append(experience)

                # Train the agent when batch is full
                if len(batch) >= agent.n or done:
                    actor_loss, critic_loss = train(agent, actor_optimizer, critic_optimizer, batch)
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                    batch = []
                state = next_state

                # logging procedures
                # if agent.num_steps % 20000 == 0: 
                #     print(f"---- Proceeding to evaluate model {i} ... ----")
                #     mean_reward, std_reward, value_trajectories = agent.evaluate_agent(num_episodes=10)
                #     all_evaluation_reward_means[i].append(mean_reward)
                #     all_evaluation_reward_stds[i].append(std_reward)
                #     all_evaluation_value_trajectories[i].append(value_trajectories[0])
                #     print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}, total steps: {agent.num_steps}")
                #     print("----     Evaluation finished        ----")
                
                if agent.num_steps % 1000 == 0:
                    all_episode_rewards[i].append(episode_rewards[-1])
                    all_actor_losses[i].append(actor_losses[-1])
                    all_critic_losses[i].append(critic_losses[-1])

                if (agent.num_steps >= total_steps_budget): 
                    reached_train_budget = True
                    break

            done = False
            episode += 1
            episode_rewards.append(episode_reward)
            if episode % 100 == 0:
                print(f"-------- Episode {episode} ended with reward {episode_reward} for model {i} --------")
                print(f"Actor loss: {actor_losses[-1]:.4f}, Critic loss: {critic_losses[-1]:.4f}")
                print(f"Total steps taken during training: {agent.num_steps}")
            
        if reached_train_budget:
            print(f"Reached total training budget of {total_steps_budget} steps ----> Stopping training at episode {episode}")
        
        agent3[i] = agent # record the agent
        env.close()
    return agent3, all_critic_losses, all_actor_losses, all_episode_rewards, all_evaluation_reward_means, all_evaluation_reward_stds, all_evaluation_value_trajectories


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