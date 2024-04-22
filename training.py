import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(agent, actor_optimizer, critic_optimizer, experience):

    state, action, reward, next_state, terminated = experience
    reward = torch.tensor(reward, dtype=torch.float).to(agent.device)
    terminated = torch.tensor(terminated, dtype=torch.float).to(agent.device)
    
    # Ensure that action is a 1D tensor
    if action.dim() == 0:
        action = action.unsqueeze(0)

    # Get the current policy output 
    current_policy_output = agent.actor(state).unsqueeze(0)  # Ensure that current_policy_output is a 2D tensor
    taken_action_prob = current_policy_output.gather(1, action.unsqueeze(-1))

    # Get the V values from critic network
    current_V_value = agent.critic(state)
    next_V_value = agent.critic(next_state)

    # Compute the target and advantage
    # target = reward + agent.gamma * next_V_value * (1 - terminated)
    # advantage = target - current_V_value
    advantage = reward + (1-terminated)*agent.gamma*agent.critic(next_state) - agent.critic(state)
    # Compute the log probability of the action
    log_prob = torch.log(taken_action_prob)

    # Gradient descent for the critic
    critic_loss = 0.5*advantage.pow(2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Gradient descent for the actor
    actor_loss = -(log_prob * advantage.detach())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


    return actor_loss.item(), critic_loss.item()

