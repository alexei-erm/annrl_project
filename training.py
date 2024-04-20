import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(agent, experience):
    
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), agent.lr_critic)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), agent.lr_actor)

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

    # Compute the target V values and advantage
    # with torch.no_grad():
    #     target = reward + agent.gamma * next_V_value * (1 - terminated)
    target = reward + agent.gamma * next_V_value.detach() * (1 - terminated)
    advantage = target - current_V_value
    # Compute the log probability of the action
    log_prob = torch.log(taken_action_prob)

    # Compute the losses
    actor_loss = -log_prob * advantage
    critic_loss = 0.5 * advantage.pow(2)

    # Gradient descent for the actor
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()

    # Gradient descent for the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()

