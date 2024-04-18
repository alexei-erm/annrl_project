import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_critic(critic, experience, gamma_, lr=1e-4, device="cpu"):
    
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr)

    state, action, reward, next_state, terminated = experience
    state.requires_grad_(True)
    action = torch.tensor(action, dtype=torch.float, requires_grad=True).to(device)
    reward = torch.tensor(reward, dtype=torch.float, requires_grad=True).to(device)
    next_state.requires_grad_(True)
    terminated = torch.tensor(terminated, dtype=torch.float, requires_grad=True).to(device)

    # Get V values from critic network
    current_V_value = critic(state)
    next_V_value = critic(next_state)

    # Compute the target V values
    target = reward + (gamma_ * next_V_value * (1 - terminated))

    # Compute the critic loss
    critic_loss = F.mse_loss(current_V_value, target.detach())
    print(reward.requires_grad)
    print(terminated.requires_grad)
    print(current_V_value.requires_grad)
    print(next_V_value.requires_grad)

    # Gradient descent for the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss.item()


def train_actor(critic, actor, experience, gamma_, lr=1e-4, device="cpu"):
    
    actor_optimizer = torch.optim.Adam(actor.parameters())

    state, action, reward, next_state, terminated = experience
    state.requires_grad_(True)
    action = torch.tensor(action, dtype=torch.float, requires_grad=True).to(device)
    reward = torch.tensor(reward, dtype=torch.float, requires_grad=True).to(device)
    next_state.requires_grad_(True)
    terminated = torch.tensor(terminated, dtype=torch.float, requires_grad=True).to(device)

    # Get the current policy output FOR THE TAKEN ACTION
    current_policy_output = actor(state)[action]

    # Get the V values from critic network
    current_V_value = critic(state)
    next_V_value = critic(next_state)

    # Compute the target V values and advantage
    target = reward + (gamma_ * next_V_value * (1 - terminated))
    advantage = target.detach() - current_V_value

    # Compute the log probabilities of the actions
    log_prob = torch.log(current_policy_output)

    # Compute the actor loss
    actor_loss = log_prob * advantage
    # Gradient descent for the actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item()