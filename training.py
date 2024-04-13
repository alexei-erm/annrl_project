import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_critic(critic, batch, gamma_, lr=1e-4, device="cpu"):
    states, actions, rewards, next_states, dones = zip(*batch)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr)

    states = torch.stack(states).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)

    # Get V values from critic network
    current_V_values = critic(states).squeeze(1)
    next_V_values = critic(next_states).squeeze(1)

    # Compute the target V values (no_grad !!!!)
    with torch.no_grad():
        # print("rewards shape:", rewards.shape, "nextVvalues shape:",next_V_values.shape, "dones shape:",dones.shape)
        target = rewards + (gamma_ * next_V_values * (1 - dones))

    # Compute the critic loss
    # print("current_V_values shape:", current_V_values.shape, "target shape:", target.shape)
    critic_loss = F.mse_loss(current_V_values, target)

    # Gradient descent for the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss.item()


def train_actor(critic, actor, batch, gamma_, lr=1e-4, device="cpu"):
    states, actions, rewards, next_states, dones = zip(*batch)
    actor_optimizer = torch.optim.Adam(actor.parameters())

    actions = torch.stack(actions).to(device)
    states = torch.stack(states).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)

    # Get the current policy
    current_policy = actor(states)

    # Get the V values from critic network
    current_V_values = critic(states)
    next_V_values = critic(next_states)

    # Compute the target V values and advantage (no_grad !!!!)
    with torch.no_grad():
        target = rewards + (gamma_ * next_V_values * (1 - dones))
        advantage = target - current_V_values

    # Compute the log probabilities of the actions
    log_probs = torch.log(current_policy)

    # Gather only the log probabilities of the taken actions
    taken_log_probs = log_probs.gather(1, actions)

    # Compute the actor loss
    actor_loss = -(taken_log_probs * advantage).mean()

    # Gradient descent for the actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item()