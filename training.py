import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_critic(critics, batches, gamma_, lr=1e-4, device="cpu"):
    """
    trains the critic network on a batch of experiences, compatible with K workers 
    """
    # by default run gradient descent on the first agent
    critic_optimizer = torch.optim.Adam(critics[0].parameters(), lr)
    critic_loss = 0

    for worker_id, batch in batches.items():
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        current_V_values = critics[worker_id](states).squeeze(1)
        next_V_values = critics[worker_id](next_states).squeeze(1)

        with torch.no_grad():
            target = rewards + (gamma_ * next_V_values * (1 - dones))
        critic_loss += F.mse_loss(current_V_values, target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return critic_loss.item()



def train_actor(critics, actors, batches, gamma_, lr=1e-4, device="cpu"):
    """
    trains the actor network on a batch of experiences, compatible with K workers
    """
    # by default run gradient descent on the first agent
    actor_optimizer = torch.optim.Adam(actors[0].parameters(), lr)
    actor_loss = 0

    for worker_id, batch in batches.items():
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        current_V_values = critics[worker_id](states).squeeze(1)  # doesn't matter which worker_id is used here to infer?
        next_V_values = critics[worker_id](next_states).squeeze(1)

        current_policy = actors[worker_id](states)
        log_probs = torch.log(current_policy)
        taken_log_probs = log_probs.gather(1, actions)

        with torch.no_grad():
            target = rewards + (gamma_ * next_V_values * (1 - dones))
            advantage = target - current_V_values
        actor_loss += taken_log_probs * advantage

    actor_loss = actor_loss.mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item()