import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def train(agent, actor_optimizer, critic_optimizer, batch):

    n = len(batch)
    gamma_ = agent.gamma
    states, actions, rewards, next_states, terminated = zip(*batch)
    # Convert lists to PyTorch tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(agent.device)
    next_states = torch.stack(next_states)
    terminated = torch.tensor(terminated, dtype=torch.float32).to(agent.device)

    # Get the V values from critic network
    current_V_values = agent.critic(states)
    next_V_values = agent.critic(next_states)
    # Get the log policy for taken action from actor
    logits = agent.actor.forward(states)
    probs = Categorical(logits=logits)
    log_probs = probs.log_prob(actions)
    
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