import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical



def train(agent, actor_optimizer, critic_optimizer, experience):

    state, action, reward, next_state, terminated = experience
    reward = torch.tensor(reward, dtype=torch.float).to(agent.device)
    terminated = torch.tensor(terminated, dtype=torch.float).to(agent.device)

    # Get the V values from critic network
    current_V_value = agent.critic(state)
    next_V_value = agent.critic(next_state)

    # Get the log policy for taken action from actor
    logits = agent.actor.forward(state)
    probs = Categorical(logits=logits)
    log_prob = probs.log_prob(action)

    # Compute the advantage
    advantage = reward + (1-terminated)*agent.gamma*next_V_value - current_V_value

    # Gradient descent for the critic
    critic_loss = advantage.pow(2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Gradient descent for the actor
    actor_loss = -(log_prob * advantage.detach())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


    return actor_loss.item(), critic_loss.item()

def train_k(agent, actor_optimizer, critic_optimizer, experiences):

    actor_loss = torch.tensor(0.0, requires_grad=True)
    critic_loss = torch.tensor(0.0, requires_grad=True)
    
    for experience in experiences:
        state, action, reward, next_state, terminated = experience
        reward = torch.tensor(reward, dtype=torch.float).to(agent.device)
        terminated = torch.tensor(terminated, dtype=torch.float).to(agent.device)
    
        # Get the V values from critic network
        current_V_value = agent.critic(state)
        next_V_value = agent.critic(next_state)
    
        # Get the log policy for taken action from actor
        logits = agent.actor.forward(state)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action)
    
        # Compute the advantage
        advantage = reward + (1-terminated)*agent.gamma*next_V_value - current_V_value
        critic_loss += advantage.pow(2).mean()
        actor_loss += -(log_prob * advantage.detach())
        

    # Gradient descent for the critic    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Gradient descent for the actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


    return actor_loss.item(), critic_loss.item()
