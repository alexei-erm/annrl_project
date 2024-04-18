import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import Actor_network, Critic_network
from training import train_critic, train_actor


class Agent:
    def __init__(self, input_size, hidden_size=64, \
                output_size_actor=2, output_size_critic=1, \
                eps=0.1, gamma=0.99, lr_actor=1e-5, lr_critic=1e-3, num_workers=1, device="cpu"):
        super().__init__()
        self.actor = Actor_network(input_size, hidden_size, output_size_actor, device)
        self.critic = Critic_network(input_size, hidden_size, output_size_critic, device)
        for param in self.actor.parameters():
            param.requires_grad = True

        for param in self.critic.parameters():
            param.requires_grad = True

        self.device = device
        self.num_workers = num_workers
        self.eps = eps
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_steps = 0

    def select_action(self, state, policy="greedy"):
        """
        make a specified worker select an action given the state and policy
        return: action
        """
        actor_output = self.actor.forward(state)
        if policy == "greedy":
            return torch.argmax(actor_output)
            # return torch.tensor(action)
        
        elif policy == "eps-greedy":
            # exploration (eps)
            if np.random.rand() < self.eps: 
                return torch.randint(0, 2, (1,)).to(self.device) # !!!!!  MAY BE CHANGED LATER ON (if action space changes)
            
            # exploitation (1-eps)
            else:    
                return torch.argmax(actor_output)
                # return torch.tensor(action)

    def train(self, experience, gamma_, lr_actor, lr_critic, device):
        """
        train one instance of actor and critic networks on a batch of experiences
        return: critic_loss, actor_loss
        """
        critic_loss = train_critic(self.critic, experience, gamma_, lr_critic, device)
        actor_loss = train_actor(self.critic, self.actor, experience, gamma_, lr_actor, device)
        return critic_loss, actor_loss


    def save(self, path):
        """
        save the agent's models
        """
        torch.save(self.actor, path + "actor.pth")
        torch.save(self.critic, path + "critic.pth")

    def load(self, path):
        """
        load the agent's models
        """
        self.actor = torch.load(path + "actor.pth")
        self.critic = torch.load(path + "critic.pth")
