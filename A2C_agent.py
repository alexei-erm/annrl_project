import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import Actor_network, Critic_network
from training import train_critic, train_actor

class Agent:
    def __init__(self, input_size, hidden_size=64, \
                output_size_actor=2, output_size_critic=1, \
                eps=0.1, gamma=0.99, lr=1e-4, K=1, device="cpu"):
        super().__init__()
        self.actors = {i: Actor_network(input_size, hidden_size, output_size_actor, device) for i in range(K)}
        self.critics = {i: Critic_network(input_size, hidden_size, output_size_critic, device) for i in range(K)}
        
        self.device = device
        self.num_workers = K
        self.eps = eps
        self.gamma = gamma
        self.lr = lr

    def select_action(self, state, worker_id, policy="greedy"):
        """
        make a specified worker select an action given the state and policy
        return: action
        """
        actor_output = self.actors[worker_id].forward(state)
        
        if policy == "greedy":
            return torch.argmax(actor_output).unsqueeze(0).to(self.device)
            # return torch.tensor(action)
        
        elif policy == "eps-greedy":
            # exploration (eps)
            if np.random.rand() < self.eps: 
                return torch.randint(0, 2, (1,)).to(self.device) # !!!!!  MAY BE CHANGED LATER ON (if action space changes)
            
            # exploitation (1-eps)
            else:    
                return torch.argmax(actor_output).unsqueeze(0).to(self.device)
                # return torch.tensor(action)

    def train_worker(self, batch, worker_id, gamma_, lr, device):
        """
        train one instance of actor and critic networks on a batch of experiences
        return: critic_loss, actor_loss
        """
        critic_loss = train_critic(self.critics[worker_id], batch, gamma_, lr, device)
        actor_loss = train_actor(self.critics[worker_id], self.actors[worker_id], batch, gamma_, lr, device)
        return critic_loss, actor_loss
    
    def train(self, batch, gamma_, lr, device):
        """
        train all instances of worker networks (actors and critics) on a batch of experiences
        return: worker_losses (dictionary)
        """
        for worker_id in self.actors.keys():
            # critic_loss, actor_loss = self.train_worker(batch, worker_id, gamma_, lr)
            worker_losses = {i: self.train_worker(batch, worker_id, gamma_, lr, device) for i in range(self.num_workers)}
        return worker_losses
