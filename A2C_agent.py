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
        # initilialize workers, GD will run on worker 0 by default when calling train() method
        self.actors = {i: Actor_network(input_size, hidden_size, output_size_actor, device) for i in range(num_workers)}
        self.critics = {i: Critic_network(input_size, hidden_size, output_size_critic, device) for i in range(num_workers)}
        # initialize the original actor and critic networks, used once training is done for inference
        self.actor = Actor_network(input_size, hidden_size, output_size_actor, device)
        self.critic = Critic_network(input_size, hidden_size, output_size_critic, device)
        # hyperparameters
        self.device = device
        self.num_workers = num_workers
        self.eps = eps
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_steps = 0

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

    def train(self, batch, gamma_, lr_actor, lr_critic, device):
        """
        train one instance of actor and critic networks on a batch of experiences
        return: critic_loss, actor_loss
        """
        # training routine
        critic_loss = train_critic(self.critics, batch, gamma_, lr_critic, device)
        actor_loss = train_actor(self.critics, self.actors, batch, gamma_, lr_actor, device)
        
        # copy updated worker 0 into all other workers
        state_dict_worker0_actor = self.actors[0].state_dict()
        state_dict_worker0_critic = self.critics[0].state_dict()
        for worker_id in range(1,self.num_workers):
            self.actors[worker_id].load_state_dict(state_dict_worker0_actor)
            self.critics[worker_id].load_state_dict(state_dict_worker0_critic)
        
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

    def training_done(self):
        self.actor.load_state_dict(self.actors[0].state_dict())
        self.critic.load_state_dict(self.critics[0].state_dict())
