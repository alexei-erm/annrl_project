import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym

from models import Actor_network, Critic_network


class Agent:
    def __init__(self, input_size, hidden_size=64, \
                output_size_actor=2, output_size_critic=1, \
                gamma=0.99, lr_actor=1e-5, lr_critic=1e-3, device="cpu"):
        super().__init__()
        self.actor = Actor_network(input_size, hidden_size, output_size_actor, device)
        self.critic = Critic_network(input_size, hidden_size, output_size_critic, device)
        
        self.device = device
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_steps = 0

    def select_action(self, state, mode="evaluation"):
        """
        select an action given the state and policy, in evaluation or learning mode
        return: action
        """
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if mode == "learning":
            action = probs.sample()
        else:
            action = torch.argmax(probs.probs)
        return action


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

    def evaluate_agent(self, num_episodes=10):
        """
        evaluate the agent's performance
        return: mean reward, std reward, value trajectories (list of lists)
        """
        rewards = []
        value_trajectories = []
        test_env = gym.make("CartPole-v1")
        for _ in range(num_episodes):
            value_trajectory = []
            reset_seed = np.random.randint(0, 1000000)
            state, _ = test_env.reset(seed=reset_seed)
            state = torch.from_numpy(state).float().to(self.device)  # Convert state to a tensor
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action(state, mode="evaluation")
                value = self.critic(state)
                value_trajectory.append(value)
                next_state, reward, terminated, truncated, _ = test_env.step(action.item())
                next_state = torch.from_numpy(next_state).float().to(self.device)  # Convert next_state to a tensor
                episode_reward += reward
                state = next_state
                done = terminated or truncated
            value_trajectories.append(value_trajectory)
            rewards.append(episode_reward)
        return np.mean(rewards), np.std(rewards), value_trajectories
