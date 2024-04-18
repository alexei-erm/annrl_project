import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor_network(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, device="cpu"):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        ).to(device)

    def forward(self, x):
        return self.network(x)


class Critic_network(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, device="cpu"):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        ).to(device)

    def forward(self, x):
        return self.network(x)



