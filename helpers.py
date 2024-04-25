import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def tensor(x): 
    """
    helper function to convert numpy arrays to tensors
    """
    return torch.from_numpy(x).float()

def mps_is_available():
    """
    a function analogous to `torch.cuda.is_available()` but for MPS
    """
    try:
        torch.ones(1).to('mps')
        return True
    except Exception:
        return False


def device_selection():
    """
    a function to select the device: mps -> cuda -> cpu
    """
    if mps_is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_episodes(env, agent, num_ep):
        """
        function runs episodes given the environment and agent for a given number 
        of episodes that the user wishes to run.
        """
        print(f'Starting the simulation... Num. Episodes: {num_ep}')
        
        for ep in range(num_ep):
            done = False
            state, _ = env.reset()
            episode_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward

                state = next_state
                done = terminated or truncated
            
            # store episode total reward.
            agent.record(episode_reward) # to implement in agent????

        print(f"Episode reward after taking random actions: {episode_reward}")


def get_stats(dict_of_lists):
    min_values = np.min([dict_of_lists[i] for i in dict_of_lists], axis=0)
    max_values = np.max([dict_of_lists[i] for i in dict_of_lists], axis=0)
    avg_values = np.mean([dict_of_lists[i] for i in dict_of_lists], axis=0)
    return min_values, max_values, avg_values


def plot_stats(min_values, max_values, avg_values, title, ylabel='Value', xlabel='Step'):
    plt.figure(figsize=(12, 6))
    plt.plot(avg_values, label='Average', color='purple')
    plt.plot(min_values, label='Min', color='violet')
    plt.plot(max_values, label='Max', color='indigo')
    plt.fill_between(range(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_smoothed_stats(min_values, max_values, avg_values, window_size, title, ylabel='Value', xlabel='Step'):
    plt.figure(figsize=(12, 6))
    steps = list(range(len(avg_values)))  # Create a list of steps

    # Convert lists to pandas Series for smoothing
    min_values_pd = pd.Series(min_values).rolling(window=window_size).mean()
    max_values_pd = pd.Series(max_values).rolling(window=window_size).mean()
    avg_values_pd = pd.Series(avg_values).rolling(window=window_size).mean()

    plt.plot(steps, avg_values_pd, label='Average', color='purple')
    plt.plot(steps, min_values_pd, label='Min', color='violet')
    plt.plot(steps, max_values_pd, label='Max', color='indigo')
    plt.fill_between(steps, min_values_pd, max_values_pd, color='lightblue', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

