import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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