
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import numpy as np

from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

import torch 
from torch import nn

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import os
from PIL import Image

class SkipFrame(Wrapper):

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0 
        done = False
        for _ in range(self.skip):

            next_state, reward, done, trunc, info = self.env.step(action)

            total_reward += reward
            if done:
                break
        
        return next_state, total_reward, done, trunc, info
    
def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env

env = gym_super_mario_bros.make('SuperMarioBros-v2',apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)  

num_episodes = 1
returns = {}
state_action_counts = {}
gamma = 0.99  
epsilon = 0.1  


for episode in tqdm.trange(num_episodes):
    state, _ = env.reset()
    state = preprocess_observation(state)
    done = False
    episode_rewards = []
    states_actions = []  

    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_observation(next_state) 
        
        
        states_actions.append((state, action, reward))
        state = next_state  
        env.render()
    
   
    total_return = 0
    for (state, action, reward) in reversed(states_actions):
        total_return = reward + gamma * total_return

        state_tuple = state_to_key(state)
        
        
        if (state_tuple, action) not in state_action_counts:
            state_action_counts[(state_tuple, action)] = 0
        if (state_tuple, action) not in returns:
            returns[(state_tuple, action)] = 0

        state_action_counts[(state_tuple, action)] += 1
        returns[(state_tuple, action)] += total_return

save_state_action_data(returns, state_action_counts)
env.close()

