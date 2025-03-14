import gym.wrappers
import gym.wrappers
import gym.wrappers.gray_scale_observation
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import gym
from gym.wrappers import GrayScaleObservation, FrameStack

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import cv2
import numpy as np
from collections import deque
import tqdm
import matplotlib.pyplot as plt

class MonteCarloAgent():

    def __init__(self,):
        pass



def preprocess_observation(observation):

    if observation.shape[0] == 3:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(observation, (84, 84))
 
    return gray / 255.0


def state_to_key(state):
   
    return tuple(state.flatten())


def choose_action(state):
    state_key = state_to_key(state)
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(returns.get(state_key, np.zeros(env.action_space.n)))

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, RIGHT_ONLY)  
# env = FrameStack(env, num_frames=4)  
env = GrayScaleObservation(env, keep_dim=True)

env = FrameStack(env, 4)





state = env.reset()
print(state.shape)

state, reward, done, info = env.step(env.action_space.sample())
plt.imshow(state.shape[0])
plt.show()

# obs, info = env.reset()
# print("Observation shape:", obs.shape)  
# print(obs[0][0].shape)
"""
num_episodes = 1  
returns = {}


state_action_counts = {}
gamma = 0.99  
epsilon = 0.1  

for episode in tqdm.trange(num_episodes):
    state = env.reset()
    state = preprocess_observation(state[0])  
    done = False
    episode_rewards = []
    states_actions = []  

    while not done:
        action = choose_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = preprocess_observation(next_state) 
        
        
        states_actions.append((state, action, reward))
        state = next_state  
        env.render()
    
   
    total_return = 0
    for (state, action, reward) in reversed(states_actions):
        total_return = reward + gamma * total_return

        state_tuple = tuple(state.flatten())
        
        
        if (state_tuple, action) not in state_action_counts:
            state_action_counts[(state_tuple, action)] = 0
        if (state_tuple, action) not in returns:
            returns[(state_tuple, action)] = 0

        state_action_counts[(state_tuple, action)] += 1
        returns[(state_tuple, action)] += total_return

env.close()
epsilon = 0.0

for episode in range(5):  
    state = env.reset()
    state = preprocess_observation(state[0])  
    done = False

    while not done:
        action = choose_action(state)  
        next_state, reward, done, _ = env.step(action)
        
        env.render()  

env.close()"""