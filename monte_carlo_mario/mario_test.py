import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import gym

import cv2
import numpy as np
from collections import deque
import tqdm

class MonteCarloAgent():

    def __init__(self,):
        pass

class FrameStack(gym.Wrapper):
    
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames #4
        self.frames = deque(maxlen=num_frames)
        obs_shape = (num_frames, 84, 84)
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=obs_shape)

    def reset(self, **kwargs):
        obs_info = self.env.reset(**kwargs)
    
        
        if isinstance(obs_info, tuple):  
            obs, info = obs_info
        else:  
            obs, info = obs_info, {}
        
        obs = self.convert_frame(obs)

        for _ in range(self.num_frames):
            self.frames.append(obs)

        return np.array(self.frames), info
    
    def step(self, action):
        
        state, reward, done, truncated, info = self.env.step(action)
        state = self.convert_frame(state)
        self.frames.append(state)
        return np.array(self.frames), reward, done, truncated, info
    
    def convert_frame(self, frame):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)

        return frame / 255.0



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

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, RIGHT_ONLY)  
# env = FrameStack(env, num_frames=4)  


# obs, info = env.reset()
# print("Observation shape:", obs.shape)  
# print(obs[0][0].shape)

num_episodes = 5  
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

epsilon = 0.0

for episode in range(5):  
    state = env.reset()
    state = preprocess_observation(state[0])  
    done = False

    while not done:
        action = choose_action(state)  
        next_state, reward, done, _ = env.step(action)
        
        env.render()  