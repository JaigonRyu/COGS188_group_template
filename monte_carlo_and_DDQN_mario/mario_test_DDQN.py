 
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

import wrapper
from wrapper import SkipFrame

import agent_and_agent_nn
from agent_and_agent_nn import Agent, AgentNN

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")



import time
import datetime

def get_current_date_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
NUM_EPS = 1
IS_TRAIN = True
env = gym_super_mario_bros.make('SuperMarioBros-v2',apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper.apply_wrappers(env)  

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
CKPT_SAVE_INTERVAL = 5000

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

if not IS_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

for i in range(NUM_EPS):
    done = False
    state, _ = env.reset()

    rewards = 0

    while not done:

        action = agent.choose_action(state)

        new_state, reward, done, trunc, info = env.step(action)

        agent.store_in_memory(state, action, reward, new_state, done)

       
        rewards += reward

        

        if IS_TRAIN:
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()

        state = new_state

    if IS_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    if IS_TRAIN:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    

env.close()