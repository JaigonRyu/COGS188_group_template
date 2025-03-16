import gymnasium as gym  
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gymnasium.wrappers import GrayScaleObservation, FrameStack, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym import spaces

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import cv2
import numpy as np
from collections import deque
import tqdm
import matplotlib.pyplot as plt
import os
import pickle

video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

env = gym_super_mario_bros.make('SuperMarioBros-v2',apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, RIGHT_ONLY)  

env = FrameStack(env, num_stack=4)


state, info = env.reset()
print(state.shape)

state, reward, done,_, info = env.step(env.action_space.sample())

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def preprocess_observation(observation):

    frames = []

    for i in range(observation.shape[0]):

        frame = observation[i]

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        resize = cv2.resize(gray, (84,84))

        frames.append(resize/255)
    
    return np.array(frames)


def state_to_key(state):
   
    return tuple(state.flatten())


def save_state_action_data(returns, state_action_counts, filename="state_action_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((returns, state_action_counts), f)
    print(f"State-action data saved to {filename}")



CHECK_DIR = './train/' 
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECK_DIR)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00001,
            n_steps=512)

model.learn(total_timesteps=10)