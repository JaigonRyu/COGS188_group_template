 
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import numpy as np
import torch 
import os
import wrapper
import tqdm
from agent_and_agent_nn import Agent, AgentNN
import matplotlib.pyplot as plt
import datetime

#check for gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#getting datetime for folder creation to store models
def get_current_date_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

#variables to keep track of the episode length, training or not, and save interval
NUM_EPS = 5000
IS_TRAIN = True
CKPT_SAVE_INTERVAL = 500

#create the envriroment and apply wrappers
env = gym_super_mario_bros.make('SuperMarioBros-v2',apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper.apply_wrappers_score_based(env)  

#init our agent with the corrent observation input and action space output dims 
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

#create models folder to store them in
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

#if we are not straining load a model and set the exploration very low 
if not IS_TRAIN:
    folder_name = "2025-03-15-18_55_35"
    ckpt_name = "model_15000_iter.pt"
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

#caputure rewards
reward_history = []

#start for loop with tqdm for number of episodes specified
for i in tqdm.tqdm(range(NUM_EPS)):

    #reset env and flags
    done = False
    state, _ = env.reset()

    #reset rewards
    episode_reward = 0

    while not done:

        # use our agents choose action and then take a step in the env
        action = agent.choose_action(state)
        new_state, reward, done, trunc, info = env.step(action)

        #if training store this in the replay buffer and continue to learn
        if IS_TRAIN:
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()

        #add rewards and change state
        episode_reward += reward

        state = new_state

    reward_history.append(episode_reward)

    #check for training and if so save the model every interval specified
    if IS_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

# Plot rewards after training
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Episode Reward")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Reward Over Time")
plt.legend()
plt.grid()
plt.show()

env.close()