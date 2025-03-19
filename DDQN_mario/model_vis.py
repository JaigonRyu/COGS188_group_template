from torchinfo import summary
import torch
from torchviz import make_dot
from agent_and_agent_nn import Agent
import os
import matplotlib.pyplot as plt
import wrapper
import gym_super_mario_bros
from tqdm import tqdm

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np

def load_agent(model_path):
    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
    agent.load_model(model_path)
    agent.epsilon = 0.1
    agent.eps_min = 0.0
    agent.eps_decay = 0.0
    return agent

def evaluate_agent(agent, num_episodes=5):
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)  
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
        rewards.append(total_reward)
    
    return np.mean(rewards)



#model_folder = "models\\2025-03-18-00_09_42" <- new hyper parameters
#model_folder = 'models\\2025-03-15-18_55_35'
model_folder = 'models\\2025-03-18-17_40_23'
model_files = [f for f in os.listdir(model_folder) if f.endswith(".pt")]

model_paths = [os.path.join(model_folder, f) for f in model_files]
creation_times = [os.path.getctime(m) for m in model_paths]

sorted_indices = sorted(range(len(model_files)), key=lambda i: creation_times[i])
model_files_sorted = [model_files[i] for i in sorted_indices]
print(model_files_sorted)

performance = []
import cv2
env = gym_super_mario_bros.make('SuperMarioBros-v2',apply_api_compatibility=True, render_mode='rgb_array')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper.apply_wrappers(env) 


for model_file in tqdm(model_files_sorted):
    model_path = os.path.join(model_folder, model_file)
    agent = load_agent(model_path)
    avg_reward = evaluate_agent(agent)
    performance.append((model_file, avg_reward))

# Convert to NumPy array for easy plotting
performance = np.array(performance, dtype=object)

plt.figure(figsize=(10, 5))
plt.plot(performance[:, 0], performance[:, 1], marker="o", linestyle="-", color="b", label="Avg Reward")
plt.xlabel("Model Checkpoint")
plt.ylabel("Average Reward")
plt.xticks(rotation=45)
plt.title("Performance of Trained Models")
plt.legend()
plt.grid()
plt.show()

env.close()