import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import gym
from gym.wrappers import FrameStack 
from gym.spaces import Box

import cv2
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T

from dataclasses import dataclass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@dataclass
class HyperParams:
    BATCH_SIZE: int = 512
    GAMMA: float = 0.99
    EPS_START: float = 1.0
    EPS_END: float = 0.05
    EPS_DECAY: int = 5000
    TAU: float = 0.005
    LR: float = 1e-4

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
    
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten_size = self._get_conv_output((4, 84, 84))

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.relu = nn.ReLU()

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(torch.prod(torch.tensor(x.shape[1:])))
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNTrainer:
    def __init__(
            self,
            env: gym_super_mario_bros.make('SuperMarioBros-v3'),
            memory: ReplayMemory,
            device: torch.device,
            params: HyperParams,
            max_steps_per_episode: int = 1000,
            num_episodes: int = 50
    ) -> None:
        
        self.env = env
        self.policy_net = DQN(env.action_space.n).to(device)
        self.target_net = DQN(env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.LR, amsgrad=True)
        self.memory = memory
        self.device = device
        self.params = params
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes
        self.episode_rewards = []
        self.steps_done = 0
    
    def select_action(self, state_tensor: torch.Tensor) -> torch.Tensor:
        eps_threshold = self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * np.exp(-self.steps_done / self.params.EPS_DECAY)

        self.steps_done += 1

        if np.random.rand() < 1 - eps_threshold:
            with torch.no_grad():
                action_idx = self.policy_net(state_tensor).max(1).indices.item()
        else:
            action_idx = np.random.randint(self.env.action_space.n)

        
            
        return torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
    
    def optimize_model(self) -> None:
        if len(self.memory) < self.params.BATCH_SIZE:
            return
        
        minibatch = self.memory.sample(self.params.BATCH_SIZE)
        state_batch, action_batch, next_state_batch, reward_batch = zip(*minibatch)

        state_batch = torch.stack(state_batch).to(self.device)
        action_batch = torch.stack(action_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)

        non_final_mask = torch.tensor([s is not None for s in next_state_batch], dtype=torch.bool, device=self.device)
        non_final_next_states = torch.cat([s for s in next_state_batch if s is not None], dim=0).to(self.device)

        policy_output = self.policy_net(state_batch.squeeze(1))
        action_batch = action_batch.view(-1)
        state_action_values = policy_output.gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.params.BATCH_SIZE).to(self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.params.GAMMA * next_state_values

        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def soft_update(self) -> None:
        policy_state_dict = self.policy_net.state_dict()
        target_state_dict = self.target_net.state_dict()

        for target_param, policy_param in zip(target_state_dict.values(), policy_state_dict.values()):
            target_param.data.copy_((self.params.TAU * policy_param + (1 - self.params.TAU) * target_param))
        
        # self.target_net.load_state_dict(target_state_dict)

    def plot_rewards(self, show_result: bool = False) -> None:
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)

        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training (Reward)")
        
        plt.xlabel("Episode")
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy(), label='Episode Reward')

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    
    def train(self) -> None:
        for _ in range(self.num_episodes):
            obs = self.env.reset()
            obs = np.array(obs)
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_reward = 0.0

            for _ in range(self.max_steps_per_episode):
                action_idx = self.select_action(state)

                obs, reward, done, info = self.env.step(action_idx.item())
                obs = np.array(obs)

                if not done:
                    next_state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    next_state = None
                
                self.memory.push(state, action_idx, next_state, reward)
                state = next_state
                self.optimize_model()
                self.soft_update()
                episode_reward += reward

                frame = self.env.render(mode='rgb_array')
                cv2.imshow("Super Mario Bros", frame)
                cv2.waitKey(1)

                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.plot_rewards()
        
        print('Training Complete')
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.savefig('rewards_plot_dqn.png')
        plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = HyperParams()

env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, 4)

env.metadata['render_modes'] = ['human', 'rgb_array']
env.metadata['render_fps'] = 30

n_actions = env.action_space.n

memory = ReplayMemory(20000)
steps_done = 0

trainer = DQNTrainer(env, memory, device, params, num_episodes=2000)
trainer.train()
