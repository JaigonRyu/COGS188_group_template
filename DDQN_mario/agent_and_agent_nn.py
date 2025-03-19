import torch
from torch import nn
import numpy as np

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

#creating our CNN
class AgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        #we have 3 conv layers and 3 relu layers
        #the conv layer starts by taking the input channel and producing 32  
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # conv out size is a dummy forward pass to get the num of neurons for the first linear layer
        # keeping it dynamic helps for any size issues in the future if we change
        conv_out_size = self._get_conv_out(input_shape)

        # linear layers
        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # stops pytorch from computing the gradients that we need for the target network
        if freeze:
            self._freeze()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    #forward pass 
    def forward(self, x):
        return self.network(x)

    #helper funciton to get input for first linear layer
    def _get_conv_out(self, shape):

        #dummy first pass
        o = self.conv_layers(torch.zeros(1, *shape))

        #return the size
        return int(np.prod(o.size()))
    
    #stops the gradient using False
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False

#Our DDQN agent
#we will use tensordicts ReplayBuffer
# using lazy maps from tensor as well to save space and time compared to list
class Agent:

    #hyperparameters 
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.0001, 
                 gamma=0.95, 
                 epsilon=1.0, 
                 eps_decay=0.99999, 
                 eps_min=0.1, 
                 replay_buffer_capacity=100_000, 
                 batch_size=64, 
                 sync_network_rate=5000):
        
        #num actions with right only = 5
        self.num_actions = num_actions
        self.learn_step_counter = 0

        #hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        #networks for target and online 
        #notice that the target network has the freeze to make sure the parameters are frozen
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss() 

        # replay buffer using lazy map storage
        storage = LazyMemmapStorage(replay_buffer_capacity)

        #add the storage into our replay buffer
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):

        #epsilon greedy approach
        if np.random.random() < self.epsilon:

            # select random action
            return np.random.randint(self.num_actions)
    
        #we want to get current observation in the network 
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        
        #return the best action of the observation
        return self.online_network(observation).argmax().item()
    
    #using decay to reduce epsilon steadily and to make sure it does not go past min value
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    #store in memory takes in the tuple (s, a, r, s', done) and stores it as a tensordict into the replay buffer
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
    
    #this is how we copy the weights of the online network to the target network after enought steps has passed
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    #saves online model to use later
    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    #loads models to use later
    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    #learning function
    def learn(self):

        #need to check if there is enought experiences to get a useful batch
        if len(self.replay_buffer) < self.batch_size:
            return
        
        #sync the networks
        self.sync_networks()
        
        #clear the gradients
        self.optimizer.zero_grad()

        #obtain a sample from the replay buffer
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        #create keys for our tuple
        keys = ("state", "action", "reward", "next_state", "done")

        #store the samples parts into variables
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        # pass these states values through the network to get the predicted q value for the online network
        predicted_q_values = self.online_network(states) # Shape is (batch_size, n_actions)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        #get the target q values as well by obtaining next state max action
        target_q_values = self.target_network(next_states).max(dim=1)[0]
       
        #calculate the target q value using 
        # 1- dones sets the future rewards to zero if in a done state
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # calc loss of the predicted and target values
        loss = self.loss(predicted_q_values, target_q_values)

        #back prop to calculate gradients
        loss.backward()

        #grad dec with those gradients
        self.optimizer.step()

        #inc the counter and decay epsilon
        self.learn_step_counter += 1
        self.decay_epsilon()