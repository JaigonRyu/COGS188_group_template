import numpy as np
from gym import Wrapper, Space

from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from gym_super_mario_bros import smb_env


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

class RewardMoveRight(Wrapper):
    
    """Notes
    Scores dont matter in single player mode
    _time : time left 0 - 999
    _coins: # of coins collected 0 - 99
    _life : # of lives remaining
    _player_status: tall, short, fire etc
    _is_dead: is dead then true
    There are 3 lives to the game
    _flag_get: self.is_world_over or _is_stage_over is returned

    """

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
        self.last_pos = 40
    
    def _x_reward(self, info):

        x_pos = info['x_pos']

        if x_pos > self.last_pos:
            reward = 5  # Reward for moving forward
        else:
            reward = -10  # Penalize moving backward
        
        # Update last known X position
        self.last_pos = x_pos
        
        return reward

    def _time_penalty(self, info):
        vals = {
    range(0, 101): -1000,
    range(101, 201): 25,
    range(201, 301): 50,
    range(301, 401): 100,
    range(401, 501): 100,
    range(501, 601): 200,
    range(601, 701): 200,
    range(701, 801): 200,
    range(801, 901): 200,
    range(901, 1000): 200,
}       

        _time = info['time']
        _reward = next((v for k, v in vals.items() if _time in k), 0)  # Default to 0 if not found
        return _reward
    
    def _get_reward(self, info):
        """Return the reward after a step occurs."""
        #might be repetitive ngl, TODO: look at source, does get reward only occur is successful stage clear
        #removed death penalty attmept to encourage riskier but faster methods
        #lower learning rate, as it might be too unstable
        return self._x_reward(info) + self._time_penalty(info)

    def step(self, action):
        x_last = 40
        total_reward = 0.0 
        done = False
        for _ in range(self.skip):
            
            next_state, _ , done, trunc, info = self.env.step(action) 
            
            custom_reward = self._get_reward(info)
            total_reward += custom_reward
            if done:
                break
        
        return next_state, total_reward, done, trunc, info



def apply_wrappers_move_right(env):
    # Step-by-step application for debugging
    print("Applying RewardMoveRight wrapper")
    env = RewardMoveRight(env, skip=4)
    print("Applying ResizeObservation wrapper")
    env = ResizeObservation(env, shape=84)
    print("Applying GrayScaleObservation wrapper")
    env = GrayScaleObservation(env)
    print("Applying FrameStack wrapper")
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env


class RewardScoreBased(Wrapper):
    """ based on score, coins, power ups"""
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def _score_reward(self,  info):
        """google note: 10,000 is high
        using this as a basis, anything over 12,000 will be the same
        
        """
        score = info['score']
        vals ={  
        range(0, 1001): 10,
        range(1001, 3001): 100,
        range(3001, 5001): 1000,
        range(5001, 7001): 2000,
        range(7001, 9001): 3000,
        range(9001, 12001): 4000,
        range(12001, 999990): 5000,
    }       
        
        _reward = next(v for k, v in vals.items() if score in k)
        return _reward
    
    def _coin_reward(self, info):
        return int(info['coins'])
    
    def status(self,info):
        status = info['status']:
        if status == 'tall' or status == 'fireball':
            reward = 5
        else:
            reward = 0
        return reward

    
    def _get_reward(self, info):
        #might be repetitive ngl, TODO: look at source, does get reward only occur is successful stage clear
        #removed death penalty attmept to encourage riskier but faster methods
        #lower learning rate, as it might be too unstable
        return self._score_reward(info) + self._coin_reward(info)
    
    # TODO: add status, like power up into the reward function maybe
    
    def step(self, action):
        total_reward = 0.0 
        done = False
        for _ in range(self.skip):

            next_state, _, done, trunc, info = self.env.step(action) 

            custom_reward = self._get_reward(info)
            total_reward += custom_reward
            if done:
                break
        
        return next_state, total_reward, done, trunc, info
    
def apply_wrappers_score_based(env):
    env = RewardScoreBased(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env


class RewardAntiDeath(Wrapper):
    
    """
    Compatibility and privacy issue, cannot access env.unwrapped
    trying to reinforce safe movement. 
    No time or movement parameter could make it inefficient 
    increasing the 
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def _death_penatly(self):
        if self.env.unwrapped._is_dying or self.env.unwrapped._is_dead:
            return -50
        
        return 0
    
    def _time_penatly(self):
        """
        Attempting to mitigate time loss, because the agent could 
        theoretically just stand still for 3 lives and end with a 
        find reward of -50.
        """
        if self.env.unwrapped._time > 100:
            return -1000
        return 0


    def _reward_remaining_life(self):
        """
        currently is at *100, life is set to 3, 
        if it gets a 1 up by collecting coins it also gets more bonus
        but very unlikely this will happen
        could add to points system to reinforce that to see if it will try to get more lives

        """
        if self.env.unwrapped._get_done == "True":
            return self.env.unwrapped._get_info["life"] * 100
        return 0   

    def _get_reward(self):
        return self._death_penatly + self._reward_remaining_life + self._time_penatly


    def step(self, action):
        """
        start with a reward and take it away as it goes"
        if the goal state isnt reached, aka game over, take away more
        if goal state is reached, give some back

        Every time it dies, it loses points
        When it finishes, if it doesnt die at all, it doesnt lose any
        plus ist gains some

        to get optimal points, it shouldnt die

        No time penalty, so it shuold start being super safe
        take note of impact on the time count. 
        Hopefully, it learns to win in less episodes but will be less effecient in clearing levels

        """
        total_reward = 100 
        done = False
        for _ in range(self.skip):

            next_state, _, done, trunc, info = self.env.step(action) 

            custom_reward = self._get_reward()
            total_reward += custom_reward
            if done:
                break
        
        return next_state, total_reward, done, trunc, info


def apply_wrappers_death_based(env):
    env = RewardAntiDeath(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
    


        
    
