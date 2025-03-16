import numpy as np
from gym import Wrapper, Space

from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

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
        
    def _x_reward(self):
        #x delta changes alot when death happens
        if self.env.unwrapped._x_position - self.env.unwrapped._x_position_last < -5 or self.env.unwrapped._x_position - self.env.unwrapped._x_position_last > 5:
            return 0
        #heavily promte moving foward
        if self.env.unwrapped._x_position > self.env.unwrapped._x_position_last:
            _reward = 5
        #punish moving back, make it run foward asap
        elif self.env.unwrapped._x_position_last >  self.env.unwrapped._x_position:
            reward  = - 10
        self.env.unwrapped._x_position_last = self.env.unwrapped._x_position
        return _reward
    
    def _time_penalty(self):
        vals = {
    range(0, 101): -100,
    range(101, 201): -75,
    range(201, 301): -40,
    range(301, 401): -20,
    range(401, 501): 0,
    range(501, 601): 20,
    range(601, 701): 40,
    range(701, 801): 75,
    range(801, 901): 100,
    range(901, 1000): 200,
}       
        # _reward = vals(self.env.unwrapped._time)
        _reward = next(v for k, v in vals.items() if self.env.unwrapped._time in k)
        return _reward

    def _get_reward(self):
        """Return the reward after a step occurs."""
        #might be repetitive ngl, TODO: look at source, does get reward only occur is successful stage clear
        #removed death penalty attmept to encourage riskier but faster methods
        #lower learning rate, as it might be too unstable
        if self.env.unwrapped._is_stage_over(self) == True:
            return self._x_reward + self._time_penalty + 1000
        else:
            self._x_reward + self._time_penalty - 1000

    def step(self, action):
        total_reward = 0.0 
        done = False
        for _ in range(self.skip):

            next_state, _, done, trunc, info = self.env.step(action) 

            custom_reward = self._get_reward()
            total_reward += custom_reward
            if done:
                break
        
        return next_state, total_reward, done, trunc, info


class RewardScoreBased(Wrapper):
    """ based on score, coins, power ups"""
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def _score_reward(self):
        """google note: 10,000 is high
        using this as a basis, anything over 12,000 will be the same
        
        """
        vals ={  
        range(0, 101): -100,
        range(101, 201): -75,
        range(201, 301): -40,
        range(301, 401): -20,
        range(12000, 999990): 100,
    }       


class RewardAntiDeath(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    


        
    
