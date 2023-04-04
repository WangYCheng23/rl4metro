import gym
import os
import torch
import numpy as np
import random
from metro_env_eventbased import MetroEnvEventbased


class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

class ExtendedObservation(gym.ObservationWrapper):
    def observation(self, observation):
        return observation.reshape(1,-1)

class NormalizedObservation(gym.ObservationWrapper):
    def observation(self, observation):
        ob1 = observation[:, :-1].reshape(1, -1)
        ob2 = observation[:, -1].reshape(1, -1)
        return np.concatenate((ob1, ob2), axis=1)


class TruncActions(gym.ActionWrapper):
    def action(self, action):
        low_st = self.parameters['stop_time_low']
        upper_st = self.parameters['stop_time_upper']
        low_cs = self.parameters['cruise_speed_low']
        upper_cs = self.parameters['cruise_speed_upper']

        action[0] = low_st + (action[0]+1)*0.5*(upper_st-low_st)
        action[1] = low_cs + (action[1]+1)*0.5*(upper_cs-low_cs)
        return action

