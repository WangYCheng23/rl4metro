import gym
import os
import torch
import numpy as np
import random
from model import Actor, Critic
from replaybuffer import ReplayBuffer
from ddpg import DDPG
from metro_env_eventbased import MetroEnvEventbased
# from stable_baselines3 import DDPG


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


def env_agent_config(cfg):
    env = MetroEnvEventbased(cfg)
    env = TruncActions(env)
    env = ExtendedObservation(env)
    # env = NormalizedActions(gym.make('Pendulum-v1', g=9.81)) # 装饰action噪声
    # if cfg['seed'] !=0:
    #     all_seed(env,seed=cfg['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    # 更新n_states和n_actions到cfg参数中
    cfg.update({"n_states": n_states, "n_actions": n_actions})
    models = {"actor": Actor(n_states, n_actions, hidden_dim=cfg['actor_hidden_dim']), "critic": Critic(
        n_states, n_actions, hidden_dim=cfg['critic_hidden_dim'])}
    memories = {"memory": ReplayBuffer(cfg['memory_capacity'])}
    agent = DDPG(models, memories, cfg)
    # agent = DDPG("MlpPolicy", env, verbose=1)
    return env, agent
