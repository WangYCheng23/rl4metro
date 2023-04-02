import time
from config import get_args, get_env_args
import simpy
import gym
import math
import torch
import pandas as pd
import numpy as np
import random
import datetime as dt
import matplotlib.pyplot as plt
from collections import defaultdict


class Metro:
    def __init__(self, parameters, env, id, direction, init_time, mass, stations):
        self.parameters = parameters
        self.env = env
        self.id = id
        self.direction = direction
        self.name = str(direction) + '-' + str(id)
        self.init_time = init_time
        self.mass = mass
        self.stop_time = 0
        self.cruise_speed = 0

        if direction == 0:
            self.departure_time = self.init_time + \
                (self.id) * self.parameters["intervals"]
            # self.stations = stations
        elif direction == 1:
            self.departure_time = self.init_time + 1e-9 + \
                (self.id) * self.parameters["intervals"]
            # self.stations = stations[::-1]
        self.stations = stations    

        self.last_metro_station = None
        self.cur_metro_station = self.stations[0]
        self.next_metro_station = self.stations[1]

        self.wait_action = False
        self.finished = False

        self.info = defaultdict(dict)
        self.speed_info = []
        self.mechanics_info = []
        self.power_info = []
        self.time_info = []

        self.cur_traffic_state = "stop"
        self.process = self.env.process(self.running())

    def get_cur_info(self):
        if self.cur_traffic_state == "stop":
            pass
        elif self.cur_traffic_state == "traction":
            pass
        elif self.cur_traffic_state == "cruise":
            pass
        elif self.cur_traffic_state == "brake":
            pass
        else:
            raise ValueError

    def wait_instruction(self):
        self.parameters['step_criteria'].succeed()
        self.parameters['step_criteria'] = self.env.event()

        self.wait_action = True
        yield self.parameters['continue_criteria']
        self.wait_action = False

    def running(self):
        n = 0
        while True:
            # 初始站点和出发时间
            if n == 0:
                assert self.departure_time >= 0
                yield self.env.timeout(self.departure_time)
                self.info['departure'] = self.env.now

            # 决策过程
            yield self.env.process(self.wait_instruction())
            total_time, traction_time, cruise_time, brake_time = self.time_model(
                self.stop_time, self.cruise_speed, self.cur_metro_station)
            self.info[self.cur_metro_station.name] = {'cur_time': self.env.now, 'traction_time': sum(traction_time),
                                                      'cruise_time': cruise_time, 'brake_time': sum(brake_time), 'stop_time': self.stop_time}
            cumsum_time_list = np.cumsum(
                [self.env.now, sum(traction_time), cruise_time, sum(brake_time)])
            # self.parameters['global_info'][self.id,
            #    n*4:(n+1)*4] = cumsum_time_list
            self.parameters['brake_time_list'].append(
                [cumsum_time_list[2], cumsum_time_list[3]])

            # 执行过程
            self.mechanics_info.append([self.env.now, 0])
            self.mechanics_info.append(
                [self.env.now + 1e-9, self.parameters['a1']])
            self.speed_info.append([self.env.now, 0])
            self.cur_traffic_state = "traction"
            yield self.env.timeout(traction_time[0])
            self.mechanics_info.append([self.env.now, self.parameters['a1']])
            self.speed_info.append([self.env.now, self.parameters['v1']])
            yield self.env.timeout(traction_time[1])
            self.mechanics_info.append([self.env.now, self.parameters['b1']])
            self.speed_info.append([self.env.now, self.cruise_speed])
            self.time_info.append(dict(traction_time=[
                                  self.env.now - (traction_time[0]+traction_time[1]), self.env.now]))

            self.cur_traffic_state = "cruise"
            self.mechanics_info.append([self.env.now + 1e-9, 0])
            yield self.env.timeout(cruise_time)
            self.mechanics_info.append([self.env.now - 1e-9, 0])
            self.speed_info.append([self.env.now, self.cruise_speed])
            self.time_info.append(
                dict(cruise_time=[self.env.now - cruise_time, self.env.now]))

            self.cur_traffic_state = "brake"
            self.mechanics_info.append([self.env.now, -self.parameters['a2']])
            yield self.env.timeout(brake_time[0])
            self.mechanics_info.append([self.env.now, -self.parameters['b2']])
            self.speed_info.append([self.env.now, self.parameters['v2']])
            yield self.env.timeout(brake_time[1])
            self.mechanics_info.append([self.env.now, -self.parameters['b2']])
            self.speed_info.append([self.env.now, 0])
            self.time_info.append(
                dict(brake_time=[self.env.now - (brake_time[0]+brake_time[1]), self.env.now]))

            self.mechanics_info.append([self.env.now + 1e-9, 0])
            self.cur_traffic_state = "stop"
            yield self.env.timeout(self.stop_time)
            self.time_info.append(
                dict(stop_time=[self.env.now - self.stop_time, self.env.now]))

            # 确定前后站
            n += 1
            self.last_metro_station = self.stations[n-1]
            self.cur_metro_station = self.stations[n]
            if n >= len(self.stations)-1:
                self.next_metro_station = None
                self.finished = True
                if all([m.finished for m in self.parameters['metros']]) == True:
                    self.parameters['step_criteria'].succeed()
                # self.parameters['global_info'][self.id][-1] = 1
                break
            else:
                self.next_metro_station = self.stations[n+1]

            #扰动
            if not self.finished:
                disturbance = random.choice(range(0, 14))
                yield self.env.timeout(disturbance)
                self.info[self.cur_metro_station.name]['disturbance'] = disturbance

    # To Do
    def energy_model(self, label):
        if label == "t":
            Et = 0
            return Et
        elif label == "b":
            Er = 0
            return Er

    def mechanics_model(self, label, vc):
        if label == "t":
            t1 = self.parameters['v1']/self.parameters['a1']
            t2 = (vc-self.parameters['v1'])/self.parameters['b1']
            return [t1, t2]

        if label == "b":
            t1 = (vc - self.parameters['v2'])/self.parameters['a2']
            t2 = self.parameters['v2']/self.parameters['b2']
            return [t1, t2]

    def time_model(self, stop_time, cruise_speed, metro_station):
        assert cruise_speed is not None
        traction_time = self.mechanics_model('t', cruise_speed)
        brake_time = self.mechanics_model('b', cruise_speed)

        traction_distance = 0.5*self.parameters['a1']*(
            traction_time[0]**2) + self.parameters['v1']*traction_time[1] + 0.5*self.parameters['b1']*(traction_time[1]**2)
        brake_distance = 0.5*self.parameters['b2']*(
            brake_time[1]**2) + self.parameters['v2']*brake_time[0] + 0.5*self.parameters['a2']*(brake_time[0]**2)

        assert (metro_station.dis2nextstation -
                traction_distance - brake_distance) > 0
        cruise_time = (metro_station.dis2nextstation -
                       traction_distance-brake_distance)/cruise_speed
        total_time = stop_time + \
            sum(traction_time) + sum(brake_time) + cruise_time

        return total_time, traction_time, cruise_time, brake_time

    def render(self):
        t_v = list(list(zip(*self.speed_info))[0])
        v = list(list(zip(*self.speed_info))[1])

        t_f = list(list(zip(*self.mechanics_info))[0])
        f = list(list(zip(*self.mechanics_info))[1])

        fig, ax = plt.subplots(3, 1, figsize=(25, 10))
        # 速度图
        ax1 = ax[0]
        ax1.set_title(f"metro{self.name} speed graph")
        # ax1.set_ylim(bottom = 0)
        ax1.set_ylabel('speed (m/s)')
        ax1.set_xlabel('time (s)')
        ax1.plot(t_v, v)
        # 功率图
        ax2 = ax[1]
        ax2.set_title(f"metro{self.name} mechanics graph")
        ax2.set_ylabel('acceleration (m/s2)')
        ax2.set_xlabel('time (s)')
        ax2.plot(t_f, f)
        # 能耗图
        ax3 = ax[2]

        plt.show()


class MetroStation:
    def __init__(self, parameters, env, id, name, dis2nextstation):
        self.parameters = parameters
        self.env = env
        self.id = id
        self.name = name
        self.dis2nextstation = dis2nextstation
        self.info = defaultdict(dict)

    def reset(self):
        self.info = defaultdict(dict)


class MetroEnvEventbased(gym.Env):
    def __init__(self, parameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.seed(seed=self.parameters["seed"])
        self.first_metro_time = parameters["first_metro_time"]
        self.last_metro_time = parameters["last_metro_time"]
        self.global_clock = 0

        # 停站时间 and 巡航时间
        self.action_space = gym.spaces.Box(low=np.array([self.parameters['stop_time_low'], self.parameters['stop_time_upper']]), high=np.array(
            [self.parameters['cruise_speed_low'], self.parameters['cruise_speed_upper']]), shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, ((self.parameters['num_metros']-1)*(20*4+4),), np.float32)

    def step(self, action):
        # action: 停车等待时间 and 巡航速度
        action = action.tolist()
        # print(action)
        stop_time = action[0]
        cruise_speed = action[1]
        self.steps += 1
        for m in self.metros:
            if m.wait_action == True:
                decision_metro = m
                m.stop_time = stop_time
                m.cruise_speed = cruise_speed
                self.parameters['continue_criteria'].succeed()
                self.parameters['continue_criteria'] = self.env.event()
                break
        self.env.run(until=self.parameters['step_criteria'])

        # next_obs = self.parameters['global_info']
        next_obs = self.get_observation(decision_metro)
        reward = self.get_reward(decision_metro)
        terminal = True if all([m.finished for m in self.metros]) else False
        return next_obs, reward, terminal, {}

    def reset(self):
        self.env = simpy.Environment()
        self.cur_time = 0
        self.steps = 0
        self.reward = 0
        self.parameters.update({'brake_time_list': []})
        self.parameters.update({'step_criteria': self.env.event()})
        self.parameters.update({'continue_criteria': self.env.event()})
        self.config_environment()
        self.parameters['global_info'] = np.zeros(
            [self.num_metros, (self.num_metro_stations)*4 + 1])
        self.env.run(until=self.parameters['step_criteria'])
        # obs = self.parameters['global_info']
        obs = self.get_observation(self.metros[0])
        return obs

    def get_observation(self, decision_metro=None):
        # 
        obs = np.zeros((self.num_metros-1, self.num_metro_stations*4+4))
        for i, m in enumerate([m for m in self.metros if m is not decision_metro]):
            if m.time_info == []:
                observation = []
            else: 
                observation =[list(x.values())[-1][-1] for x in m.time_info]
            observation.extend(0 for _ in range(self.num_metro_stations*4-len(observation)))
            observation.extend([m.cur_metro_station.id, m.direction, m.cur_metro_station.dis2nextstation, m.finished])
            obs[i] = observation
        return obs

    def get_reward(self, decision_metro):
        rewards_matrix = np.zeros((2, math.ceil(self.env.now)))
        for i, m in enumerate(self.metros):
            # 先读取time——info
            if m.time_info != []:
                for info in m.time_info:
                    if list(info.keys())[0] == 'traction_time':
                        rewards_matrix[0, math.ceil(list(info.values())[0][0]):math.ceil(
                            list(info.values())[0][1])] += 1
                    elif list(info.keys())[0] == 'brake_time':
                        rewards_matrix[1, math.ceil(list(info.values())[0][0]):math.ceil(
                            list(info.values())[0][1])] += 1
            # 再计算到当前的
            if m.cur_traffic_state == 'traction':
                if m.time_info != []:
                    rewards_matrix[0, math.ceil(
                        self.env.now - list(m.time_info[-1].values())[0][-1]):math.ceil(self.env.now)] += 1
                else:
                    rewards_matrix[1, 0:math.ceil(self.env.now)] += 1
            elif m.cur_traffic_state == 'brake':
                rewards_matrix[1, math.ceil(
                    self.env.now - list(m.time_info[-1].values())[0][-1]):math.ceil(self.env.now)] += 1
        r = np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum() - self.reward
        self.reward = np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum()

        return r

    def config_environment(self):
        # 真实数据
        if self.parameters["test_mode"]:
            metro_stations = pd.read_csv(
                './raw_data/地铁站点信息_2920000403624.csv', sep=',')
            metro_stations = metro_stations[metro_stations['LINE_NAME']
                                            == 1]['SITE_NAME'].tolist()
        # 随机生成车站数据
        else:
            metro_stations = ["".join([chr(random.randint(0x4e00, 0x9fbf)) for _ in range(2)]) for _ in range(
                self.parameters["low_num_stations"], self.parameters["upper_num_stations"])]
        self.metro_stations_name_list = metro_stations
        # ~m
        metro_station_distances = np.random.uniform(
            low=self.parameters["distance_low"], high=self.parameters["distance_high"], size=len(metro_stations)-1).round(2)

        # metro_station_distances = [2250 for _ in range(len(metro_stations))]

        # 初始化车站数据
        self.num_metro_stations = len(self.metro_stations_name_list)
        self.metro_stations_forward = [MetroStation(self.parameters, self.env, i, metro_stations[i], metro_station_distances[i] if i<self.num_metro_stations-1 else 0) for i in range(self.num_metro_stations)]
        self.metro_stations_backward = [MetroStation(self.parameters, self.env, self.num_metro_stations-i-1, metro_stations[::-1][i], metro_station_distances[::-1][i] if i<self.num_metro_stations-1 else 0) for i in range(self.num_metro_stations)]

        # 初始化列车数据
        self.num_metros = self.parameters["num_metros"]
        self.metros = [Metro(self.parameters, self.env, i, 0, 0, 40, self.metro_stations_forward) for i in range(
            self.num_metros//2)] + [Metro(self.parameters, self.env, i, 1, 0, 40, self.metro_stations_backward) for i in range(self.num_metros//2)]

        self.parameters['metros'] = self.metros
        self.parameters['metro_stations_forward'] = self.metro_stations_forward
        self.parameters['metro_station_backward'] = self.metro_stations_backward

    def render(self):
        color_li = self.parameters['colors']
        plt.rcParams["font.sans-serif"] = ['simhei']  # 设置字体
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(15, 10))
        cum_dis = np.cumsum([m.dis2nextstation for m in self.metro_stations])
        # plt.yticks(list(range(0, len(self.metro_stations))),
        #            [str(m.name) for m in self.metro_stations])
        ax.set_yticks(list(np.insert(cum_dis, 0, 0)[:-1]))  # 设置刻度
        ax.set_yticklabels([str(m.name) for m in self.metro_stations],
                           rotation=30, fontsize=14)  # 设置刻度标签

        # 运行时图
        for metro in self.metros:
            if metro.direction == 0:
                dot_list_y = []
                dot_list_x = []
                # t_tmp = metro.departure_time
                n = 0
                for k, v in metro.info.items():
                    if k == 'departure':
                        dot_list_x.append(v)
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n])
                    else:
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n+1])
                        dot_list_x.append(np.cumsum(list(v.values()))[3])

                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n+1])
                        dot_list_x.append(np.cumsum(list(v.values()))[-1])
                        n += 1
                        if n == len(self.parameters['metro_stations'])-1:
                            break
                plt.plot(dot_list_x, dot_list_y, color=random.choice(
                    color_li), label=f'Metro-Train-Station')
            else:
                dot_list_y = []
                dot_list_x = []
                # t_tmp = metro.departure_time
                n = len(self.parameters['metro_stations'])-1
                for k, v in metro.info.items():
                    if k == 'departure':
                        dot_list_x.append(v)
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n])
                    else:
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n-1])
                        dot_list_x.append(np.cumsum(list(v.values()))[3])

                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n-1])
                        dot_list_x.append(np.cumsum(list(v.values()))[-1])
                        n -= 1
                        if n == 0:
                            break
                plt.plot(dot_list_x, dot_list_y, label=f'Metro-Train-Station')

        plt.grid(axis='y', linestyle='-.',
                 linewidth=1, color='black', alpha=0.5)
        plt.xticks(fontsize=14)
        plt.xlabel('时间 (s)', fontsize=14)
        plt.ylabel('地铁站', fontsize=14)
        plt.show()

        for m in self.metros:
            m.render()


if __name__ == "__main__":
    args = get_args()
    params = get_env_args(args)
    metro_env = MetroEnvEventbased(params)
    next_obs = metro_env.reset()
    rew_li = []
    while True:
        action = torch.FloatTensor(
            [random.randint(10, 15), random.randint(25, 40)])
        next_obs, reward, terminal, _ = metro_env.step(action)
        rew_li.append(reward)
        if terminal:
            break
    print(rew_li)
    print(sum(rew_li))
    metro_env.render()
    print('finish')
