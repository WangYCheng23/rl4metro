import simpy
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from environment.metro import Metro
from environment.metro_station import MetroStation


class MetroEnvEventbased(gym.Env):
    def __init__(self, parameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.seed(seed=self.parameters["seed"])
        self.first_metro_time = parameters["first_metro_time"]
        self.last_metro_time = parameters["last_metro_time"]
        self.global_clock = 0

        # 停站时间 and 巡航时间
        self.action_space = gym.spaces.Box(low=np.array([self.parameters['stop_time_low'], self.parameters['stop_time_upper']]), 
                                           high=np.array([self.parameters['cruise_speed_low'], self.parameters['cruise_speed_upper']]), shape=(2,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-1, np.inf, shape=(1, (self.parameters['num_metros']-1)*(self.parameters["num_metro_stations"]*4+4)), dtype=np.float64)

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

        next_obs = self.get_observation(decision_metro)
        reward = self.get_reward(decision_metro)
        done = True if all([m.finished for m in self.metros]) else False

        return next_obs, reward, done, {'steps':self.steps}

    def reset(self, seed=2023):
        self.env = simpy.Environment()
        self.cur_time = 0
        self.steps = 0
        self.old_reward_matrix = np.zeros((2, 1))
        self.parameters.update({'step_criteria': self.env.event()})
        self.parameters.update({'continue_criteria': self.env.event()})
        self.config_environment()
        self.env.run(until=self.parameters['step_criteria'])
        obs = self.get_observation()
        return obs

    def get_observation(self, decision_metro=None):
        obs = np.zeros((self.num_metros-1, self.num_metro_stations*4+4))
        if decision_metro == None:
            return obs.reshape(1,-1)
        for i, m in enumerate([m for m in self.metros if m is not decision_metro]):
            if m.time_info == []: 
                observation = []
            else:
                observation = [list(x.values())[-1][-1] for x in m.time_info]
            observation.extend(0 for _ in range(self.num_metro_stations*4-len(observation)))
            observation.extend([m.cur_metro_station.id, m.direction,m.cur_metro_station.dis2nextstation, m.finished])
            obs[i] = observation
        return obs.reshape(1,-1)

    def get_reward(self, decision_metro=None):
        rewards_matrix = np.zeros((2, math.ceil(self.env.now)))
        for i, m in enumerate(self.metros):
            # 先读取time—info
            if m.time_info != []:
                for info in m.time_info:
                    # if (m.direction==0 and list(info.keys())[0] == 'traction_time') or (m.direction==1 and list(info.keys())[0] == 'brake_time'):
                    if list(info.keys())[0] == 'traction_time':
                        rewards_matrix[0, math.ceil(list(info.values())[0][0]):math.ceil(
                            list(info.values())[0][1])] += 1
                    # elif (m.direction==0 and list(info.keys())[0] == 'brake_time') or (m.direction==1 and list(info.keys())[0] == 'traction_time'):
                    elif list(info.keys())[0] == 'brake_time':
                        rewards_matrix[1, math.ceil(list(info.values())[0][0]):math.ceil(
                            list(info.values())[0][1])] += 1
                    elif list(info.keys())[0] == 'cruise_time' or list(info.keys())[0] == 'stop_time':
                        continue
                    else:
                        raise AssertionError
            # # 再计算到当前的
            # if m.cur_traffic_state == "stop" or m.cur_traffic_state == "cruise":
            #     continue
            # # elif (m.direction==0 and m.cur_traffic_state == 'traction') or (m.direction==1 and m.cur_traffic_state == 'brake'):
            # elif m.cur_traffic_state == 'traction':
            #     if m.time_info != []:
            #         rewards_matrix[0, math.ceil(self.env.now - list(m.time_info[-1].values())[0][-1]):math.ceil(self.env.now)] += 1
            #     else:
            #         rewards_matrix[0, 0:math.ceil(self.env.now)] += 1
            # # elif (m.direction==0 and m.cur_traffic_state == 'brake') or (m.direction==1 and m.cur_traffic_state == 'traction'):
            # elif m.cur_traffic_state == 'brake':
            #     if m.time_info != []:
            #        rewards_matrix[1, math.ceil(self.env.now - list(m.time_info[-1].values())[0][-1]):math.ceil(self.env.now)] += 1
            #     else:
            #         rewards_matrix[1, 0:math.ceil(self.env.now)] += 1
            # else:
            #     raise AssertionError
        r = np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum() - \
            np.clip(self.old_reward_matrix.min(axis=0), 0, np.inf).sum()
        assert r >= 0
        self.old_reward_matrix = rewards_matrix

        return r

    def config_environment(self):
        # # 真实数据
        self.metro_stations_name_list = self.parameters["metro_stations_name_list"]
        # ~m
        metro_station_distances = self.parameters["metro_station_distances"]

        # 初始化车站数据
        self.num_metro_stations = len(self.metro_stations_name_list)
        self.metro_stations_forward = [MetroStation(self.parameters, self.env, i,  self.metro_stations_name_list[i],
                                                    metro_station_distances[i] if i < self.num_metro_stations-1 else 0) for i in range(self.num_metro_stations)]
        self.metro_stations_backward = [MetroStation(self.parameters, self.env, self.num_metro_stations-i-1,  self.metro_stations_name_list[::-1]
                                                     [i], metro_station_distances[::-1][i] if i < self.num_metro_stations-1 else 0) for i in range(self.num_metro_stations)]

        # 初始化列车数据
        self.num_metros = self.parameters["num_metros"]
        self.metros = [Metro(self.parameters, self.env, i, 0, 0, 40, self.metro_stations_forward) for i in range(
            self.num_metros//2)] + [Metro(self.parameters, self.env, i, 1, 0, 40, self.metro_stations_backward) for i in range(self.num_metros//2)]

        self.parameters['metros'] = self.metros
        self.parameters['metro_stations_forward'] = self.metro_stations_forward
        self.parameters['metro_station_backward'] = self.metro_stations_backward

    def render(self, mode="human"):
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
                plt.plot(dot_list_x, dot_list_y, color=np.np.random.choice(
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
