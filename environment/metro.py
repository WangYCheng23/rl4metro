from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

class Metro:
    def __init__(self, parameters, env, id, direction, init_time, mass, stations):
        self.parameters = parameters
        self.env = env
        self.id = id
        self.direction = direction
        self.name = str(direction) + '-' + str(id)
        self.init_time = init_time
        self.mass = mass

        # Variables
        self.stop_time = 0
        self.cruise_speed = 0

        if direction == 0:
            self.departure_time = self.init_time + (self.id) * self.parameters["intervals"]
            # self.stations = stations
        elif direction == 1:
            self.departure_time = self.init_time + 1e-9 + (self.id) * self.parameters["intervals"]
            # self.stations = stations[::-1]
        self.stations = stations

        self.last_metro_station = None
        self.cur_metro_station = self.stations[0]
        self.next_metro_station = self.stations[1]

        self.wait_action = False
        self.finished = False

        self.info = defaultdict(dict)
        self.distance_info = []
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
            total_time, traction_time, cruise_time, brake_time = self.time_model(self.stop_time, self.cruise_speed, self.cur_metro_station)
            self.info[self.cur_metro_station.name] = {'cur_time': self.env.now, 'traction_time': sum(traction_time),
                                                      'cruise_time': cruise_time, 'brake_time': sum(brake_time), 'stop_time': self.stop_time}

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
            self.time_info.append(dict(traction_time=[self.env.now - (traction_time[0]+traction_time[1]), self.env.now]))

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
            self.time_info.append(dict(brake_time=[self.env.now - (brake_time[0]+brake_time[1]), self.env.now]))

            self.mechanics_info.append([self.env.now + 1e-9, 0])
            self.cur_traffic_state = "stop"
            # print(f"station: {self.cur_metro_station} stop_time: {self.stop_time}")
            yield self.env.timeout(self.stop_time)
            self.time_info.append(dict(stop_time=[self.env.now - self.stop_time, self.env.now]))

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

            # 扰动
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

        traction_distance = 0.5*self.parameters['a1']*(traction_time[0]**2) + self.parameters['v1']*traction_time[1] + 0.5*self.parameters['b1']*(traction_time[1]**2)
        brake_distance = 0.5*self.parameters['b2']*(brake_time[1]**2) + self.parameters['v2']*brake_time[0] + 0.5*self.parameters['a2']*(brake_time[0]**2)
        cruise_distance = metro_station.dis2nextstation - traction_distance - brake_distance
        self.distance_info.append([traction_distance, cruise_distance, brake_distance])

        if cruise_distance < 0:
            raise ValueError(f"metro_name: {self.name} | metro_station: {metro_station.name} | distance:{metro_station.dis2nextstation} | t_distance: {traction_distance} | b_distance: {brake_distance}")
        cruise_time = cruise_distance/cruise_speed
        total_time = stop_time + sum(traction_time) + sum(brake_time) + cruise_time

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
