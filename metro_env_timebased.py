import simpy
import gym
import numpy as np

class SubwayTrain(gym.Env):
    def __init__(self, env, train_id):
        self.env = env
        self.id = train_id
        self.position = 0
        self.velocity = 0
        self.energy_consumption = 0
        self.arrived = False
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1000, 100]), dtype=np.float32)

    def cruise(self):
        self.velocity = self.action[0]
        self.position += self.velocity
        self.energy_consumption += self.velocity ** 2

    def stop(self):
        self.velocity = 0
        self.energy_consumption += 50

    def step(self, action):
        self.action = action
        self.cruise()
        if self.position >= 1000:
            self.arrived = True

class Station:
    def __init__(self, env):
        self.env = env
        self.is_open = False
        self.waiting_passengers = 0

    def open(self):
        self.is_open = True
        self.waiting_passengers = self.env.now % 50
        self.env.process(self.close())

    def close(self):
        yield self.env.timeout(10)
        self.is_open = False
        self.waiting_passengers = 0

class MetroEnvTimebased(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = simpy.Environment()
        self.train_list = [SubwayTrain(self.env, i) for i in range(3)]
        self.station = Station(self.env)
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(6,))
        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([100.0, 60.0]), dtype=np.float32)
        self.process = self.env.process(self.step())

    def step(self):
        while True:
            # 等待事件触发
            yield simpy.Event()

            # 解析动作
            action = self.action_space.sample()

            # 执行动作
            for i in range(len(self.train_list)):
                self.train_list[i].step(action[i])

            # 运行一步
            self.env.run(until=self.env.now + 1)

            # 计算状态
            state = []
            for train in self.train_list:
                state.append(train.position)
                state.append(train.velocity)
            state.append(self.station.is_open)
            state.append(self.station.waiting_passengers)

            # 计算奖励
            reward = -sum([train.energy_consumption for train in self.train_list])

            # 判断是否结束
            done = all([train.arrived for train in self.train_list])

            yield (state, reward, done, {})

    def reset(self):
        self.env = simpy.Environment()
        self.train_list = [SubwayTrain(self.env, i) for i in range(3)]
        self.station = Station(self.env)
        return self.step()[0]

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = MetroEnvTimebased()
    state = env.reset()