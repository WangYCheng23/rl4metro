import time
import torch
import numpy as np
from config import get_args, get_env_args
from environment.metro_env_eventbased import MetroEnvEventbased
from stable_baselines3 import PPO


def test():
    args = get_args()
    params = get_env_args(args)
    metro_env = MetroEnvEventbased(params)
    iterations = params['test_iterations']

    # 随机动作test]
    iter = 0
    time1 = time.time() 
    rew_mean1 = 0  
    while iter < iterations:
        rew_li = []
        steps = 0
        next_obs = metro_env.reset()
        for _ in range(1000):
            action = torch.FloatTensor([np.random.random()*2-1, np.random.random()*2-1])
            next_obs, reward, done, _ = metro_env.step(action)
            rew_li.append(reward)
            steps += 1
            if done:
                print(f"iter: {iter} | ep_len:{steps} | ep_rew:{sum(rew_li)}")
                break
        iter += 1
        rew_mean1 += sum(rew_li)/iterations
    print(time.time()-time1)
    print(rew_mean1)

    # policy动作test
    agent = PPO.load('./models/2023-0418-10-40-28-PPO-20-24/PPO/7280640.zip')
    iter = 0
    time2 = time.time()
    rew_mean2 = 0 
    while iter < iterations:
        rew_li = []
        steps = 0
        next_obs = metro_env.reset()
        for _ in range(1000):
            action = agent.predict(next_obs)[0]
            next_obs, reward, done, _ = metro_env.step(action)
            rew_li.append(reward)
            steps += 1
            if done:
                print(f"iter: {iter} | ep_len:{steps} | ep_rew:{sum(rew_li)}")
                break
        iter += 1
        rew_mean2 += sum(rew_li)/iterations
    print(time.time()-time2)
    print(rew_mean2)
    # metro_env.render()
    print('finish')

    
if __name__ == "__main__":
    test()