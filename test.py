import time
import torch
import numpy as np
from config import get_args, get_env_args
from environment.metro_env_eventbased import MetroEnvEventbased


def test():
    args = get_args()
    params = get_env_args(args)
    metro_env = MetroEnvEventbased(params)
    iter = 0

    time1 = time.time()   
    while iter < 10:
        rew_li = []
        steps = 0
        next_obs = metro_env.reset()
        for _ in range(1000):
            action = torch.FloatTensor(
                [(params["stop_time_upper"]-params["stop_time_low"])*np.random.random()+params["stop_time_low"], 
                 (params["cruise_speed_upper"]-params["cruise_speed_low"])*np.random.random()+params["cruise_speed_low"]])
            next_obs, reward, done, _ = metro_env.step(action)
            rew_li.append(reward)
            steps += 1
            if done:
                print(f"iter: {iter} | ep_len:{steps} | ep_rew:{sum(rew_li)}")
                break
        iter += 1
    print(time.time()-time1)

    # metro_env.render()
    print('finish')

    
if __name__ == "__main__":
    test()