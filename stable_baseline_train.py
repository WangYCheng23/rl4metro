import os
import time
from config import get_args, get_env_args
from stable_baselines3 import PPO
from metro_env_eventbased import MetroEnvEventbased
from env import ExtendedObservation, TruncActions


def train():
    # model_name = f'{int(time.time())}'
    # 加载模型用的代码
    # model_name = '1672300506'
    model_dir = f'models/'
    logs_dir = f'logs/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    args = get_args()
    params = get_env_args(args)
    env = MetroEnvEventbased(params)
    env = ExtendedObservation(env)
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)
    # 加载模型用的代码
    # model = PPO.load(f'{model_dir}/1470000.zip', env, verbose=1, tensorboard_log=logs_dir)

    time_steps = 10000
    iters = 0
    # 加载模型用的代码
    # iters = 148
    while True:
        print(f'On iteration:{iters}')
        model.learn(total_timesteps=time_steps,
                    tb_log_name='PPO', reset_num_timesteps=False)
        model.save(f'{model_dir}/{time_steps * iters}')
        iters += 1


if __name__ == '__main__':
    train()
