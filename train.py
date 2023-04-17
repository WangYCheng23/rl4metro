import os
import datetime as dt
from config import get_args, get_env_args
from stable_baselines3 import PPO
from environment.metro_env_eventbased import MetroEnvEventbased
from stable_baselines3.common.env_checker import check_env


def train():
    args = get_args()
    params = get_env_args(args)
    env = MetroEnvEventbased(params)
    check_env(env=env, warn=True, skip_render_check=False)

    time_str = dt.datetime.strftime(dt.datetime.now(),'%Y-%m%d-%H-%M-%S')
    model_name = f"{time_str}-{params['algo_name']}-{params['num_metros']}-{params['num_metro_stations']}"
    log_name = f"{time_str}-{params['algo_name']}-{params['num_metros']}-{params['num_metro_stations']}"
    model_dir = os.path.join('./models', model_name)
    logs_dir = os.path.join('./logs', log_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    model_ppo = PPO("MlpPolicy", env, n_steps=420, verbose=1, tensorboard_log=os.path.join(logs_dir, "PPO_tensorboard"))
    # model_a2c = A2C("MlpPolicy", env, n_steps=2048, verbose=1, tensorboard_log=os.path.join(logs_dir, "A2C_tensorboard"))
    # model_ddpg = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(logs_dir, "DDPG_tensorboard"), action_noise=action_noise)
    # 加载模型用的代码
    # model = PPO.load(f'{model_dir}/1470000.zip', env, verbose=1, tensorboard_log=logs_dir)

    time_steps = 10240
    iters = 0
    # 加载模型用的代码
    # iters = 148
    while True:
        print(f'On iteration:{iters}')
        model_ppo.learn(total_timesteps=time_steps, tb_log_name='PPO', reset_num_timesteps=False)
        # model_a2c.learn(total_timesteps=time_steps, tb_log_name='A2C', reset_num_timesteps=False)
        # model_ddpg.learn(total_timesteps=time_steps, tb_log_name='DDPG', reset_num_timesteps=False)
        model_ppo.save(f'{model_dir}/PPO/{time_steps * iters}')
        # model_a2c.save(f'{model_dir}/A2C/{time_steps * iters}')
        # model_ddpg.save(f'{model_dir}/DDPG/{time_steps * iters}')
        iters += 1


if __name__ == '__main__':
    train()
