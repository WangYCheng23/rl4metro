import argparse
import datetime as dt


def get_args():
    """ 超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='PPO',
                        type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Metro-Train-Env',
                        type=str, help="name of environment")
    parser.add_argument('--train_eps', default=100,
                        type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=1500, type=int,
                        help="episodes of testing")
    parser.add_argument('--max_steps', default=1500, type=int,
                        help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma', default=0.99,
                        type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3,
                        type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4,
                        type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=8000,
                        type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=3000, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--tau', default=1e-2, type=float)
    parser.add_argument('--critic_hidden_dim', default=256, type=int)
    parser.add_argument('--actor_hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cuda',
                        type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--test_mode', default=False, type=int, help="for test")
    parser.add_argument('--log_path', default='./logs')
    args = parser.parse_args([])
    args = {**vars(args)}  # 将args转换为字典
    return args


def get_env_args(args):
    args.update({"station_id":[1,2,3]})
    args.update({"low_num_stations":10})
    args.update({"upper_num_stations": 30})
    args.update({'intervals':300})
    args.update({'stop_time_low':5})
    args.update({'stop_time_upper': 100})
    # 巡航速度平均30m/s
    args.update({'cruise_speed_low':30})
    args.update({'cruise_speed_upper':40})
    args.update({'num_actions': 2})
    args.update({'num_observations': 5})
    args.update({'first_metro_time': dt.datetime.strptime(
        '2023-02-08 07:00:00', '%Y-%m-%d %H:%M:%S')})
    args.update({'last_metro_time': dt.datetime.strptime(
        '2023-02-08 22:00:00', '%Y-%m-%d %H:%M:%S')})
    # 站间距离 m
    args.update({'distance_low': 4000})
    args.update({'distance_high': 5000})
    args.update({'num_metros':36})
    # args.update({'colors':['#7B68EE', '#87CEFA', '#DA70D6', '#9370DB', '#FF69B4', '#6A5ACD', '#9400D3', '#00008B', '#F8F8FF',
    #                 '#8A2BE2', '#483D8B', '#6495ED', '#FF00FF', '#0000CD', '#EE82EE', '#DB7093', '#BA55D3', '#DC143C',
    #                 '#000080', '#9932CC', '#D8BFD8', '#FFB6C1', '#C71585', '#DDA0DD', '#191970', '#4169E1', '#F0F8FF',
    #                 '#FFF0F5', '#B0C4DE', '#800080', '#FF1493', '#E6E6FA', '#0000FF', '#4682B4', '#8B008B', '#4B0082',
    #                 '#87CEEB']})
    # 地铁电机牵引功率 Kw
    args.update({'pr':190})
    # 牵引能耗能耗
    args.update({'fb1':0.98})
    args.update({'fb2':0.98})
    # 加速临界速度
    args.update({'v1':25})
    # 减速临界速度
    args.update({'v2':5})  
    # 牵引参数 m/s2
    args.update({'a1':0.95})
    args.update({'b1':0.65})
    # 制动参数 m/s2
    args.update({'a2': 0.75})
    args.update({'b2': 0.92})

    # # 打印参数
    # print("训练参数如下：")
    # print(''.join(['=']*80))
    # tplt = "{:^20}\t{:^20}\t{:^20}"
    # print(tplt.format("参数名", "参数值", "参数类型"))
    # for k, v in args.items():
    #     print(tplt.format(k, v, str(type(v))))
    # print(''.join(['=']*80))
    return args
