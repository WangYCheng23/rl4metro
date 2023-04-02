# 获取参数
import time
from config import get_args, get_env_args
from env import env_agent_config
from train import train
from test import test
from plot import plot_rewards, plot_energy

cfg = get_args()
params = get_env_args(cfg)
# 训练
env, agent = env_agent_config(params)
t1 = time.time()
res_dic = train(cfg, env, agent)
print(time.time()-t1)

# plot_rewards(res_dic['rewards'], cfg, tag="train")
# 测试
# res_dic = test(cfg, env, agent)
#plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
# plot_energy(cfg)
