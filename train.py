import os
from noise import OUNoise
from tensorboardX import SummaryWriter
import datetime as dt

def train(cfg, env, agent):
    writer = SummaryWriter(os.path.join(cfg['log_path'],dt.datetime.now().strftime("%Y%m%d_%H%M%S")))
    print("开始训练！")
    # ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = [] # 记录所有回合的奖励
    for i_ep in range(cfg['train_eps']):
        state = env.reset()
        # ou_noise.reset()
        ep_reward = 0
        for i_step in range(cfg['max_steps']):
            action = agent.sample_action(state)
            # action = ou_noise.get_action(action, i_step+1) 
            next_state, reward, done, _ = env.step(action)   
            ep_reward += reward
            agent.memory.push((state, action, reward, next_state, done))
            agent.update()
            state = next_state
            if done:
                break
        writer.add_scalar("train_reward", ep_reward, i_ep)
        if (i_ep+1)%10 == 0:
            print(f"回合：{i_ep+1}/{cfg['train_eps']}，奖励：{ep_reward:.2f}")
        rewards.append(ep_reward)
    print("完成训练！")
    return {'rewards':rewards}
