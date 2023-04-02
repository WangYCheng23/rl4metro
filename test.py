def test(cfg, env, agent):
    print("开始测试！")
    rewards = [] # 记录所有回合的奖励
    for i_ep in range(cfg['test_eps']):
        state = env.reset()[0] 
        ep_reward = 0
        for i_step in range(cfg['max_steps']):
            action = agent.predict_action(state)
            next_state, reward, done, trunc, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    print("完成测试！")
    return {'rewards':rewards}