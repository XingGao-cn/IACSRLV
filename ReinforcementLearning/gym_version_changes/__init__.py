# import gym
# env = gym.make("LunarLander-v2", options={})
# env.seed(123)	# seed指定了随机数种子
# observation = env.reset()  # 注意env.reset()只返回observation,无附加信息
#
# done = False
# while not done:
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, done, info = env.step(action)
#
#     env.render(mode="human")
#
# env.close()


import gym
env = gym.make("CartPole-v1")
# env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=123, options={})

done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()
