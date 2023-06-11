import gymnasium as gym
import kuka_env_example
# import pybullet_envs
import time

env = gym.make('iiwaEnv-v0')
# env = gym.make('KukaBulletEnv-v0')
observation = env.reset()
print("**************************************************")
print ("observation", observation)
print("action space", env.action_space)
print("observation_space", env.observation_space)
# env.render()
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation = env.reset()
    # time.sleep(0.01)

env.close()