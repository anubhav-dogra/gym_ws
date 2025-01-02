
import gymnasium as gym
import kuka_env_example
import time
import os
import numpy as np
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


env = gym.make('iiwaEnvPos-v0')
# vec_env = make_vec_env('iiwaEnvPos-v0', n_envs=1)
observation = env.reset()
# print(observation)
# env.render()

model_dir = "models/A2C"
logdir = "logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="logs/A2C")
TIMESTEPS=100000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    model.save(f"{model_dir}/{TIMESTEPS*i}")


#%%
# observation, info = env.reset(seed=123)
# print ("observation", observation, "info", info)
# for episode in range(10):
#     observation, info = env.reset()
#     for t in range(1000):  
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             observation, info = env.reset()
#             print("Episode Finished after {} timesteps".format(t+1))
#             break
# env.close()

