# %%
from __future__ import annotations
import gymnasium as gym
import kuka_env_example
# import pybullet_envs
import time
import os
from stable_baselines3 import A2C

#  %%

env = gym.make('iiwaEnv-v0')
# env = gym.make('KukaBulletEnv-v0')
observation = env.reset()
print("**************************************************")
print ("observation", observation)
print("action space", env.action_space)
print("observation_space", env.observation_space)
# env.render()
# check_env(env=env)
# time.sleep(10)
env.close()
# %%
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30_000, progress_bar=True)
vec_env = model.get_env()
obs = vec_env.reset()


# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("rgb-array")

# for i in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         observation = env.reset()
    # time.sleep(0.01)

