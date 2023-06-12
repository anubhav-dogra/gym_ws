# %%
import gymnasium as gym
import kuka_env_example
# import pybullet_envs
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#  %%

env = gym.make('iiwaEnv-v0')
# vec_env = make_vec_env("iiwaEnv-v0", n_envs=1)
env.reset()
# env = gym.make('KukaBulletEnv-v0')
# observation = env.reset()
# print("**************************************************")
# print ("observation", observation)
# print("action space", env.action_space)
# print("observation_space", env.observation_space)
# env.render()
# check_env(env=env)
# time.sleep(10)
# env.close()
# %%
model_dir = "models/PPO"
logdir = "logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="PPO")
TIMESTEPS=10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    model.save(f"{model_dir}/{TIMESTEPS*i}")

env.close()
# obs = vec_env.reset()
# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_iiwaEnv")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, terminated, truncated, info = vec_env.step(action)
#     vec_env.render("rgb_array")


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

