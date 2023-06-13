
import gymnasium as gym
import kuka_env_example
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env 


env = gym.make('iiwaEnvPos-v0')
observation, info = env.reset()
env.render()
model_dir = "models/PPO"
logdir = "logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/PPO")
TIMESTEPS=10000
for i in range(1,10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    model.save(f"{model_dir}/{TIMESTEPS*i}")

env.close()
