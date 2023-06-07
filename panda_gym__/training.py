import gym as gym
import panda_gym
from stable_baselines3 import DDPG

env = gym.make("PandaReach-v2")
model = DDPG(policy="MultiInputPolicy", env=env)
model.train(30_000)