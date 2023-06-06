import gymnasium
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
import time

env = gymnasium.make("LunarLanderContinuous-v2", render_mode="rgb_array")
model = TD3.load("best_model", env=env)
model.load("best_model")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")