import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import time

env = gymnasium.make("LunarLander-v2", render_mode="rgb_array")
model = DQN.load("dqn_lunar", env=env)
model.load("dqn_lunar")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")