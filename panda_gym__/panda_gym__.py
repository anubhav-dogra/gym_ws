import gymnasium as gym
import panda_gym
import time

env = gym.make("PandaReach-v2", render_mode = "human")
observation, info = env.reset()

for _ in range(1000):
  current_position = observation["observation"][0:3]
  desired_position = observation["desired_goal"][0:3]
  action = 5.0*(desired_position-current_position)
  # action = env.action_space.sample()
  observation, rewards, terminated, truncated, info = env.step(action)
  # time.sleep(0.05)
  if terminated or truncated:
    # time.sleep(1)
    observation, info = env.reset()

env.close()