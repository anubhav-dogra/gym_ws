import gymnasium as gym
import panda_gym
import numpy as np
import time

env = gym.make("PandaReachDense-v3", render_mode="human")
observations, _ = env.reset()

for _ in range(1000):
    state_id = env.save_state()

    #sample 5 actions and choose the one that yields the best reward
    best_reward = -np.inf
    best_action = None
    for _ in range(5):
        env.restore_state(state_id)
        action = env.action_space.sample()
        observations, reward, _, _, _ = env.step(action)
        if reward > best_reward:
            best_reward = reward
            best_action = action
            print(best_action)
    
    env.restore_state(state_id)
    env.remove_state(state_id) # discard the state, as it is no longer needed
    observation, reward, terminated, truncated, info = env.step(best_action)
    if terminated:
        observation, info = env.reset()


env.close()


