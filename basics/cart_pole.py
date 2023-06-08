import gymnasium as gym

# env = gym.make("CartPole-v1", render_mode="human")

#mujoco environment.
env = gym.make("InvertedPendulum-v4",render_mode="human")
# env = gym.make("KukaEnv-v0", render_mode = "human")

observation, info = env.reset(seed=42)
print ("observation", observation, "info", info)
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
