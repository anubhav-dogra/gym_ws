from gymnasium.envs.registration import register
# from gym.envs.registration import register

register(
    id='iiwaEnv-v0',
    entry_point='kuka_env_example.envs:iiwaEnv',
    max_episode_steps=1000,
)
register(
    id='iiwaEnvPos-v0',
    entry_point='kuka_env_example.envs:iiwaEnvPos',
    max_episode_steps=2000,

)
