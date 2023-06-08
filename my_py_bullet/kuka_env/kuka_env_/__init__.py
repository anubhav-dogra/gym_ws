from gymnasium.envs.registration import register

register(
    id='kuka_env/KukaEnv-v0',
    entry_point='kuka_env_.envs:KukaEnv'
)