from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("iiwaEnvPos-v0", n_envs=2)