import gym
import tensorflow as tf

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function, get_env_type, get_learn_function_defaults, build_env

from types import SimpleNamespace

# Write all the arguments into a dictionary that we can references e.g. args.env
args_dict={
    'alg': 'ppo2',
    'total_timesteps': 1e7,
    'seed': 0,
    'env': 'BipedalWalkerHardcore-v2',
    'network': 'mlp',
    'num_env': 1,
    'reward_scale': 1,
    'flatten_dict_observations': True
}
args = SimpleNamespace(**args_dict)

env_type, env_id = get_env_type(args.env)

learn = get_learn_function(args.alg)
alg_kwargs = get_learn_function_defaults(args.alg, env_type)

env = build_env(args)

alg_kwargs['network'] = args.network

model = learn(
    env=env,
    seed=args.seed,
    total_timesteps=args.total_timesteps,
    **alg_kwargs
)

# Save the model and variables
model.save('./baseline_weights/ppo2_biphard')