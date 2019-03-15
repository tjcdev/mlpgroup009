import gym
import tensorflow as tf
import os

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function, get_env_type, get_learn_function_defaults, build_env

import time

from types import SimpleNamespace

save_path = './' + str(time.time()).replace('.', '')
model_load_path = './15525043569503987/BipedalWalker-v2-ppo2/checkpoints'
# Write all the arguments into a dictionary that we can references e.g. args.env
args_dict={
    'alg': 'ppo2',
    'total_timesteps': 10000,
    'seed': 0,
    'env': 'BipedalWalker-v2',
    'network': 'mlp',
    'num_env': 1,
    'reward_scale': 1,
    'flatten_dict_observations': True,
    'save_interval': 1,
    'num_epochs': 100,
    'steps_per_update': 100,
    'log_interval': 1,
    'save_path': save_path
}
args = SimpleNamespace(**args_dict)

env_type, env_id = get_env_type(args.env)

learn = get_learn_function(args.alg)
alg_kwargs = get_learn_function_defaults(args.alg, env_type)

env = build_env(args)

alg_kwargs['network'] = args.network

# The path we will store the results of this experiment
full_path = args.save_path + '/' + args.env + '-' + args.alg

# Make folders that we will store the checkpoints, models and epoch results
if not os.path.exists(full_path):
    os.makedirs(full_path)
    os.makedirs(full_path + '/checkpoints')

model = learn(
    env=env,
    seed=args.seed,
    total_timesteps=args.total_timesteps,
    save_interval=args.save_interval,
    noptepochs = args.num_epochs,
    nsteps = args.steps_per_update,
    log_interval = args.log_interval,
    save_path = full_path,
    model_load_path = model_load_path,
    **alg_kwargs
)

# Save the model and variables
model.save(full_path + '/final')