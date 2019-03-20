import gym
import tensorflow as tf
import os

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.run import get_learn_function, get_env_type, get_learn_function_defaults, build_env

import time

from types import SimpleNamespace

# Set the save path for this model
save_path = os.path.basename(__file__) + '.' + str(time.time()).replace('.', '')[-6:]

# The model path we want to load from
model_load_path = './models/biphard'

# Write all the arguments into a dictionary that we can references e.g. args.env
args_dict={
    'alg': 'ppo2',
    'env': 'BipedalWalker-v2',
    'second_env': 'BipedalWalkerHardcore-v2',
    'network': 'mlp',
    'learning_rate': 0.001,
    'discount_factor':0.99,
    'nminibatches': 64,
    'cliprange': 0.2,
    'total_timesteps': 1e7,
    'num_env': 1,
    'nsteps': 20480,
    'noptepochs': 10,
    'save_interval': 20,
    'log_interval': 1,
    'save_path': save_path,
    'model_load_path': model_load_path,
    'seed': 0,
    'reward_scale': 1,
    'flatten_dict_observations': True,
}
args = SimpleNamespace(**args_dict)

second_env_args_dict={
    'alg': 'ppo2',
    'env': 'BipedalWalkerHardcore-v2',
    'network': 'mlp',
    'learning_rate': 0.001,
    'discount_factor':0.99,
    'nminibatches': 64,
    'cliprange': 0.2,
    'total_timesteps': 1e7,
    'num_env': 1,
    'nsteps': 20480,
    'noptepochs': 10,
    'save_interval': 20,
    'log_interval': 1,
    'save_path': save_path,
    'model_load_path': model_load_path,
    'seed': 0,
    'reward_scale': 1,
    'flatten_dict_observations': True
}
second_env_args = SimpleNamespace(**second_env_args_dict)


# Prepare the environment and learning algorithm
env_type, env_id = get_env_type(args.env)
learn = get_learn_function(args.alg)
alg_kwargs = get_learn_function_defaults(args.alg, env_type)
env = build_env(args)

# Prepare the second environment if needed
second_env = build_env(second_env_args)

alg_kwargs['network'] = args.network

# The path we will store the results of this experiment
full_path = args.save_path + '/' + args.env + '-' + args.alg

# Make folders that we will store the checkpoints, models and epoch results
if not os.path.exists(full_path):
    os.makedirs(full_path)
    os.makedirs(full_path + '/checkpoints')

print("About to start learning model")

model = learn(
    env=env,
    second_env=second_env,
    seed=args.seed,
    total_timesteps=args.total_timesteps,
    save_interval=args.save_interval,
    lr=args.learning_rate,
    noptepochs = args.noptepochs,
    nsteps = args.nsteps,
    log_interval = args.log_interval,
    save_path = full_path,
    model_load_path = args.model_load_path,
    transfer_weights = args.transfer_weights,
    **alg_kwargs
)

# Save the model and variables
print("Attempting to save model")
model.save(full_path + '/final')