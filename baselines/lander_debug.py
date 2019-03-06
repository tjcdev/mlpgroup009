# baselines.run --alg=ppo2 --env=LunarLander-v2 --num_timesteps=1000 --save_path=baseline_experiments/lunar_ppo.pkl

import gym
from baselines import ppo2

env = gym.make("LunarLander-v2")
act = ppo2.learn(network='mlp', env=env, total_timesteps=1000)