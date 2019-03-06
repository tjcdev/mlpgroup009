# PPO on FetchReach with PPO2
python -m baselines.run --alg=ppo2 --env=FetchReach-v1 --save_path=baseline_experiments/fetchreach_model_ppo2 --num_timesteps=10000

# HER on FetchReach with HER
python -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=10000 --save_path=baseline_experiments/fetchreach_model_her.pkl

# Bipedal Walker with PPO2
python -m baselines.run --alg=ppo2 --env=BipedalWalker-v2 --num_timesteps=1000 --save_path=baseline_experiments/walker_ppo2.pkl

# Bipedal Hardcore Walker with PPO2
python -m baselines.run --alg=ppo2 --env=BipedalWalkerHardcore-v2 --num_timesteps=1000 --save_path=baseline_experiments/walker_hardcore_ppo2.pkl

# Lunar Lander with PPO2
python -m baselines.run --alg=ppo2 --env=LunarLander-v2 --num_timesteps=1000 --save_path=baseline_experiments/lunar_ppo.pkl

# Lunar Lander Continuous with PPO2
python -m baselines.run --alg=ppo2 --env=LunarLanderContinuous-v2 --num_timesteps=1000 --save_path=baseline_experiments/lunar_continuous_ppo.pkl

# Bipedal Walker with TRPO
python -m baselines.run --alg=trpo_mpi --env=BipedalWalker-v2 --num_timesteps=1000 --save_path=baseline_experiments/walker_trpo.pkl

# Bipedal Hardcore Walker with TRPO
python -m baselines.run --alg=trpo_mpi --env=BipedalWalkerHardcore-v2 --num_timesteps=1000 --save_path=baseline_experiments/walker_hardcore_trpo.pkl

# Lunar Lander with TRPO
python -m baselines.run --alg=trpo_mpi --env=LunarLander-v2 --num_timesteps=1000 --save_path=baseline_experiments/lunar_trpo.pkl

# Lunar Lander Continuous with TRPO
python -m baselines.run --alg=trpo_mpi --env=LunarLanderContinuous-v2 --num_timesteps=1000 --save_path=baseline_experiments/lunar_continuous_trpo.pkl
