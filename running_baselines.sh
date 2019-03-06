# PPO on FetchReach
python -m baselines.run --alg=ppo2 --env=FetchReach-v1 --save_path=baseline_experiments/fetchreach_model_ppo2 --num_timesteps=10000

# HER on FetchReach
python -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=10000 --save_path=baseline_experiments/fetchreach_model_her.pkl

# Bipedal Walker
python -m baselines.run --alg=ppo2 --env=BipedalWalker-v2 --num_timesteps=1000 --save_path=baseline_experiments/walker_ppo2.pkl
