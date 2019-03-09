import pickle
file = open('/Users/danielmiskell/mlpgp/mlpgroup009/baseline_experiments/lunar_continuous_ppo.pkl','rb')
with file as handle:
    thing = pickle.load(handle)

print(thing)