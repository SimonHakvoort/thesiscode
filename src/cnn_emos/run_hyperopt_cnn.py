import optuna
from src.cnn_emos.hyperopt_cnn import ObjectiveCNN

filepath = '/net/pc200239/nobackup/users/hakvoort/hyperopt/hyperopt_cnn/'

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

objectives = ['CRPS', 'twCRPS12']

# Select the MOTPE samples
sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)

directions = ['minimize' for _ in range(len(objectives))]

# Create the study.
study = optuna.create_study(sampler=sampler, directions=directions, study_name='study_name', storage=f'sqlite:///{filepath}/study.db')

### This can be used to load an existing study object.
# study = optuna.load_study(study_name='...', storage=f'sqlite:///{filepath}/study.db')

# Create the objective function.
objective = ObjectiveCNN(features_names_dict, objectives, train_amount=2)

study.optimize(objective)