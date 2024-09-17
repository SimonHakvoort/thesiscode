import optuna
from src.linreg_emos.hyperopt_emos import Objective


filepath = '/net/pc200239/nobackup/users/hakvoort/hyperopt/hyperopt_emos/'

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

objectives = ['CRPS', 'twCRPS12']

saving = True

# Select the MOTPE sampler
sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)

directions = ['minimize' for _ in range(len(objectives))]

# Create the study
study = optuna.create_study(sampler=sampler, directions=directions, study_name='study_name', storage=f'sqlite:///{filepath}/study.db')

### This can be used to load back in an existing study.
# study = optuna.load_study(study_name='...', storage=f'sqlite:///{filepath}/study.db')

objective = Objective(features_names_dict, objectives, twCRPS=True, train_amount=1)

study.optimize(objective)