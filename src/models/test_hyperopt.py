import optuna
import os
# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle as pkl
import numpy as np
from src.models.hyperopt_emos import Objective
import warnings
import logging





study_name = "crps_obj_CRPS_twCRPS12_MOTPE.pkl"
filepath = '/net/pc200239/nobackup/users/hakvoort/hyperopt/hyperopt_emos/'

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

objectives = ['CRPS', 'twCRPS12']

saving = True

if len(objectives) == 1:
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=15)
    if saving:
        study = optuna.create_study(sampler=sampler, direction='minimize')
    else:
        study = optuna.create_study(sampler=sampler, direction='minimize')
else:
    sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)
    directions = ['minimize' for _ in range(len(objectives))]
    if saving:
        study = optuna.create_study(sampler=sampler, directions=directions)
    else:
        study = optuna.create_study(sampler=sampler, directions=directions)


objective = Objective(features_names_dict, objectives, twCRPS=True)

# study.optimize(objective, n_trials=50)

directory = os.path.join(filepath, study_name)

# make study2 where we save it with a storage in filepath, with name study_name

# study2 = optuna.create_study(sampler=sampler, direction='minimize', study_name='my_study', storage=f'sqlite:///{directory}/study.db')

# study2 = optuna.load_study(study_name='my_study', storage=f'sqlite:///{directory}/study.db')

with open(directory, 'wb') as file:
    pkl.dump(study, file)

for i in range(0, 50):
    study.optimize(objective, n_trials=5)

    with open(directory, 'wb') as file:
        pkl.dump(study, file)

        print("Finished iteration ", i)

# with open(directory, 'wb') as file:
#     pkl.dump(study, file)

# print("Finished!")

# best_trials = study.best_trials

# # Print the parameters of the best 5 trials
# for i, trial in enumerate(best_trials):
#     print(f"Best trial {i+1}:")
#     print("  Value: ", trial.value)
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))