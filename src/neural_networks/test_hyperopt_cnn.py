import optuna
import os
# Set environment variable to suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle as pkl
import numpy as np
from src.neural_networks.hyperopt_cnn import ObjectiveCNN

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

objectives = ['CRPS', 'twCRPS12']

sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)
directions = ['minimize' for _ in range(len(objectives))]
study = optuna.create_study(sampler=sampler, directions=directions) #, study_name='MOTPE_CRPS_CRPS12_less_epochs_final', storage=f'sqlite:///{directory}/study.db')

objective = ObjectiveCNN(features_names_dict, objectives, train_amount=3)

study.optimize(objective)