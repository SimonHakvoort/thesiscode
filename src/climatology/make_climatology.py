from src.climatology.climatology import Climatology
from src.loading_data.get_data import load_cv_data


features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

folds = [0,1,2,3]

folder = '/net/pc200239/nobackup/users/hakvoort/models/climatology/'

# For each fold we make a climatology based on the training data
for fold in folds:
    train_data, test_data, data_info = load_cv_data(fold, features_names_dict)

    train_data = train_data.batch(len(train_data))

    climatology = Climatology(train_data)

    climatology.save(folder + 'climatology_cv_' + str(fold))
    
    print('Cimatology of fold ', fold, ' saved!')


    