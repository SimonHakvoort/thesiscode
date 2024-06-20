import os
import tensorflow as tf
import pickle

from src.models.get_data import get_fold_i, get_station_info

def get_tf_data(fold, feature_names, ignore = [], add_emos=True, features_emos_mean = None, features_emos_std = None, normalize_features = True, features_1d_mean = None, features_1d_std = None):
    """
    Generates a TensorFlow Dataset from given forecast data.

    Parameters:
    fold (int or iterable): Specifies the fold(s) of data to include in the dataset.
    feature_names (dict): Dictionary mapping feature names to their dimensions.
    ignore (list, optional): List of features to ignore. Defaults to an empty list.
    add_emos (bool, optional): Whether to add emotion-related features. Defaults to True.
    features_emos_mean (Tensor, optional): Mean of emotion-related features. Used for normalization.
    features_emos_std (Tensor, optional): Standard deviation of emotion-related features. Used for normalization.
    normalize_features (bool, optional): Whether to normalize 1D features. If None, no normalization is performed.
    features_1d_mean (Tensor, optional): Mean of 1D features. Used for normalization.
    features_1d_std (Tensor, optional): Standard deviation of 1D features. Used for normalization.

    Returns:
    dict: A dictionary containing the generated TensorFlow Dataset and additional information such as feature names, ignored features, and normalization parameters (if applicable).
    """
    fold_list = []
    if isinstance(fold, int):
        fold_list.append(get_fold_i(fold))
    else:
        for i in fold:
            fold_list.append(get_fold_i(i))
    station_info = get_station_info()

    X_dicts = []
    y_list = []
    
    for forecasts in fold_list:
        for forecast in forecasts:
            if forecast.has_observations():
                samples = forecast.generate_ForecastSample(station_info, feature_names, ignore = ignore)
                for sample in samples:
                    X_dicts.append(sample.get_X())
                    y_list.append(sample.get_y())

    X = {key: [] for key in X_dicts[0]}

    for x_dict in X_dicts:
        for key, value in x_dict.items():
            X[key].append(value)

    X = {key: tf.convert_to_tensor(value) for key, value in X.items()}
    y = tf.convert_to_tensor(y_list)

    if add_emos is True:        
        # Add a key 'features_emos' to X that contains all the features_names that are 1D, and the centre grid cell of wind_speed_grid
        temp = {}
        for feature in feature_names:
            if feature_names[feature] == 1:
                temp[feature] = X[feature]
            else:
                name = feature + '_grid'
                temp[feature] = X[name][:, feature_names[feature] // 2, feature_names[feature] // 2]
        X['features_emos'] = tf.stack([temp[feature] for feature in temp], axis=1)

        if features_emos_mean is None or features_emos_std is None: 
            features_emos_mean = tf.reduce_mean(X['features_emos'], axis=0)
            features_emos_std = tf.math.reduce_std(X['features_emos'], axis=0)            

        X['features_emos'] = (X['features_emos'] - features_emos_mean) / features_emos_std


    # To ensure that the wind_speed_grid is a 3D tensor, where the final dimension is for the number of channels.
    if 'wind_speed_grid' in X:
        X['wind_speed_grid'] = tf.expand_dims(X['wind_speed_grid'], axis=-1)


    data = tf.data.Dataset.from_tensor_slices((X, y))

    data = data.map(lambda x, y: stack_1d_features(x, y))

    output = {}

    if normalize_features is True:
        if features_1d_mean is not None and features_1d_std is not None:
            data = normalize_1d_features_with_mean_std(data, features_1d_mean, features_1d_std)
        else:
            data, features_1d_mean, features_1d_std = normalize_1d_features(data)
        
        output['features_1d_mean'] = features_1d_mean
        output['features_1d_std'] = features_1d_std

    output['data'] = data
    output['ignore'] = ignore
    output['feature_names'] = feature_names

    if add_emos:
        output['features_emos_mean'] = features_emos_mean
        output['features_emos_std'] = features_emos_std


    return output


def load_train_test_data(cv, feature_names, ignore = []):
    if cv == 1:
        train_folds = [2,3]
        test_folds = [1]
    elif cv == 2:
        train_folds = [1,3]
        test_folds = [2]
    elif cv == 3:
        train_folds = [1,2]
        test_folds = [3]
    elif cv == 0:
        train_folds = [1,2,3]
        test_folds = [0]
    else:
        raise ValueError('Invalid value for cv')
    
    train_data_dict = get_tf_data(train_folds, feature_names, ignore=ignore, add_emos=True, normalize_features=True)
    test_data_dict = get_tf_data(test_folds, 
                                 feature_names, 
                                 ignore=ignore, 
                                 add_emos=True, 
                                 features_emos_mean=train_data_dict['features_emos_mean'],
                                 features_emos_std=train_data_dict['features_emos_std'],
                                 normalize_features=True,
                                 features_1d_mean=train_data_dict['features_1d_mean'],
                                 features_1d_std=train_data_dict['features_1d_std'])
    
    train_data = train_data_dict['data']
    test_data = test_data_dict['data']

    extra_info = {
        'features_emos_mean': train_data_dict['features_emos_mean'],
        'features_emos_std': train_data_dict['features_emos_std'],
        'features_1d_mean': train_data_dict['features_1d_mean'],
        'features_1d_std': train_data_dict['features_1d_std']
    }

    return train_data, test_data, extra_info

def save_cv_data(feature_names, ignore = []):
    # save the data in /net/pc200239/nobackup/users/hakvoort/cv_data where there is a folder containing the grid size of wind speed forecast, and in that folder there are the files for the different folds.
    # The files should contain the data in the format of the output of load_train_test_data

    for fold in [0,1,2,3]:
        train_data, test_data, data_info = load_train_test_data(fold, feature_names, ignore = ignore)
        # make a folder 
        filepath = '/net/pc200239/nobackup/users/hakvoort/cv_data/'  + str(feature_names['wind_speed']) + '/' + str(fold) + '/'
        # create the folder
        os.makedirs(filepath, exist_ok=True)

        train_data_filepath = filepath + 'train_data'
        test_data_filepath = filepath + 'test_data'
        data_info_filepath = filepath + 'data_info'
        train_data.save(train_data_filepath)
        test_data.save(test_data_filepath)
        with open(data_info_filepath, 'wb') as f:
            pickle.dump(data_info, f)
        

def load_cv_data(cv, feature_names):
    filepath = '/net/pc200239/nobackup/users/hakvoort/cv_data/' + str(feature_names['wind_speed']) + '/' + str(cv) + '/'
    train_data_filepath = filepath + 'train_data'
    test_data_filepath = filepath + 'test_data'
    data_info_filepath = filepath + 'data_info'

    train_data = tf.data.Dataset.load(train_data_filepath)
    test_data = tf.data.Dataset.load(test_data_filepath)
    with open(data_info_filepath, 'rb') as f:
        data_info = pickle.load(f)
    
    return train_data, test_data, data_info

        


def stack_1d_features(features, label):
    
    feature_names_1d = [key for key, value in features.items() if value.shape == () and key != 'wind_speed_forecast' and key != 'station_code']
    features_1d = [features[key] for key in feature_names_1d]
    features_1d = tf.stack(features_1d, axis=0)
    features['features_1d'] = features_1d

    # remove the 1d features
    for key in feature_names_1d:
        features.pop(key)

    return features, label

def normalize_1d_features(dataset):
    """
    Normalizes the key 'features_1d' in the given dataset.

    Args:
        dataset (tf.data.Dataset): The dataset containing the features to be normalized.

    Returns:
        tf.data.Dataset: The normalized dataset.
        float: The mean value used for normalization.
        float: The standard deviation value used for normalization.
    """
    
    mean = 0

    for x, y in dataset:
        mean += x['features_1d']

    mean /= len(dataset)

    std = 0

    for x, y in dataset:
        std += (x['features_1d'] - mean)**2

    std = tf.sqrt(std / len(dataset))

    def normalize(x, y):
        x['features_1d'] = (x['features_1d'] - mean) / std
        return x, y
    
    return dataset.map(normalize), mean, std

def normalize_1d_features_with_mean_std(dataset, mean, std):
    """
    Normalizes the key 'features_1d' in the given dataset using the provided mean and standard deviation.

    Args:
        dataset (tf.data.Dataset): The dataset containing the features to be normalized.
        mean (float): The mean value used for normalization.
        std (float): The standard deviation value used for normalization.

    Returns:
        tf.data.Dataset: The normalized dataset.

    """
    
    def normalize(x, y):
        x['features_1d'] = (x['features_1d'] - mean) / std
        return x, y
    
    return dataset.map(normalize)

def make_importance_sampling_dataset(data: tf.data.Dataset) -> tf.data.Dataset:
    def filter_func(X, y, lower, upper):
        return (lower <= y) & (y < upper)
    
    data_less_than_9 = data.filter(lambda X, y: filter_func(X, y, 0, 9))
    data_9_12 = data.filter(lambda X, y: filter_func(X, y, 9, 12))
    data_12_15 = data.filter(lambda X, y: filter_func(X, y, 12, 15))
    data_greater_than_15 = data.filter(lambda X, y: filter_func(X, y, 15, 1000))

    data_9_12 = data_9_12.repeat(4)
    data_12_15 = data_12_15.repeat(8)
    data_greater_than_15 = data_greater_than_15.repeat(12)

    def weight_func(X, y, weight):
        return X, y, tf.constant(weight, dtype=tf.float32)

    data_less_than_9 = data_less_than_9.map(lambda X, y: weight_func(X, y, 1))
    data_9_12 = data_9_12.map(lambda X, y: weight_func(X, y, 1/4))
    data_12_15 = data_12_15.map(lambda X, y: weight_func(X, y, 1/8))
    data_greater_than_15 = data_greater_than_15.map(lambda X, y: weight_func(X, y, 1/12))

    output_data = data_less_than_9.concatenate(data_9_12).concatenate(data_12_15).concatenate(data_greater_than_15)

    return output_data



    


            



