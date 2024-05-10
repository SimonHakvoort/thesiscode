import tensorflow as tf

import pdb

from src.models.get_data import get_fold_i, get_station_info

def get_tf_data(fold, feature_names, ignore = [], add_emos = False):
    """
    Gets a specific fold number and feature names and returns the data as a tf.data.Dataset

    Args:
    - fold: int or list of ints
    - feature_names: dictionary with as key the feature name and as value the grid size. If grid size = 1 or 0 or None, the value at the gridcell is returned
    - ignore: list of strings of station codes to ignore
    """
    fold_list = []
    if type(fold) == int:
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

    if add_emos:
        # Add a key 'features_emos' to X that contains all the features_names that are 1D, and the centre grid cell of wind_speed_grid
        temp = {}
        for feature in feature_names:
            if feature_names[feature] == 1:
                temp[feature] = X[feature]
            else:
                name = feature + '_grid'
                temp[feature] = X[name][:, feature_names[feature] // 2, feature_names[feature] // 2]
        X['features_emos'] = tf.stack([temp[feature] for feature in temp], axis=1)

        mean = tf.reduce_mean(X['features_emos'], axis=0)
        std = tf.math.reduce_std(X['features_emos'], axis=0)

        X['features_emos'] = (X['features_emos'] - mean) / std

    # To ensure that the wind_speed_grid is a 3D tensor, where the final dimension is for the number of channels.
    if 'wind_speed_grid' in X:
        X['wind_speed_grid'] = tf.expand_dims(X['wind_speed_grid'], axis=-1)



    data = tf.data.Dataset.from_tensor_slices((X, y))

    return data


def stack_1d_features(features, label):
    
    feature_names_1d = [key for key, value in features.items() if value.shape == () and key != 'wind_speed_forecast']
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



            



