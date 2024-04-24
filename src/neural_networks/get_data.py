import tensorflow as tf

import pdb

from models.get_data import get_fold_i, get_station_info

def get_tf_data(fold, feature_names, ignore = []):
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
    # dataset is a tf.data.Dataset. We normalize the value from the key 'features_1d' in the dataset
    # by subtracting the mean and dividing by the standard deviation
    # The mean and standard deviation are computed over the entire dataset

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
    def normalize(x, y):
        x['features_1d'] = (x['features_1d'] - mean) / std
        return x, y
    
    return dataset.map(normalize)



            



