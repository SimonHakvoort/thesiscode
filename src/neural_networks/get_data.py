import os
import numpy as np
import tensorflow as tf
import pickle

from typing import Dict, Tuple
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

def make_importance_sampling_dataset(data: tf.data.Dataset, factors: dict) -> tf.data.Dataset:
    """
    Implements importance sampling, by upsampling and downweighting from the samples with larger observations.

    Arguments:
        data (tf.data.Dataset): dataset on which importance sampling is performed.
        factors (dict): a dictionary where keys are upper bounds and values are corresponding factors.

    Returns:
        tf.data.Dataset where weights are included.
    """
    def filter_func(X, y, lower, upper):
        """
        Filters the data based on whether y is in between upper and lower.
        """
        return (lower <= y) & (y < upper)
    
    # Sort bounds
    bounds = sorted(factors.keys())

    # Check if the first bound is greater than 0
    if bounds[0] <= 0:
        raise Exception("The smallest bound should be greater than 0!")

    datasets = []
    previous_bound = 0

    for bound in bounds:
        factor = factors[bound]
        filtered_data = data.filter(lambda X, y, lb=previous_bound, ub=bound: filter_func(X, y, lb, ub))
        repeated_data = filtered_data.repeat(factor)
        weighted_data = repeated_data.map(lambda X, y, w=factor: (X, y, tf.constant(1 / w, dtype=tf.float32)))
        datasets.append(weighted_data)
        previous_bound = bound

    # Concatenate all the datasets
    output_data = datasets[0]
    for ds in datasets[1:]:
        output_data = output_data.concatenate(ds)

    return output_data


def get_fold_is(features_names_dict: Dict, fold: int, factors: dict, batch_size: int, get_info: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the training and testing datasets with and without importance sampling for a given fold.

    Args:
        features_names_dict (Dict): A dictionary of feature names.
        fold (int): The fold number for cross-validation.
        factors (dict): Factors used for importance sampling.
        batch_size (int): The batch size for the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: The standard training dataset,
        the importance sampling training dataset, and the testing dataset.
    """

    train_data, test_data, data_info = load_cv_data(fold, features_names_dict)

    def remove_other_info(X, y):
        return {'features_emos': X['features_emos']}, y

    def remove_label(X, y):
        return X['features_emos'], y

    train_data = train_data.map(remove_other_info)

    train_data_is = train_data.map(remove_label)

    train_data_is = make_importance_sampling_dataset(train_data_is, factors)

    data_list = list(train_data_is.as_numpy_iterator())

    features = np.array([x[0] for x in data_list])
    targets = np.array([x[1] for x in data_list])
    sample_weights = np.array([x[2] for x in data_list])

    dataset_dict = {
        'features_emos': features,
        'y': targets,
        'w': sample_weights
    }

    train_data_is = tf.data.Dataset.from_tensor_slices(dataset_dict)

    def correct_map(sample_dict):
        return {'features_emos': sample_dict['features_emos']}, sample_dict['y'], sample_dict['w']

    train_data_is = train_data_is.map(correct_map)

    dataset_length_is = train_data_is.cardinality()

    print(dataset_length_is)

    train_data_is = train_data_is.shuffle(dataset_length_is)

    train_data_is = train_data_is.batch(batch_size)

    train_data_is = train_data_is.prefetch(tf.data.experimental.AUTOTUNE)

    def const_weight_func(X, y):
        """
        Attaches a uniform weight to each sample in the dataset.
        """
        return X, y, tf.constant(1, dtype=tf.float32)

    train_data = train_data.map(const_weight_func)

    dataset_length = train_data.cardinality()

    train_data = train_data.shuffle(dataset_length)

    print(dataset_length)

    train_data = train_data.batch(batch_size)

    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    test_data = test_data.batch(test_data.cardinality())

    if get_info:
       return train_data, train_data_is, test_data, data_info 

    return train_data, train_data_is, test_data


    


            



