import os
import numpy as np
import tensorflow as tf
import pickle

from typing import Dict, Tuple
from src.linreg_emos.get_data import get_fold_i
from src.loading_data.station import get_station_info

def get_tf_data(fold, 
                feature_names: dict,
                ignore: list[str] = [],
                add_emos: bool = True,
                features_emos_mean: tf.Tensor = None, 
                features_emos_std: tf.Tensor = None, 
                normalize_features: bool = True, 
                features_1d_mean: tf.Tensor = None, 
                features_1d_std: tf.Tensor = None) -> dict:
    """
    Generates a TensorFlow Dataset from given forecast data. 

    The output is a dictionary with the following keys:
        'features_1d_mean' (tf.Tensor): the mean of the features except the wind speed.
        'features_1d_std' (tf.Tensor): the std of the features except the wind speed.
        'features_emos_mean' (tf.Tensor): the mean of all of the features (including wind speed).
        'features_emos_std' (tf.Tensor): the std of all of the features (including wind speed).
        'data' (tf.data.Dataset): the data.
        'ignore' (list[str]): the codes of the station names that are ignored.
        'feature_names' (dict): the feature names (the value is should be 1 for all the features, except for wind_speed, then it contains the grid size).

    Each element in data consists of two parts, X and y. X is a dictionary containing multiple features, which can be used for training neural networks and
    linear regression. The value y is a tf.Tensor, containing the observation.

    The following keys are contained in the dictionary X:
        'features_emos' (tf.Tensor): a Tensor containing the features from 'feature_names', each is a real valued number.
        'features_1d' (tf.Tensor): a Tensor containing the features from 'feature_names' except for the wind speed.
        'wind_speed_grid' (tf.Tensor): a grid of the wind speeds, where the size is determined by the value in feature_names. The observation station is in the central grid point.
        'wind_speed_forecast' (tf.Tensor): the forecasted wind speed, which is never normalized.


    Parameters:
    fold (int or iterable): Specifies the fold(s) of data to include in the dataset.
    feature_names (dict): Dictionary mapping feature names to their dimensions (should be 1 for all features except for wind speed).
    ignore (list, optional): List of stations to ignore. Defaults to an empty list.
    add_emos (bool, optional): Whether to add the key 'features_emos'. Defaults to True.
    features_emos_mean (Tensor, optional): Mean of 'features_emos'. Used for normalization.
    features_emos_std (Tensor, optional): Standard deviation of 'features_emos. Used for normalization.
    normalize_features (bool, optional): Whether to normalize 'features_1d'. If False, no normalization is performed.
    features_1d_mean (Tensor, optional): Mean of 'features_1d'. Used for normalization.
    features_1d_std (Tensor, optional): Standard deviation of 'features_1d'. Used for normalization.

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
                # Makes for each observation in the Forecast a ForecastSample and puts them in a list.
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
        # Add a key 'features_emos' to X that contains all the features_names that are 1D, and the center grid cell of wind_speed_grid
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


    # To ensure that the wind_speed_grid is a 3D tensor, where the final dimension is for the number of channels (which is equal to 1 in our case).
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


def load_train_test_data(cv: int, feature_names: dict, ignore: list = []) -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Loads the train and test set for cross-validation. 
    Normalizes the training data, and then normalizes the test data with the mean and std from the training data.

    Arguments:
        cv (int): fold to load. 1, 2 and 3 are for validation, fold 0 contains the test set.
        feature_names (dict): dictionary with names of the features that are loaded. As key they should contain the grid size.
        ignore (list, optional): list of station IDs that are ignored.

    Returns:
        train_data (tf.data.Dataset): the training data.
        test_data (tf.data.Dataset): the test data.
        data_info (dict): a dictionary containing the mean and std of the training data.
    """

    # decide which folds to use for training and which for testing.
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
    
    # load the traninig data.
    train_data_dict = get_tf_data(train_folds, feature_names, ignore=ignore, add_emos=True, normalize_features=True)

    # load the test data, which we normalize with the mean and std from the training data.
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

def save_cv_data(feature_names: dict, ignore: list = []) -> None:
    """
    Saves the data of all folds, such that they can be easily loaded back in.

    Arguments:
        feature_names (dict): a dict with 'wind_speed' as key and the corresponding grid size as value.
        ignore (list, optional): list of stations to ignore.

    Returns:
        None.
    """

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
        

def load_cv_data(cv: int, feature_names: dict) -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Loads a specific fold, for a specific grid size for the wind speeds.
    This can be used after save_cv_data has been used.
    
    Arguments:
        cv (int): 0, 1, 2 or 3, which specifies the fold that is used for testing.
        feature_names (dict): a dict with 'wind_speed' as key and the corresponding grid size as value.

    Returns:
        The training data, the test data and addtional info about the fold (mean, std of the features) (Tuple[tf.data.Dataset, tf.data.Dataset, dict]).
    """
    filepath = '/net/pc200239/nobackup/users/hakvoort/cv_data/' + str(feature_names['wind_speed']) + '/' + str(cv) + '/'
    train_data_filepath = filepath + 'train_data'
    test_data_filepath = filepath + 'test_data'
    data_info_filepath = filepath + 'data_info'

    train_data = tf.data.Dataset.load(train_data_filepath)
    test_data = tf.data.Dataset.load(test_data_filepath)
    with open(data_info_filepath, 'rb') as f:
        data_info = pickle.load(f)
    
    return train_data, test_data, data_info


def stack_1d_features(features: dict, label: tf.Tensor) -> Tuple[dict, tf.Tensor]:
    """
    This function makes from all the features that have a 1-dimensional shape a single tf.Tensor.
    It only skips the features 'station_code' and 'wind_speed_forecast'.

    Arguments:
        features (dict): the features (X) of the tf.data.Dataset.
        label (tf.Tensor): the observed wind speed (y).

    Returns:
        features (dict), where all the features that are 1-dimensional stacked, and the corresponding keys removed. The rest of the dictionary is not changed.
        label (tf.Tensor), the observed wind speed.
    """
    # get a list of the keys for the features that are removed.
    feature_names_1d = [key for key, value in features.items() if value.shape == () and key != 'wind_speed_forecast' and key != 'station_code']

    # stack the features and add them under a new key, 'features_1d'
    features_1d = [features[key] for key in feature_names_1d]
    features_1d = tf.stack(features_1d, axis=0)
    features['features_1d'] = features_1d

    # remove the 1d features since they are now contained in 'features_1d'
    for key in feature_names_1d:
        features.pop(key)

    return features, label

def normalize_1d_features(dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.Tensor, tf.Tensor]:
    """
    Normalizes the key 'features_1d' in the given dataset, by computing the mean and standard deviation.

    Args:
        dataset (tf.data.Dataset): The dataset containing the features to be normalized.

    Returns:
        tf.data.Dataset: The normalized dataset.
        mean (tf.Tensor): The mean value used for normalization.
        std (tf.Tensor): The standard deviation value used for normalization.
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

def normalize_1d_features_with_mean_std(dataset: tf.data.Dataset, mean: tf.Tensor, std: tf.Tensor):
    """
    Normalizes the key 'features_1d' in the given dataset using the provided mean and standard deviation.

    Args:
        dataset (tf.data.Dataset): The dataset containing the features to be normalized.
        mean (tf.Tensor): The mean value used for normalization.
        std (tf.Tensor): The standard deviation value used for normalization.

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
    
    # We the construct seperate tf.data.Datasets for each bucket, and add a weight to it.
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


def get_fold_is(features_names_dict: Dict, cv: int, factors: dict, batch_size: int, get_info: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the training and testing datasets with and without importance sampling for a given fold.

    Args:
        features_names_dict (Dict): A dictionary of feature names.
        cv (int): The fold number for cross-validation (indicates which fold is used for testing).
        factors (dict): Factors used for importance sampling.
        batch_size (int): The batch size for the datasets.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: The standard training dataset,
        the importance sampling training dataset, and the testing dataset.
    """

    train_data, test_data, data_info = load_cv_data(cv, features_names_dict)

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


    


            



