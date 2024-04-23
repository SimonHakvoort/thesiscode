import tensorflow as tf

from models.get_data import get_fold_i, get_station_info

def get_tf_data(fold, feature_names, ignore = []):
    """
    Gets a specific fold number and feature names and returns the data as a tf.data.Dataset

    Args:
    - fold: int
    - feature_names: dictionary with as key the feature name and as value the grid size. If grid size = 1 or 0 or None, the value at the gridcell is returned
    - ignore: list of strings of station codes to ignore
    """
    forecasts = get_fold_i(fold)
    station_info = get_station_info()

    X = []
    y = []

    for forecast in forecasts:
        if forecast.has_observations():
            samples = forecast.generate_ForecastSample(station_info, feature_names, ignore = ignore)
            for sample in samples:
                X.append(sample.get_X())
                y.append(sample.get_y())

    data = tf.data.Dataset.from_tensor_slices((X, y))
    return data

            



