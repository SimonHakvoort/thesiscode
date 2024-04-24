import numpy as np
from datetime import datetime, timedelta
from src.loading_data.station import Station
import tensorflow as tf


class Forecast:
    """
    The Forecast class represents a forecast. It contains the day and time that the forecast is made, the lead time and the wind speed.
    It can optionally contain other variables. These other variables and the wind speed are stored as numpy arrays.
    We can add observations to the forecast, which are stored in a dictionary with the station code as key and the observation and date_time as value.
    """
    def __init__(self, date, initial_time, lead_time, u_wind10, v_wind10, **kwargs):
        """
        Creates a Forecast instance.

        Arguments:
        - date: the date of  the forecast, which is a string of the form 'YYYY-MM-DD'. It will be converted to a datetime object.
        - initial_time: the time of the forecast, which is a string of the form 'HHMM'. It will be converted to a datetime object.
        """
        year, month, day = map(int, date.split('-'))
        hour, minute = divmod(int(initial_time), 100)
        self.date = datetime(year, month, day)
        self.initial_time = datetime(year, month, day, hour, minute)
        self.lead_time = timedelta(hours=int(lead_time))
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.wind_speed = np.sqrt(u_wind10 ** 2 + v_wind10 ** 2)

        self.observations = {}
        self.ignore = ['observations', 'date', 'initial_time', 'lead_time', 'ignore']

    def add_observation(self, station_code, observation, date_time):
        # add the observation and date_time as tuple to the dictionary
        self.observations[station_code] = (observation, date_time)

    def neighbourhood_variance(self, gridcell, size):
        # calculate the variance of the wind speed in a neighbourhood of size x size around the gridcell
        # gridcell is a tuple of the form (x, y)
        # size is an odd number
        x, y = gridcell
        half = size // 2
        wind_speeds = self.wind_speed[x - half:x + half + 1, y - half:y + half + 1]
        return np.var(wind_speeds)

    def generate_sample(self, station, variables_names, neighbourhood_size = None):
        """
        Method to generate a sample for a station. The sample consists of the values of the variables at the gridcell of the station and the observation at the station.
        The features are ordered in the same order as the variables_names list.

        Arguments:
        - station: Station instance
        - variables_names: list of strings
        - neighbourhood_size: int

        Returns:
        - X: numpy array
        - y: float
        """
        i, j = station.gridcell

        X = []
        y = self.observations[station.code][0]



        for key in variables_names:
            if key == 'spatial_variance':
                if neighbourhood_size == None:
                    raise ValueError('Neighbourhood size must be specified when using spatial variance as a predictor')
                X.append(self.neighbourhood_variance((i, j), neighbourhood_size))
            else:
                X.append(getattr(self, key)[i, j])

        return np.array(X), y
    
    def get_predictors(self):
        # get the predictors for the forecast
        predictors = []
        for key, value in self.__dict__.items():
            if key not in self.ignore:
                predictors.append(value)
        return predictors
    
    def has_observations(self):
        # check if the forecast has observations
        return len(self.observations) > 0
    
    def get_grid_variable(self, station, variable_name, grid_size):
        i, j = station.gridcell
        half = grid_size // 2
        variable = getattr(self, variable_name)
        return variable[i - half:i + half + 1, j - half:j + half + 1]
    
    def get_sample_grid(self, station, parameter_names):
        # parameter names is a dict, with as key the parameter name and as value the grid size. If grid size = 1 or 0 or None, the value at the gridcell is returned

        X = []
        y = self.observations[station.code][0]
        for key, value in parameter_names.items():
            if key == 'spatial_variance':
                if value == None:
                    raise ValueError('Neighbourhood size must be specified when using spatial variance as a predictor')
                X.append(self.neighbourhood_variance(station.gridcell, value))
            else:
                if value == 0 or value == 1 or value == None:
                    X.append(getattr(self, key)[station.gridcell])
                else:
                    X.append(self.get_grid_variable(station, key, value))

        return X, y
    
    def generate_all_samples_grid(self, station_info, parameter_names, station_ignore = []):
        X = {}
        y = []
        for station in station_info.values():
            if station.code in self.observations and station.code not in station_ignore:
                X[station.code], observation = self.get_sample_grid(station, parameter_names)
                y.append(observation)
        
        return X, y
    
    def generate_ForecastSample(self, station_info, variable_names, ignore = []):
        samples = []
        for station in station_info.values():
            if station.code in self.observations and station.code not in ignore:
                sample = ForecastSample(variable_names, station.code)
                for variable_name, grid_size in variable_names.items():
                    if grid_size == 0 or grid_size == 1 or grid_size == None:
                        sample.add_feature(variable_name, getattr(self, variable_name)[station.gridcell])
                    else:
                        sample.add_feature(variable_name, self.get_grid_variable(station, variable_name, grid_size))
                sample.add_y(self.observations[station.code][0])
                samples.append(sample)
        return samples
        

        

    def generate_all_samples(self, station_info, variable_names, station_ignore = [], neighbourhood_size = None):
        # generate samples for all stations in station_info
        # station_info is a dictionary with station codes as keys and Station instances as values
        X = []
        y = []
        for station in station_info.values():
            if station.code in self.observations and station.code not in station_ignore:
                x, observation = self.generate_sample(station, variable_names, neighbourhood_size)
                X.append(x)
                y.append(observation)

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)
    
    def __contains__(self, station_code):
        """
        Checks if the forecast contains an observation for the given station code.

        Args: 
        - station_code: string

        Returns:
        - boolean
        """
        return station_code in self.observations


class ForecastSample():
    def __init__(self, feature_names, station_code):
        self.feature_names = feature_names
        self.station_code = station_code
        # make attribute for each feature name
        for feature_name in feature_names:
            setattr(self, feature_name, None)
        self.y = None

    def add_feature(self, feature_name, value):
        setattr(self, feature_name, value)
        if type(value) == np.ndarray:
            # add a boolean attribute for each feature name that is True if the feature is a grid
            setattr(self, feature_name + '_grid', True)
        else:
            setattr(self, feature_name + '_grid', False)

    def add_y(self, y):
        self.y = y

    def get_tensor(self):
        return tf.convert_to_tensor([getattr(self, feature_name) for feature_name in self.feature_names], dtype=tf.float32), tf.convert_to_tensor(self.y, dtype=tf.float32)
    
    def check_if_everything_is_set(self):
        for feature_name in self.feature_names:
            if getattr(self, feature_name) is None:
                return False
        return self.y is not None
    
    def get_X(self):
        if not self.check_if_everything_is_set():
            raise ValueError('Not all features are set')
        # make a dictionary with the feature names as keys and the values as tensors. In case of a grid, the name of the key is the feature name + '_grid'
        X = {}
        for feature_name in self.feature_names:
            if getattr(self, feature_name + '_grid'):
                X[feature_name + '_grid'] = tf.convert_to_tensor(getattr(self, feature_name), dtype=tf.float32)
            else:
                X[feature_name] = tf.convert_to_tensor(getattr(self, feature_name), dtype=tf.float32)

        if 'wind_speed' in X:
            X['wind_speed_forecast'] = X['wind_speed']
        elif 'wind_speed_grid' in X:
            grid = X['wind_speed_grid']
            central_value = grid[grid.shape[0] // 2, grid.shape[1] // 2]
            X['wind_speed_forecast'] = central_value
        return X
     
    def get_y(self):
        return tf.convert_to_tensor(self.y, dtype=tf.float32)
    


