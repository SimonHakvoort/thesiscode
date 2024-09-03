from typing import Tuple, Union
import numpy as np
from datetime import datetime, timedelta
from src.loading_data.station import Station
import tensorflow as tf


class Forecast:
    """
    The Forecast class represents a forecast. It contains the day and time that the forecast is made, the lead time and the wind speed.
    It can optionally contain other variables. These other variables and the wind speed are stored as 2-dimensional numpy arrays.
    We can add observations to the forecast, which are stored in a dictionary with the station code as key and the observation and date_time as value.
    """
    def __init__(self, date: str, initial_time: str, lead_time: str, u_wind10: np.ndarray, v_wind10: np.ndarray, **kwargs):
        """
        Creates a Forecast instance.

        Arguments:
            date (str): the date of  the forecast, which is a string of the form 'YYYY-MM-DD'. It will be converted to a datetime object.
            initial_time (str): the time of the forecast, which is a string of the form 'HHMM'. It will be converted to a datetime object.
            lead_time (str): lead time of the forecast, which is a string of the form 'HH'. Will be converted to a timedelta object.
            u_wind10 (np.ndarray): U-component of the wind speed, which is an np.ndarray representing the grid.
            v_wind10 (np.ndarray): V-component of the wind speed, which is an np.ndarray representing the grid.
            kwargs: additional weather variables, where the key represents the name of the variable, and the value is an np.ndarray of the grid.
        """
        year, month, day = map(int, date.split('-'))
        hour, minute = divmod(int(initial_time), 100)
        self.date = datetime(year, month, day)
        self.initial_time = datetime(year, month, day, hour, minute)
        self.lead_time = timedelta(hours=int(lead_time))

        # Save each element in kwargs in the class.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Compute the total wind speed.
        self.wind_speed = np.sqrt(u_wind10 ** 2 + v_wind10 ** 2)

        # Dictionary to save the observations. 
        self.observations = {}
        self.ignore = ['observations', 'date', 'initial_time', 'lead_time', 'ignore']

    def add_observation(self, station_code: str, observation: float, date_time: datetime):
        self.observations[station_code] = (observation, date_time)

    def neighbourhood_variance(self, gridcell: Tuple[int,int], size: int):
        """
        Compute the variance of the wind speeds surrounding the gridcell. Size should be an odd integer.
        """
        x, y = gridcell
        half = size // 2
        wind_speeds = self.wind_speed[x - half:x + half + 1, y - half:y + half + 1]
        return np.var(wind_speeds)

    def generate_sample(self, station: Station, variables_names: list, neighbourhood_size: int = None) -> Tuple[np.ndarray, float]:
        """
        Method to generate a sample for a station. The sample consists of the values of the variables at the gridcell of the station and the observation at the station.
        The features are ordered in the same order as the variables_names list.

        Arguments:
            station: Station instance
            variables_names: list of strings
            neighbourhood_size: int

        Returns:
            X: numpy array
            y: float
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
    
    def get_predictors(self) -> dict:
        """
        Returns the names of all the features (weather variables) that are in the forecast.

        Returns:
            the features (dict)
        """
        predictors = []
        for key, value in self.__dict__.items():
            if key not in self.ignore:
                predictors.append(value)
        return predictors
    
    def has_observations(self) -> bool:
        # check if the forecast has observations
        return len(self.observations) > 0
    
    def get_grid_variable(self, station: Station, variable_name: str, grid_size: int) -> np.ndarray:
        """
        Based on a Station and feature it takes the surrounding grid around the station for the feature.

        Arguments:
            station (Station): station that we want to find the surrounding grid for.
            variable_name (str): name of the feature
            grid_size (int): size of the grid

        Returns:
            The 2-dimensional grid (np.ndarray).
        """
        i, j = station.gridcell
        half = grid_size // 2
        variable = getattr(self, variable_name)
        return variable[i - half:i + half + 1, j - half:j + half + 1]
    
    
    def generate_ForecastSample(self, station_info: dict, variable_names: dict, ignore: list = []) -> list:
        """
        Makes for all stations in station_info (except for the ones in ignore) a ForecastSample (which contains only the relevant information, not the full grids).

        Arguments:
            station_info (dict): a dictionary containing station codes as keys and Station objects as values.
            variable_names (dict): a dictionary with as keys the name of the features that need to get included (should be 1 for all features except for the wind speed).
            ignore (list): a list of station codes that need to be ignored.
        """
        samples_list = []
        for station in station_info.values():
            if station.code in self.observations and station.code not in ignore:
                sample = ForecastSample(variable_names, station.code)

                
                for variable_name, grid_size in variable_names.items():
                    # we add each feature to the ForecastSample
                    if grid_size == 0 or grid_size == 1 or grid_size == None:
                        sample.add_feature(variable_name, getattr(self, variable_name)[station.gridcell])

                    # in case grid size is not 1, we add _grid to the name of the feature.
                    else:
                        sample.add_feature(variable_name, self.get_grid_variable(station, variable_name, grid_size))
                
                # we add the obervation to the Forecast Sample
                sample.add_y(self.observations[station.code][0])

                samples_list.append(sample)
        return samples_list
    
    def __contains__(self, station_code: str) -> bool:
        """
        Checks if the forecast contains an observation for the given station code.

        Args: 
            station_code: string

        Returns:
            boolean which checks whether there is an observation.
        """
        return station_code in self.observations


class ForecastSample():
    """
    The ForecastSample class represents a sample of forecast data for a specific station. 
    Each sample consists of various weather features and the observed value at the station.
    """
    def __init__(self, feature_names: list, station_code: str):
        self.feature_names = feature_names
        self.station_code = station_code
        # make attribute for each feature name
        for feature_name in feature_names:
            setattr(self, feature_name, None)
        self.y = None

    def add_feature(self, feature_name: str, value: Union[np.ndarray, float]) -> None:
        """
        Adds a features to the class.

        Arguments:
            feature_name (str): the name of the feature
            value (np.ndarray): the value, which can be a grid or a single value.
        """
        setattr(self, feature_name, value)
        if type(value) == np.ndarray:
            # add a boolean attribute for each feature name that is True if the feature is a grid
            setattr(self, feature_name + '_grid', True)
        else:
            setattr(self, feature_name + '_grid', False)


    def add_y(self, y: float):
        """
        Adds the observations.
        """
        self.y = y

    
    def check_if_everything_is_set(self) -> bool:
        """
        Checks whether all features in feature_names have a value.

        Returns:
            boolean indicating whether every feature has a value.
        """
        for feature_name in self.feature_names:
            if getattr(self, feature_name) is None:
                return False
            
        return self.y is not None
    

    def get_X(self) -> dict:
        """
        Makes a dictionary, with as keys the feature names and as values the associated value, which can be a grid (in case of wind speed) or a singular value.
        All the values are converted to tensors.
        """
        if not self.check_if_everything_is_set():
            raise ValueError('Not all features are set')
        
        # make a dictionary with the feature names as keys and the values as tensors. In case of a grid, the name of the key is the feature name + '_grid'
        X = {}
        for feature_name in self.feature_names:
            if getattr(self, feature_name + '_grid'):
                X[feature_name + '_grid'] = tf.convert_to_tensor(getattr(self, feature_name), dtype=tf.float32)
            else:
                X[feature_name] = tf.convert_to_tensor(getattr(self, feature_name), dtype=tf.float32)

        # We add the wind_speed_forecast to the features, for the peephole in the CNNs.
        if 'wind_speed' in X:
            # In case wind_speed is a single value, we set it to the value. 
            X['wind_speed_forecast'] = X['wind_speed']
        elif 'wind_speed_grid' in X:
            # In case we have a wind speed grid, we take thie middle value.
            grid = X['wind_speed_grid']
            central_value = grid[grid.shape[0] // 2, grid.shape[1] // 2]
            X['wind_speed_forecast'] = central_value
        else:
            raise ValueError("No wind speed found in the features!")

        # Only used for the climatology, which needs station codes.
        X['station_code'] = self.station_code
        
        return X
     

    def get_y(self) -> tf.Tensor:
        """
        Returns the observations as a tensor.
        """
        return tf.convert_to_tensor(self.y, dtype=tf.float32)
    


