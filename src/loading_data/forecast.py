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

    def generate_sample(self, station, neighbourhood_size, variables_names):
        # generate a sample for the given station, which includes the wind speed, the wind speed variance in the neighbourhood and the other variables
        # station is an instance of the Station class
        # neighbourhood_size is an odd number
        i, j = station.gridcell

        X = []
        y = self.observations[station.code][0]

        # for key, value in self.__dict__.items():
        #     if key not in self.ignore and key in variables_names:
        #         X.append(value[i, j])

        for key in variables_names:
            X.append(getattr(self, key)[i, j])

        if neighbourhood_size == 0:
            variance = 0
        else:
            variance = self.neighbourhood_variance((i, j), neighbourhood_size)

        return np.array(X), y, variance
    
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
    
        

    def generate_all_samples(self, neighbourhood_size, station_info, variable_names, station_ignore = []):
        # generate samples for all stations in station_info
        # station_info is a dictionary with station codes as keys and Station instances as values
        X = []
        y = []
        variances = []
        for station in station_info.values():
            if station.code in self.observations and station.code not in station_ignore:
                x, observation, variance = self.generate_sample(station, neighbourhood_size, variable_names)
                X.append(x)
                y.append(observation)
                variances.append(variance)

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32), tf.convert_to_tensor(variances, dtype=tf.float32)


