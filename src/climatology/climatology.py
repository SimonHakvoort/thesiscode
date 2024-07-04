from typing import Callable
import numpy as np
import pickle

import tensorflow as tf

class Climatology:
    """
    Class for the climatology model. The model is based on the empirical distribution of past observations, and is done on a station level.
    """
    def __init__(self, data: tf.data.Dataset):
        self.calibrate(data)

    def calibrate(self, data: tf.data.Dataset) -> None:
        """
        Based on the data, we make a dictionary with as keys the station codes, and as values a sorted array of all observations in data.
        The data should contain all the data in a single batch.

        Arguments:
            data (tf.data.Dataset): data for which we make the climatology.

        Returns:
            None.
        """
        X, y = next(iter(data))
        X = X['station_code'].numpy()
        y = y.numpy()

        self.observations = {}
        for i in range(len(X)):
            if X[i] not in self.observations:
                # store the observations in a new np.array
                self.observations[X[i]] = np.array([y[i]])
            else:
                # append the observation to the existing np.array
                self.observations[X[i]] = np.append(self.observations[X[i]], y[i])

        # sort each array
        for key in self.observations:
            self.observations[key] = np.sort(self.observations[key])

        self.station_codes = np.unique(X)

    def Brier_Score(self, data: tf.data.Dataset, values: np.ndarray) -> np.ndarray:
        """
        Computes the Brier score for a single batch of the data, for all the values.

        Arguments:
            data (tf.data.Dataset): the data for which we compute the Brier score.
            values (np.ndarray): values for which Brier score is computed.

        Returns:
            an np.ndarray containing the Brier scores.
        """
        X, y = next(iter(data))
        X = X['station_code'].numpy()

        brier_scores = np.zeros(len(values))
        for i, threshold in enumerate(values):
            brier_scores[i] = self._comp_Brier_Score(X, y, threshold)

        return brier_scores
    
    def _get_cdf_station(self, station_code: str) -> Callable[[float], float]:
        """
        Compute the cdf for climatology for a specific station. 
        We simply what the fraction of observations are that exceed x.

        Arguments:
            station_code (str): the station code for which the cdf is computed.

        Returns:
            the cdf as a Callable[[float], float].
        """
        def cdf(x: float):
            """
            Checks what the fraction of observations are that exceed x.

            Arguments:
                x (float): the value to compare against the station's observations.

            Returns:
                float: the fraction of observation less than or equal to x.
            """
            return np.mean(self.observations[station_code] <= x)
        
        return cdf
    
    def _get_inverse_cdf_station(self, station_code: str) -> Callable[[float], float]:
        """
        Compute the inverse cumulative distribution function (inverse CDF or quantile function) for climatology for a specific station.

        The inverse CDF function for the given station is defined as the value below which a given fraction of observations fall.

        Arguments:
            station_code (str): The code of the station for which the inverse CDF is to be computed.

        Returns:
            Callable[[float], float]: A function that takes a float p (probability) and returns the value below which the fraction p 
                                    of observations fall for the station.
        """
        def inverse_cdf(p: float) -> float:
            """
            Compute the value below which the fraction p of observations fall for the specified station.

            Arguments:
                p (float): The probability (between 0 and 1) for which the quantile is computed.

            Returns:
                float: The value below which the fraction p of observations fall.
            """
            return np.quantile(self.observations[station_code], p)
        
        return inverse_cdf
    
    def _comp_Brier_Score(self, X: tf.Tensor, y: tf.Tensor, threshold: float) -> float:
        """
        Internal method to compute the Brier score for a single threshold for all the samples in X and y.

        Arguments:
            X (tf.Tensor): contains the station codes.
            y (tf.Tensor): contains the observations.

        Returns:
            The average Brier score for all the stations.
        """
        brier_scores = 0

        # cdf station is a method which computes how many of the observations for which Climatology is calibrated exceed a given threshold.
        # we store these as lambda functions inside a dictionary.
        cdfs = {station_code: self._get_cdf_station(station_code) for station_code in self.station_codes}

        # Exceeding contains 1 is y < threshold, otherwise it is 0.
        exceedings = (y < threshold).numpy()
        exceedings = exceedings.astype(int)

        for i in range(len(X)):
            brier_scores += (exceedings[i] - cdfs[X[i]](threshold))**2

        return brier_scores / len(X)
    
    def twCRPS(self, data: tf.data.Dataset, thresholds, sample_size = 1000):
        X, y = next(iter(data))
        X = X['station_code'].numpy()
        y = y.numpy()

        twcrps = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            twcrps[i] = self._comp_twCRPS(X, y, threshold, sample_size)

        return twcrps
    
    def CRPS(self, data, sample_size = 1000):
        return self.twCRPS(data, [0], sample_size)[0]
    
    
    def _comp_twCRPS(self, X, y, threshold, sample_size = 1000):
        # Get unique station codes and create a mapping for the inverse CDFs
        station_codes = self.station_codes
        inverse_cdfs = {station_code: self._get_inverse_cdf_station(station_code) for station_code in station_codes}

        # Precompute the samples for each station code
        random_samples = {station_code: np.random.uniform(0, 1, (sample_size, 2)) for station_code in station_codes}

        # Create an array to store the twcrps for each sample
        twcrps_array = np.zeros(len(X))

        def chain_function_indicator(x, threshold):
            return np.maximum(x, threshold)

        # Iterate through unique station codes to compute samples and store results
        for station_code in station_codes:
            indices = np.where(X == station_code)[0]
            samples = random_samples[station_code]

            X_1 = inverse_cdfs[station_code](samples[:, 0])
            X_2 = inverse_cdfs[station_code](samples[:, 1])

            vX_1 = chain_function_indicator(X_1, threshold)
            vX_2 = chain_function_indicator(X_2, threshold)

            for idx in indices:
                y_thresholded = chain_function_indicator(y[idx], threshold)

                E_1 = np.mean(np.abs(vX_1 - y_thresholded))
                E_2 = np.mean(np.abs(vX_1 - vX_2))

                twcrps_array[idx] = E_1 - 0.5 * E_2

        # Compute the final twcrps value
        twcrps = np.mean(twcrps_array)
        
        return twcrps
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


