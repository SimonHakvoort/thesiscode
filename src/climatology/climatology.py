import numpy as np
import pickle

class Climatology:
    def __init__(self, data):
        self.calibrate(data)

    def calibrate(self, data):
        X, y = next(iter(data))
        X = X['station_code'].numpy()
        y = y.numpy()

        observations = {}
        for i in range(len(X)):
            if X[i] not in observations:
                # store the observations in a new np.array
                observations[X[i]] = np.array([y[i]])
            else:
                # append the observation to the existing np.array
                observations[X[i]] = np.append(observations[X[i]], y[i])

        # sort each array
        for key in observations:
            observations[key] = np.sort(observations[key])

        self.observations = observations
        self.station_codes = np.unique(X)

    def get_Brier_scores(self, data, values):
        X, y = next(iter(data))
        X = X['station_code'].numpy()

        brier_scores = np.zeros(len(values))
        for i, threshold in enumerate(values):
            brier_scores[i] = self.Brier_Score(X, y, threshold)

        return brier_scores
    
    def get_cdf_station(self, station_code):
        def cdf(x):
            return np.mean(self.observations[station_code] <= x)
        
        return cdf
    
    def get_inverse_cdf_station(self, station_code):
        def inverse_cdf(p):
            return np.quantile(self.observations[station_code], p)
        
        return inverse_cdf
    
    def Brier_Score(self, X, y, threshold):
        brier_scores = 0
        cdfs = {station_code: self.get_cdf_station(station_code) for station_code in self.station_codes}

        # make a variable exceedings, that is 1 if y < threshold and 0 otherwise
        exceedings = (y < threshold).numpy()
        exceedings = exceedings.astype(int)

        for i in range(len(X)):
            brier_scores += (exceedings[i] - cdfs[X[i]](threshold))**2

        return brier_scores / len(X)
    
    def get_twCRPS(self, data, thresholds, sample_size = 1000):
        X, y = next(iter(data))
        X = X['station_code'].numpy()
        y = y.numpy()

        twcrps = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            twcrps[i] = self.twCRPS_threshold(X, y, threshold, sample_size)

        return twcrps
    
    
    def twCRPS_threshold(self, X, y, threshold, sample_size = 1000):
        # Get unique station codes and create a mapping for the inverse CDFs
        station_codes = self.station_codes
        inverse_cdfs = {station_code: self.get_inverse_cdf_station(station_code) for station_code in station_codes}

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
        # # self.station_codes = np.unique(X)
        # inverse_cdfs = {station_code: self.get_inverse_cdf_station(station_code) for station_code in self.station_codes}

        # twcrps = 0

        # def chain_function_indicator(x, threshold):
        #     return np.maximum(x, threshold)

        # for i in range(len(X)):
        #     uniform_samples_1 = np.random.uniform(0, 1, sample_size)
        #     X_1 = inverse_cdfs[X[i]](uniform_samples_1)

        #     uniform_samples_2 = np.random.uniform(0, 1, sample_size)
        #     X_2 = inverse_cdfs[X[i]](uniform_samples_2)

        #     vX_1 = chain_function_indicator(X_1, threshold)
        #     vX_2 = chain_function_indicator(X_2, threshold)

        #     E_1 = np.mean(np.abs(vX_1 - chain_function_indicator(y[i], threshold)))
        #     E_2 = np.mean(np.abs(vX_1 - vX_2))

        #     twcrps += E_1 - 0.5 * E_2

        # return twcrps / len(X)
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


