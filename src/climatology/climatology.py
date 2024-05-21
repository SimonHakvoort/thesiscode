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
    
    def Brier_Score(self, X, y, threshold):
        brier_scores = 0
        cdfs = {station_code: self.get_cdf_station(station_code) for station_code in np.unique(X)}

        # make a variable exceedings, that is 1 if y < threshold and 0 otherwise
        exceedings = (y < threshold).numpy()
        exceedings = exceedings.astype(int)

        for i in range(len(X)):
            brier_scores += (exceedings[i] - cdfs[X[i]](threshold))**2

        return brier_scores / len(X)
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


