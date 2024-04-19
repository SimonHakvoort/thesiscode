from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class Normalizer():
    def __init__(self, feature_names, normalize_wind_speed = True, normalize_none = False):
        self.normalize_wind_speed = normalize_wind_speed
        self.normalize_none = normalize_none
        self.feature_names = feature_names
        self.scaler = StandardScaler()
    
    def fit(self, X):
        if self.normalize_none:
            return
        if self.normalize_wind_speed:
            self.scaler.fit(X)
        else:
            # find the indices of the features that are not wind_speed
            indices = [i for i, feature in enumerate(self.feature_names) if feature != 'wind_speed']
            self.scaler.fit(X[:, indices])
    
    def transform(self, X):
        if self.normalize_none:
            return X
        if self.normalize_wind_speed:
            return self.scaler.transform(X)
        # else:
        #     indices = [i for i, feature in enumerate(self.feature_names) if feature != 'wind_speed']
        #     index_wind_speed = self.feature_names.index('wind_speed')
        #     X_scaled = self.scaler.transform(tf.gather(X, indices, axis=-1))
        #     return tf
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        if self.normalize_none:
            return X
        if self.normalize_wind_speed:
            return self.scaler.inverse_transform(X)
        else:
            indices = [i for i, feature in enumerate(self.feature_names) if feature != 'wind_speed']
            X[:, indices] = self.scaler.inverse_transform(X[:, indices])
            return X
        
    



