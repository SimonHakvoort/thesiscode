import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

class EMOS:
    # setup is a dictionary with all the settings
    def __init__(self, setup):#num_var, loss = "CRPS_sample", samples = 100, optimizer = "SGD", learning_rate = 0.001, forecast_distribution = "distr_trunc_normal", feature_mean = None, feature_std = None):
        if 'a' in setup and 'b' in setup and 'c' in setup and 'd' in setup:
            self.a = tf.Variable(setup['a'], dtype=tf.float32)
            self.b = tf.Variable(setup['b'], dtype=tf.float32)
            self.c = tf.Variable(setup['c'], dtype=tf.float32)
            self.d = tf.Variable(setup['d'], dtype=tf.float32)
        else:
            self.a = tf.Variable(tf.zeros(1, dtype=tf.float32))
            self.b = tf.Variable(tf.zeros(setup['num_features'], dtype=tf.float32))
            self.c = tf.Variable(tf.ones(1, dtype=tf.float32))
            self.d = tf.Variable(tf.zeros(1, dtype=tf.float32))

        if setup['loss'] == "loss_CRPS_sample":
            self.loss = self.loss_CRPS_sample
            self.samples = setup['samples']
        elif setup['loss'] == "loss_log_likelihood":
            self.loss = self.loss_log_likelihood
        else:
            raise ValueError("Invalid loss function")
        
        if setup['optimizer'] == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate=setup['learning_rate'])
        elif setup['optimizer'] == "Adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=setup['learning_rate'])
        else:
            raise ValueError("Invalid optimizer")
        
        if setup['forecast_distribution'] == "distr_trunc_normal":
            self.forecast_distribution = self.distr_trunc_normal
        else:
            raise ValueError("Invalid forecast distribution")
        
        if setup['feature_mean'] is not None and setup['feature_std'] is not None:
            self.feature_mean = tf.Variable(setup['feature_mean'])
            self.feature_std = tf.Variable(setup['feature_std'])

        if 'steps_made' in setup:
            self.steps_made = setup['steps_made']
        else:
            self.steps_made = 0

        self.feature_names = setup['features']
        self.neighbourhood_size = setup['neighbourhood_size']

    def set_params(self, a, b, c, d):
        self.a = tf.Variable(a, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)
        self.c = tf.Variable(c, dtype=tf.float32)
        self.d = tf.Variable(d, dtype=tf.float32)
    
    def get_params(self):
        return self.a.numpy(), self.b.numpy(), self.c.numpy(), self.d.numpy()
    
    def to_dict(self):
        model_dict = {
            'a': self.a.numpy(),
            'b': self.b.numpy(),
            'c': self.c.numpy(),
            'd': self.d.numpy(),
            'loss': self.loss.__name__,
            'samples': self.samples,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.learning_rate.numpy(),
            'forecast_distribution': self.forecast_distribution.__name__,
            'feature_mean': self.feature_mean.numpy() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.numpy() if self.feature_std is not None else None,
            'steps_made': self.steps_made,
            'features': self.feature_names,
            'neighbourhood_size': self.neighbourhood_size
        }
        return model_dict
    
    def distr_trunc_normal(self, X, variance):
        mu = self.a + tf.tensordot(X, self.b, axes=1)
        # check whether i should take the square root
        sigma = tf.sqrt(tf.abs(self.c + self.d * variance))
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def loss_log_likelihood(self, X, y, variance):
        forecast_distribution = self.forecast_distribution(X, variance)
        return -tf.reduce_mean(forecast_distribution.log_prob(y))
    
    def indicator_function(self, y, t):
        # true -> 1, false -> 0
        return tf.cast(y <= t, tf.float32)

    def loss_CRPS_sample(self, X, y, variance):
        forecast_distribution = self.forecast_distribution(X, variance)
        X_1 = forecast_distribution.sample(self.samples)
        X_2 = forecast_distribution.sample(self.samples)
        E_1 = tf.norm(X_1 - y, axis=0)
        E_2 = tf.norm(X_2 - y, axis=0)
        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
    def Brier_score(self, X, y, variance, threshold):
        forecast_distribution = self.forecast_distribution(X, variance)
        threshold = tf.constant(threshold, dtype=tf.float32)
        return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - forecast_distribution.cdf(threshold)))

     
    def fit(self, X, y, variance, steps):
        hist = []
        self.steps_made += steps
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss_value = self.loss(X, y, variance)
            hist.append(loss_value)
            grads = tape.gradient(loss_value, [self.a, self.b, self.c, self.d])
            self.optimizer.apply_gradients(zip(grads, [self.a, self.b, self.c, self.d]))
            print("Step: {}, Loss: {}".format(step, loss_value))
        return hist
            