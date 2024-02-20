import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

class EMOS:
    def __init__(self, num_var, loss = "CRPS_sample", optimizer = "SGD", learning_rate = 0.001, samples = 100):
        #we should also consider the case of random initial values
        self.a = tf.Variable(tf.zeros(1, dtype=tf.float32))
        self.b = tf.Variable(tf.zeros(num_var, dtype=tf.float32))
        self.c = tf.Variable(tf.ones(1, dtype=tf.float32))
        self.d = tf.Variable(tf.zeros(1, dtype=tf.float32))

        if loss == "CRPS_sample":
            self.loss = self.CRPS_sample
            self.samples = samples
        elif loss == "log_likelihood":
            self.loss = self.loss_log_likelihood
        else:
            raise ValueError("Invalid loss function")

        # We can also implement Adam optimizer
        if optimizer == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "Adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def get_params(self):
        return self.a, self.b, self.c, self.d
    
    def forecast_distribution_trunc_normal(self, X, variance):
        mu = self.a + tf.tensordot(X, self.b, axes=1)
        # check whether i should take the square root
        sigma = tf.sqrt(tf.abs(self.c + self.d * variance))
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def loss_log_likelihood(self, X, y, variance):
        forecast_distribution = self.forecast_distribution_trunc_normal(X, variance)
        return -tf.reduce_mean(forecast_distribution.log_prob(y))
    
    def indicator_function(self, y, t):
        # true -> 1, false -> 0
        return tf.cast(y <= t, tf.float32)

    def CRPS_sample(self, X, y, variance):
        forecast_distribution = self.forecast_distribution_trunc_normal(X, variance)
        X_1 = forecast_distribution.sample(self.samples)
        X_2 = forecast_distribution.sample(self.samples)
        E_1 = tf.norm(X_1 - y, axis=0)
        E_2 = tf.norm(X_2 - y, axis=0)
        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)

     
    def fit(self, X, y, variance, steps):
        hist = []
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss_value = self.loss(X, y, variance)
            hist.append(loss_value)
            grads = tape.gradient(loss_value, [self.a, self.b, self.c, self.d])
            self.optimizer.apply_gradients(zip(grads, [self.a, self.b, self.c, self.d]))
            print("Step: {}, Loss: {}".format(step, loss_value))
        return hist
            