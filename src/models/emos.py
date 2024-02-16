import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

class EMOS:
    def __init__(self, num_var):
        #we should also consider the case of random initial values
        self.a = tf.Variable(tf.zeros(1, dtype=tf.float32))
        self.b = tf.Variable(tf.zeros(num_var, dtype=tf.float32))
        self.c = tf.Variable(tf.ones(1, dtype=tf.float32))
        self.d = tf.Variable(tf.zeros(1, dtype=tf.float32))

        # We can also implement Adam optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

    def get_params(self):
        return self.a, self.b, self.c, self.d
    
    def forecast_distribution_trunc_normal(self, X, variance):
        mu = self.a + tf.tensordot(X, self.b, axes=1)
        # check whether i should take the square root
        sigma = tf.sqrt(self.c + self.d * variance)
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def loss(self, X, y, variance):
        forecast_distribution = self.forecast_distribution_trunc_normal(X, variance)
        return -tf.reduce_mean(forecast_distribution.log_prob(y))
     
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
            