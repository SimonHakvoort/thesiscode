import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

class EMOS:
    """
    Class for the EMOS model

    This class contains the EMOS model, which is used to calibrate ensemble forecasts. The model is initialized with a setup, and can be fitted to data using the fit method.
    In case we want to save the model, we can use the to_dict method to get a dictionary containing the parameters and additional settings of the model.

    Attributes:
    - loss: the loss function used to fit the model
    - samples: the amount of samples used in the loss function

    - forecast_distribution: the distribution used to model the forecast
    - parameter_dict: a dictionary containing the parameters of the forecast distribution

    - feature_names: the names of the features used in the model
    - num_features: the amount of features used in the model
    - neighbourhood_size: the size of the neighbourhood used in the model
    """
    def __init__(self, setup):
        """
        Initialize the EMOS model

        We initialize the model with the given setup. This is done using the getattr function.

        Arguments:
        - setup: a dictionary containing the setup for the model.

        The setup should contain the following keys:
        - loss: the loss function used to fit the model
        - samples: the amount of samples used in the loss function in case we use a sample based loss function
        - optimizer: the optimizer used to fit the model
        - learning_rate: the learning rate used in the optimizer
        - forecast_distribution: the distribution used to model the forecast
        """

        self.need_chain = False

        # LOSS FUNCTION ( & CHAIN FUNCTION & SAMPLES)
        try:
            self.loss = getattr(self, setup['loss'])
            if self.loss == self.loss_CRPS_sample or self.loss == self.loss_twCRPS_sample:
                if 'samples' not in setup:
                    raise ValueError("Amount of samples not specified")
                else:
                    self.samples = setup['samples']
                if self.loss == self.loss_twCRPS_sample:
                    self.need_chain = True
        except AttributeError:
            raise ValueError("Invalid loss function: " + setup['loss'])
        
        # CHAIN FUNCTION
        if self.need_chain:
            try:
                self.chain_function = getattr(self, setup['chain_function'])
                self.threshold = tf.constant(setup['threshold'], dtype=tf.float32) # this needs to be changed later on, in case we have normal distribution as chain function
            except AttributeError:
                raise ValueError("Invalid chain function: " + setup['chain_function'])

        # OPTIMIZER & LEARNING RATE
        try:
            if 'learning_rate' not in setup:
                raise ValueError("Learning rate not specified")
            self.optimizer = getattr(tf.optimizers, setup['optimizer'])(learning_rate=setup['learning_rate'])
        except AttributeError:
            raise ValueError("Invalid optimizer: " + setup['optimizer'])
        
        # The setup of the forecast distribution
        if 'mixture' not in setup or setup['mixture'] == 1:
            try:
                self.forecast_distribution = getattr(self, setup['forecast_distribution'])
            except AttributeError:
                raise ValueError("Invalid forecast distribution: " + setup['forecast_distribution'])
        

        self.feature_names = setup['features']
        self.num_features = len(self.feature_names)
        self.neighbourhood_size = setup['neighbourhood_size']
        
        # PARAMETERS
        self.parameter_dict = {}
        if self.forecast_distribution == self.distr_trunc_normal:
            if 'a_tn' in setup and 'b_tn' in setup and 'c_tn' in setup and 'd_tn' in setup:
                self.initialize_trunc_normal(default = False,  a = setup['a_tn'], b = setup['b_tn'], c = setup['c_tn'], d = setup['d_tn'])
            else:
                self.initialize_trunc_normal(default = True)
        elif self.forecast_distribution == self.distr_log_normal:
            if 'a_ln' in setup and 'b_ln' in setup and 'c_ln' in setup and 'd_ln' in setup:
                self.initialize_log_normal(default = False, a = setup['a_ln'], b = setup['b_ln'], c = setup['c_ln'], d = setup['d_ln'])
            else:
                self.initialize_log_normal(default = True)

        # Optionally we can initialize the feature mean and standard deviation with the given values. Not sure whether this needs to be included
        if setup['feature_mean'] is not None and setup['feature_std'] is not None:
            self.feature_mean = tf.Variable(setup['feature_mean'])
            self.feature_std = tf.Variable(setup['feature_std'])

        # Optionally we can initialize the amount of steps made with the optimizer
        if 'steps_made' in setup:
            self.steps_made = setup['steps_made']
        else:
            self.steps_made = 0
            
  
    def initialize_trunc_normal(self, default, a = 0, b = 0, c = 1, d = 0):
        """
        Initialize the parameters of the truncated normal distribution and stores them in parameter_dict. We use a linear relationship between the parameters and the features.

        Arguments:
        - default: whether to initialize the parameters with default values
        - a: the initial value of the parameter a
        - b: the initial value of the parameter b
        - c: the initial value of the parameter c
        - d: the initial value of the parameter d
        """
        if default:
            self.parameter_dict['a_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32))
            self.parameter_dict['b_tn'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
            self.parameter_dict['c_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32))
            self.parameter_dict['d_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32))
        else:
            self.parameter_dict['a_tn'] = tf.Variable(a, dtype=tf.float32)
            self.parameter_dict['b_tn'] = tf.Variable(b, dtype=tf.float32)
            self.parameter_dict['c_tn'] = tf.Variable(c, dtype=tf.float32)
            self.parameter_dict['d_tn'] = tf.Variable(d, dtype=tf.float32)


    def initialize_log_normal(self, default, a = 0, b = 0, c = 1, d = 0):
        """
        Initialize the parameters of the log normal distribution and stores them in parameter_dict. We use a linear relationship between the parameters and the features.

        Arguments:
        - default: whether to initialize the parameters with default values.
        - a: the initial value of the parameter a.
        - b: the initial value of the parameter b.
        - c: the initial value of the parameter c.
        - d: the initial value of the parameter d.
        """
        if default:
            self.parameter_dict['a_ln'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
            self.parameter_dict['b_ln'] = tf.Variable(tf.zeros(self.num_features, dtype=tf.float32))
            self.parameter_dict['c_ln'] = tf.Variable(tf.ones(1, dtype=tf.float32))
            self.parameter_dict['d_ln'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
        else:
            self.parameter_dict['a_ln'] = tf.Variable(a, dtype=tf.float32)
            self.parameter_dict['b_ln'] = tf.Variable(b, dtype=tf.float32)
            self.parameter_dict['c_ln'] = tf.Variable(c, dtype=tf.float32)
            self.parameter_dict['d_ln'] = tf.Variable(d, dtype=tf.float32)

        
    
    def get_params(self):
        """
        Return the parameters of the model as a dictionary.
        """
        output_dict = {}
        for parameter in self.parameter_dict:
            output_dict[parameter] = self.parameter_dict[parameter].numpy()
        return output_dict
    
    def to_dict(self):
        """
        Return the model as a dictionary, including the settings and parameters.
        """
        model_dict = {
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
        for parameter in self.parameter_dict:
            model_dict[parameter] = self.parameter_dict[parameter].numpy()
        return model_dict
    
    def indicator_function(self, y, t):
        """
        The indicator function, which returns 1 if y <= t, and 0 otherwise.

        Arguments:
        - y: the input value.
        - t: the threshold.

        Returns:
        - 1 if y <= t, 0 otherwise.
        """
        return tf.cast(y <= t, tf.float32)

    
    def distr_trunc_normal(self, X, variance):
        """
        The truncated normal distribution. We use a linear relationship between the parameters and the features.

        Arguments:
        - X: the input data.
        - variance: the variance around the grid point, with a square grid around the neighbourhood station of size neighbourhood_size x neighbourhood_size.

        Returns: 
        - the truncated normal distribution.
        """
        mu = self.parameter_dict['a_tn'] + tf.tensordot(X, self.parameter_dict['b_tn'], axes=1)
        sigma = tf.sqrt(tf.abs(self.parameter_dict['c_tn'] + self.parameter_dict['d_tn'] * variance))
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def distr_log_normal(self, X, variance):
        """
        The log normal distribution. We use a linear relationship between the parameters and the features.

        Arguments:
        - X: the input data.
        - variance: the variance around the grid point, with a square grid around the neighbourhood station of size neighbourhood_size x neighbourhood_size.

        Returns:
        - the log normal distribution.
        """
        mu = self.parameter_dict['a_ln'] + tf.tensordot(X, self.parameter_dict['b_ln'], axes=1)
        sigma = tf.sqrt(tf.abs(self.parameter_dict['c_ln'] + self.parameter_dict['d_ln'] * variance))
        return tfpd.LogNormal(mu, sigma)

    
    def loss_log_likelihood(self, X, y, variance):
        """
        The loss fuction for the log likelihood, based on the forecast distribution and observations.

        Arguments
        - X: the input data.
        - y: the output data.
        - variance: the variance of the forecast distribution around the grid point.

        Returns:
        - the loss value.
        """
        forecast_distribution = self.forecast_distribution(X, variance)
        return -tf.reduce_mean(forecast_distribution.log_prob(y))

    def loss_CRPS_sample(self, X, y, variance):
        """
        The loss function for the CRPS, based on the forecast distribution and observations. We use a sample based approach to estimate the expected value of the CRPS.

        Arguments:
        - X: the input data.
        - y: the output data.
        - variance: the variance of the forecast distribution around the grid point.

        Returns:
        - the loss value.
        """
        forecast_distribution = self.forecast_distribution(X, variance)
        X_1 = forecast_distribution.sample(self.samples)
        X_2 = forecast_distribution.sample(self.samples)

        # y will be broadcasted to the shape of X_1 and X_2
        # we compute the norm for each sample (resulting in an estimation of the expected value of the CRPS)
        E_1 = tf.norm(X_1 - y, axis=0)
        E_2 = tf.norm(X_2 - y, axis=0)
        
        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
    def loss_Brier_score(self, X, y, variance, threshold):
        forecast_distribution = self.forecast_distribution(X, variance)
        threshold = tf.constant(threshold, dtype=tf.float32)
        return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - forecast_distribution.cdf(threshold)))

    
    def loss_twCRPS_sample(self, X, y, variance):
        return self.loss_twCRPS_sample_general(X, y, variance, self.chain_function, self.samples)
        
    def loss_twCRPS_sample_general(self, X, y, variance, chain_function, samples):
        forecast_distribution = self.forecast_distribution(X, variance)
        X_1 = forecast_distribution.sample(samples)
        X_2 = forecast_distribution.sample(samples)
        vX_1 = chain_function(X_1)
        vX_2 = chain_function(X_2)
        E_1 = tf.norm(vX_1 - chain_function(y), axis=0)
        E_2 = tf.norm(vX_2 - vX_1, axis=0)

        #E_1_ = tf.sqrt(tf.reduce_sum(tf.square(vX_1 - chain_function(y)) + 1.0e-20))
        #E_2_ = tf.sqrt(tf.reduce_sum(tf.square(vX_2 - vX_1) + 1.0e-20))


        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
        #return tf.reduce_mean(E_1_) - 0.5 * tf.reduce_mean(E_2_)
        

    def chain_function_indicator(self, y):
        # This is the chaining function for the indicator weight function, where t is the threshold
        # The output will be max(y, t)
        return tf.maximum(y, self.threshold)
     
    def fit(self, X, y, variance, steps):
        """
        Fit the EMOS model to the given data, using the loss function and optimizer specified in the setup.

        Arguments:
        - X: the input data
        - y: the output data
        - variance: the variance of the forecast around the grid point, with grid size neighbourhood_size
        - steps: the amount of steps to take with the optimizer

        Returns:
        - hist: a list containing the loss value at each step
        """
        hist = []
        self.steps_made += steps
        for step in range(steps):
            with tf.GradientTape() as tape:
                loss_value = self.loss(X, y, variance)
            grads = tape.gradient(loss_value, [*self.parameter_dict.values()]) 
            # check if gradient contains nan
            if tf.math.reduce_any(tf.math.is_nan(grads[0])):
                print("Gradient contains NaN")
                continue
            hist.append(loss_value)
            self.optimizer.apply_gradients(zip(grads, self.parameter_dict.values()))

            print("Step: {}, Loss: {}".format(step, loss_value))
        return hist
            