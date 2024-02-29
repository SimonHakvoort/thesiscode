# import numpy as np
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfpd = tfp.distributions

# class EMOS:
#     """
#     Class for the EMOS model

#     This class contains the EMOS model, which is used to calibrate ensemble forecasts. The model is initialized with a setup, and can be fitted to data using the fit method.
#     In case we want to save the model, we can use the to_dict method to get a dictionary containing the parameters and additional settings of the model.

#     Attributes:
#     - loss: the loss function used to fit the model
#     - samples: the amount of samples used in the loss function

#     - forecast_distribution: the distribution used to model the forecast
#     - parameter_dict: a dictionary containing the parameters of the forecast distribution

#     - feature_names: the names of the features used in the model
#     - num_features: the amount of features used in the model
#     - neighbourhood_size: the size of the neighbourhood used in the model
#     """
#     def __init__(self, setup):
#         """
#         Initialize the EMOS model

#         We initialize the model with the given setup. This is done using the getattr function.

#         Arguments:
#         - setup: a dictionary containing the setup for the model.

#         The setup should contain the following keys:
#         - loss: the loss function used to fit the model
#         - samples: the amount of samples used in the loss function in case we use a sample based loss function
#         - optimizer: the optimizer used to fit the model
#         - learning_rate: the learning rate used in the optimizer
#         - forecast_distribution: the distribution used to model the forecast
#         """
#         self.feature_names = setup['features']
#         self.num_features = len(self.feature_names)
#         self.neighbourhood_size = setup['neighbourhood_size']

#         self.init_loss(setup)
        
#         if self.need_chain:
#             self.init_chain_function(setup)

#         self.init_optimizer(setup)

        
#         self.init_forecast_distribution(setup)


#         # Optionally we can initialize the feature mean and standard deviation with the given values. Not sure whether this needs to be included
#         if setup['feature_mean'] is not None and setup['feature_std'] is not None:
#             self.feature_mean = tf.Variable(setup['feature_mean'])
#             self.feature_std = tf.Variable(setup['feature_std'])

#         # Optionally we can initialize the amount of steps made with the optimizer
#         if 'steps_made' in setup:
#             self.steps_made = setup['steps_made']
#         else:
#             self.steps_made = 0

#     def init_loss(self, setup):
#         self.need_chain = False
#         try:
#             self.loss = getattr(self, setup['loss'])
#             if self.loss == self.loss_CRPS_sample or self.loss == self.loss_twCRPS_sample:
#                 if 'samples' not in setup:
#                     raise ValueError("Amount of samples not specified")
#                 else:
#                     self.samples = setup['samples']
#                 if self.loss == self.loss_twCRPS_sample:
#                     self.need_chain = True
#         except AttributeError:
#             raise ValueError("Invalid loss function: " + setup['loss'])   

#     def init_chain_function(self, setup):
#         try:
#             if setup['chain_function'] == 'chain_function_indicator':
#                 self.chain_function = self.chain_function_indicator
#                 if 'threshold' not in setup:
#                     raise ValueError("Threshold of the chain function not specified")
#                 else:
#                     self.threshold = tf.constant(setup['threshold'], dtype=tf.float32)
#             elif setup['chain_function'] == 'chain_function_normal_cdf':
#                 self.chain_function = self.chain_function_normal_cdf
#                 if 'chain_function_mean' not in setup:
#                     raise ValueError("Mean  of the chain function not specified")
#                 else:
#                     self.chain_function_mean = tf.constant(setup['chain_function_mean'], dtype=tf.float32)
#                 if 'chain_function_std' not in setup:
#                     raise ValueError("Standard deviation of the chain function not specified")
#                 else:
#                     self.chain_function_std = tf.constant(setup['chain_function_std'], dtype=tf.float32)
#                 self.chain_normal_distr = tfpd.Normal(self.chain_function_mean, self.chain_function_std)
#         except AttributeError:
#             raise ValueError("Invalid chain function: " + setup['chain_function'])  

#     def init_optimizer(self, setup):
#         try:
#             if 'learning_rate' not in setup:
#                 raise ValueError("Learning rate not specified")
#             self.optimizer = getattr(tf.optimizers, setup['optimizer'])(learning_rate=setup['learning_rate'])
#         except AttributeError:
#             raise ValueError("Invalid optimizer: " + setup['optimizer']) 
        
#     def init_forecast_distribution(self, setup):
#         # The setup of the forecast distribution
#         try:
#             self.forecast_distribution = getattr(self, setup['forecast_distribution'])
#         except AttributeError:
#             raise ValueError("Invalid forecast distribution: " + setup['forecast_distribution'])
        
#         # PARAMETERS
#         self.parameter_dict = {}
#         if self.forecast_distribution == self.distr_trunc_normal:
#             if 'parameters' in setup:
#                 self.initialize_trunc_normal(default = False, parameters = setup['parameters'])
#             else:
#                 self.initialize_trunc_normal(default = True)
#         elif self.forecast_distribution == self.distr_log_normal:
#             if 'parameters' in setup:
#                 self.initialize_log_normal(default = False, parameters = setup['parameters'])
#             else:
#                 self.initialize_log_normal(default = True)
#         elif self.forecast_distribution == self.distr_gev:
#             if 'parameters' in setup:
#                 self.initialize_gev(default = False, parameters = setup['parameters'])
#             else:
#                 self.initialize_gev(default = True)
#         elif self.forecast_distribution == self.distr_gev2:
#             if 'parameters' in setup:
#                 self.initialize_gev2(default = False, parameters = setup['parameters'])
#             else:
#                 self.initialize_gev2(default = True)
#         elif self.forecast_distribution == self.distr_gev3:
#             if 'parameters' in setup:
#                 self.initialize_gev3(default = False, parameters = setup['parameters'])
#             else:
#                 self.initialize_gev3(default = True)    
#         elif self.forecast_distribution == self.distr_mixture:
#             if 'parameters' in setup:
#                 self.initialize_mixture(default = False, setup = setup)
#             else:   
#                 self.initialize_mixture(default = True, setup = setup)
#         elif self.forecast_distribution == self.distr_mixture_linear:
#             if 'parameters' in setup:
#                 self.initialize_mixture_linear(default = False, setup = setup)
#             else:
#                 self.initialize_mixture_linear(default = True, setup = setup)
    
#     def __len__(self):
#         return len(self.parameter_dict)
    
#     def __str__(self):
#         # Loss function info
#         loss_info = f"Loss function: {self.loss.__name__}"
#         if hasattr(self, 'samples'):
#             loss_info += f" (Samples: {self.samples})"

#         # Optimizer info
#         optimizer_info = f"Optimizer: {self.optimizer.__class__.__name__}"
#         learning_rate_info = f"Learning rate: {self.optimizer.learning_rate.numpy()}"

#         # Forecast distribution info
#         forecast_distribution_info = f"Forecast distribution: {self.forecast_distribution.__name__}"

#         # Feature info
#         feature_info = f"Features: {', '.join(self.feature_names)}"
#         num_features_info = f"Number of features: {self.num_features}"
#         neighbourhood_size_info = f"Neighbourhood size: {self.neighbourhood_size}"

#         # Parameters info
#         parameters_info = "Parameters:"
#         for parameter, value in self.parameter_dict.items():
#             parameters_info += f"\n  {parameter}: {value.numpy()}"

#         # Chaining function info
#         chaining_function_info = ""
#         if hasattr(self, 'chain_function'):
#             chaining_function_info = f"Chaining function: {self.chain_function.__name__}"
#             if hasattr(self, 'threshold'):
#                 chaining_function_info += f" (Threshold: {self.threshold.numpy()})"
#             elif hasattr(self, 'chain_function_mean') and hasattr(self, 'chain_function_std'):
#                 chaining_function_info += f" (Mean: {self.chain_function_mean.numpy()}, Std: {self.chain_function_std.numpy()})"

#         distribution_info = ""
#         if self.forecast_distribution == self.distr_mixture:
#             distribution_info = f"Distribution 1: {self.distribution_1.__name__}\n"
#             distribution_info += f"Distribution 2: {self.distribution_2.__name__}\n"
#             distribution_info += f"Mixture weight: {self.parameter_dict['weight'].numpy()}"
#         elif self.forecast_distribution == self.distr_mixture_linear:
#             distribution_info = f"Distribution 1: {self.distribution_1.__name__}\n"
#             distribution_info += f"Distribution 2: {self.distribution_2.__name__}\n"
#             distribution_info += f"Mixture weight a: {self.parameter_dict['weight_a'].numpy()}\n"
#             distribution_info += f"Mixture weight b: {self.parameter_dict['weight_b'].numpy()}\n"
#             distribution_info += f"Mixture weight c: {self.parameter_dict['weight_c'].numpy()}"


#         return (
#             f"EMOS Model Information:\n"
#             f"{loss_info}\n"
#             f"{forecast_distribution_info}\n"
#             f"{distribution_info}"
#             f"{parameters_info}\n"
#             f"{feature_info}\n"
#             f"{num_features_info}\n"
#             f"{neighbourhood_size_info}\n"
#             f"{chaining_function_info}\n"
#             f"{optimizer_info}\n"
#             f"{learning_rate_info}\n"
#         )


#     def initialize_trunc_normal(self, default, parameters = {}):
#         """
#         Initialize the parameters of the truncated normal distribution and stores them in parameter_dict. We use a linear relationship between the parameters and the features.

#         Arguments:
#         - default: whether to initialize the parameters with default values
#         - parameters: a dictionary containing the parameters of the truncated normal distribution
#         """
#         if default:
#             self.parameter_dict['a_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['b_tn'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['c_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['d_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             print("Using default parameters for truncated normal distribution")
#         else:
#             try:
#                 self.parameter_dict['a_tn'] = tf.Variable(parameters['a_tn'], dtype=tf.float32)
#                 self.parameter_dict['b_tn'] = tf.Variable(parameters['b_tn'], dtype=tf.float32)
#                 self.parameter_dict['c_tn'] = tf.Variable(parameters['c_tn'], dtype=tf.float32)
#                 self.parameter_dict['d_tn'] = tf.Variable(parameters['d_tn'], dtype=tf.float32)
#                 print("Using given parameters for truncated normal distribution")
#             except KeyError:
#                 raise ValueError("Invalid parameters for truncated normal distribution")


#     def initialize_log_normal(self, default, parameters = {}):
#         """
#         Initialize the parameters of the log normal distribution and stores them in parameter_dict. We use a linear relationship between the parameters and the features.

#         Arguments:
#         - default: whether to initialize the parameters with default values.
#         - parameters: a dictionary containing the parameters of the log normal distribution.
#         """
#         if default:
#             self.parameter_dict['a_ln'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
#             self.parameter_dict['b_ln'] = tf.Variable(tf.zeros(self.num_features, dtype=tf.float32))
#             self.parameter_dict['c_ln'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['d_ln'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
#             print("Using default parameters for log normal distribution")
#         else:
#             try:
#                 self.parameter_dict['a_ln'] = tf.Variable(parameters['a_ln'], dtype=tf.float32)
#                 self.parameter_dict['b_ln'] = tf.Variable(parameters['b_ln'], dtype=tf.float32)
#                 self.parameter_dict['c_ln'] = tf.Variable(parameters['c_ln'], dtype=tf.float32)
#                 self.parameter_dict['d_ln'] = tf.Variable(parameters['d_ln'], dtype=tf.float32)
#                 print("Using given parameters for log normal distribution")
#             except KeyError:
#                 raise ValueError("Invalid parameters for log normal distribution")
            

#     def initialize_gev(self, default, parameters = {}):
#         if default:
#             self.parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['b_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['d_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.3)
#             print("Using default parameters for Generalized Extreme Value distribution")
#         else:
#             try:
#                 self.parameter_dict['a_gev'] = tf.Variable(parameters['a_gev'], dtype=tf.float32)
#                 self.parameter_dict['b_gev'] = tf.Variable(parameters['b_gev'], dtype=tf.float32)
#                 self.parameter_dict['c_gev'] = tf.Variable(parameters['c_gev'], dtype=tf.float32)
#                 self.parameter_dict['d_gev'] = tf.Variable(parameters['d_gev'], dtype=tf.float32)
#                 self.parameter_dict['e_gev'] = tf.Variable(parameters['e_gev'], dtype=tf.float32)
#                 print("Using given parameters for Generalized Extreme Value distribution")
#             except KeyError:
#                 raise ValueError("Invalid parameters for Generalized Extreme Value distribution")
            
#     def initialize_gev2(self, default, parameters = {}):
#         if default:
#             self.parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['b_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['d_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))

#             self.parameter_dict['extra_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             print("Using default parameters for Generalized Extreme Value distribution 2")
#         else:
#             try:
#                 self.parameter_dict['a_gev'] = tf.Variable(parameters['a_gev'], dtype=tf.float32)
#                 self.parameter_dict['b_gev'] = tf.Variable(parameters['b_gev'], dtype=tf.float32)
#                 self.parameter_dict['c_gev'] = tf.Variable(parameters['c_gev'], dtype=tf.float32)
#                 self.parameter_dict['d_gev'] = tf.Variable(parameters['d_gev'], dtype=tf.float32)
#                 self.parameter_dict['e_gev'] = tf.Variable(parameters['e_gev'], dtype=tf.float32)
                
#                 self.parameter_dict['extra_gev'] = tf.Variable(parameters['extra_gev'], dtype=tf.float32)
#                 print("Using given parameters for Generalized Extreme Value distribution 2")
#             except KeyError:
#                 raise ValueError("Invalid parameters for Generalized Extreme Value distribution")
            
#     def initialize_gev3(self, default, parameters = {}):
#         if default:
#             self.parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['b_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             self.parameter_dict['d_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#             self.parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))

#             self.parameter_dict['extra_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#             print("Using default parameters for Generalized Extreme Value distribution 3")
#         else:
#             try:
#                 self.parameter_dict['a_gev'] = tf.Variable(parameters['a_gev'], dtype=tf.float32)
#                 self.parameter_dict['b_gev'] = tf.Variable(parameters['b_gev'], dtype=tf.float32)
#                 self.parameter_dict['c_gev'] = tf.Variable(parameters['c_gev'], dtype=tf.float32)
#                 self.parameter_dict['d_gev'] = tf.Variable(parameters['d_gev'], dtype=tf.float32)
#                 self.parameter_dict['e_gev'] = tf.Variable(parameters['e_gev'], dtype=tf.float32)

#                 self.parameter_dict['extra_gev'] = tf.Variable(parameters['extra_gev'], dtype=tf.float32)
#                 print("Using given parameters for Generalized Extreme Value distribution 3")
#             except KeyError:
#                 raise ValueError("Invalid parameters for Generalized Extreme Value distribution")

#     def initialize_mixture(self, default, setup):
#         try:
#             self.distribution_1 = getattr(self, setup['distribution_1'])
#             self.distribution_2 = getattr(self, setup['distribution_2'])

#             if self.distribution_1 == self.distr_trunc_normal or self.distribution_2 == self.distr_trunc_normal:
#                 if default:
#                     self.initialize_trunc_normal(default)
#                 else:
#                     self.initialize_trunc_normal(default, setup['parameters'])

#             if self.distribution_1 == self.distr_log_normal or self.distribution_2 == self.distr_log_normal:
#                 if default:
#                     self.initialize_log_normal(default)
#                 else:
#                     self.initialize_log_normal(default, setup['parameters'])

#             if self.distribution_1 == self.distr_gev or self.distribution_2 == self.distr_gev:
#                 if default:
#                     self.initialize_gev(default)
#                 else:
#                     self.initialize_gev(default, setup['parameters'])
            
#             if self.distribution_1 == self.distr_gev2 or self.distribution_2 == self.distr_gev2:
#                 if default:
#                     self.initialize_gev2(default)
#                 else:
#                     self.initialize_gev2(default, setup['parameters'])
            
#             if self.distribution_1 == self.distr_gev3 or self.distribution_2 == self.distr_gev3:
#                 if default:
#                     self.initialize_gev3(default)
#                 else:
#                     self.initialize_gev3(default, setup['parameters'])

#             constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)

#             if default: 
#                 self.parameter_dict['weight'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.5, dtype=tf.float32, trainable=True, name='weight', constraint=constraint)
#                 print("Using default weight parameter")
#             else:
#                 if 'weight' in setup['parameters']:
#                     self.parameter_dict['weight'] = tf.Variable(initial_value=setup['parameters']['weight'], dtype=tf.float32, trainable=True, name='weight', constraint=constraint)
#                     print("Using given weight parameter")
#                 else:
#                     self.parameter_dict['weight'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.5, dtype=tf.float32, trainable=True, name='weight', constraint=constraint)
#                     print("No weight parameter found, using default value of 0.5")

#         except AttributeError:
#             raise ValueError("Invalid forecast distribution: " + setup['forecast_distribution'])
        
#     def initialize_mixture_linear(self, default, setup):
#         try:
#             self.distribution_1 = getattr(self, setup['distribution_1'])
#             self.distribution_2 = getattr(self, setup['distribution_2'])

#             if self.distribution_1 == self.distr_trunc_normal or self.distribution_2 == self.distr_trunc_normal:
#                 if default:
#                     self.initialize_trunc_normal(default)
#                 else:
#                     self.initialize_trunc_normal(default, setup['parameters'])

#             if self.distribution_1 == self.distr_log_normal or self.distribution_2 == self.distr_log_normal:
#                 if default:
#                     self.initialize_log_normal(default)
#                 else:
#                     self.initialize_log_normal(default, setup['parameters'])
            
#             if default:
#                 self.parameter_dict['weight_a'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
#                 self.parameter_dict['weight_b'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#                 self.parameter_dict['weight_c'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#                 print("Using default weight parameters")
#             else:
#                 if 'weight_a' in setup['parameters'] and 'weight_b' in setup['parameters'] and 'weight_c' in setup['parameters']:
#                     self.parameter_dict['weight_a'] = tf.Variable(setup['parameters']['weight_a'], dtype=tf.float32)
#                     self.parameter_dict['weight_b'] = tf.Variable(setup['parameters']['weight_b'], dtype=tf.float32)
#                     self.parameter_dict['weight_c'] = tf.Variable(setup['parameters']['weight_c'], dtype=tf.float32)
#                     print("Using given weight parameters")
#                 else:
#                     self.parameter_dict['weight_a'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
#                     self.parameter_dict['weight_b'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32))
#                     self.parameter_dict['weight_c'] = tf.Variable(tf.ones(1, dtype=tf.float32))
#                     print("No weight parameters found, using default values")
#         except AttributeError:
#             raise ValueError("Invalid forecast distribution: " + setup['forecast_distribution'])
    

    
#     def get_parameters(self):
#         """
#         Return the parameters of the model as a dictionary.
#         """
#         output_dict = {}
#         for parameter in self.parameter_dict:
#             output_dict[parameter] = self.parameter_dict[parameter].numpy()
#         return output_dict
    
#     def set_parameters(self, parameters):
#         """
#         Set the parameters of the model to the given values.

#         Arguments:
#         - parameters: a dictionary containing the parameters of the model.
#         """
#         for parameter in parameters:
#             if parameter in self.parameter_dict:
#                 self.parameter_dict[parameter].assign(parameters[parameter])
#                 print(f"Parameter {parameter} set to {parameters[parameter]}")
#             else:
#                 raise ValueError("Invalid parameter: " + parameter)
    
#     def to_dict(self):
#         """
#         Return the model as a dictionary, including the settings and parameters.
#         """
#         model_dict = {
#             'loss': self.loss.__name__,
#             'samples': self.samples,
#             'optimizer': self.optimizer.__class__.__name__,
#             'learning_rate': self.optimizer.learning_rate.numpy(),
#             'forecast_distribution': self.forecast_distribution.__name__,
#             'feature_mean': self.feature_mean.numpy() if self.feature_mean is not None else None,
#             'feature_std': self.feature_std.numpy() if self.feature_std is not None else None,
#             'steps_made': self.steps_made,
#             'features': self.feature_names,
#             'neighbourhood_size': self.neighbourhood_size
#         }
#         model_dict['parameters'] = self.get_parameters()
        

#         if self.need_chain:
#             if self.chain_function == self.chain_function_indicator:
#                 model_dict['chain_function'] = 'chain_function_indicator'
#                 model_dict['threshold'] = self.threshold.numpy()
#             elif self.chain_function == self.chain_function_normal_cdf:
#                 model_dict['chain_function'] = 'chain_function_normal_cdf'
#                 model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
#                 model_dict['chain_function_std'] = self.chain_function_std.numpy()

#         if hasattr(self, 'distribution_1') and hasattr(self, 'distribution_2'):
#             model_dict['distribution_1'] = self.distribution_1.__name__
#             model_dict['distribution_2'] = self.distribution_2.__name__
#             if self.forecast_distribution == self.distr_mixture:
#                 model_dict['weight'] = self.parameter_dict['weight'].numpy()
#             elif self.forecast_distribution == self.distr_mixture_linear:
#                 model_dict['weight_a'] = self.parameter_dict['weight_a'].numpy()
#                 model_dict['weight_b'] = self.parameter_dict['weight_b'].numpy()
#                 model_dict['weight_c'] = self.parameter_dict['weight_c'].numpy()
#         return model_dict
    
#     def indicator_function(self, y, t):
#         """
#         The indicator function, which returns 1 if y <= t, and 0 otherwise.

#         Arguments:
#         - y: the input value.
#         - t: the threshold.

#         Returns:
#         - 1 if y <= t, 0 otherwise.
#         """
#         return tf.cast(y <= t, tf.float32)

    
#     def distr_trunc_normal(self, X, variance):
#         """
#         The truncated normal distribution. We use a linear relationship between the parameters and the features.

#         Arguments:
#         - X: the input data.
#         - variance: the variance around the grid point, with a square grid around the neighbourhood station of size neighbourhood_size x neighbourhood_size.

#         Returns: 
#         - the truncated normal distribution.
#         """
#         mu = self.parameter_dict['a_tn'] + tf.tensordot(X, self.parameter_dict['b_tn'], axes=1)
#         sigma = tf.sqrt(tf.abs(self.parameter_dict['c_tn'] + self.parameter_dict['d_tn'] * variance))
#         return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
#     def distr_log_normal(self, X, variance):
#         """
#         The log normal distribution. We use a linear relationship between the parameters and the features.

#         Arguments:
#         - X: the input data.
#         - variance: the variance around the grid point, with a square grid around the neighbourhood station of size neighbourhood_size x neighbourhood_size.

#         Returns:
#         - the log normal distribution.
#         """
#         mu = self.parameter_dict['a_ln'] + tf.tensordot(X, self.parameter_dict['b_ln'], axes=1)
#         sigma = tf.sqrt(tf.abs(self.parameter_dict['c_ln'] + self.parameter_dict['d_ln'] * variance))
#         return tfpd.LogNormal(mu, sigma)
    
#     def distr_gev(self, X, variance):
#         location = self.parameter_dict['a_gev'] + tf.tensordot(X, self.parameter_dict['b_gev'], axes=1)
#         scale = self.parameter_dict['c_gev'] + tf.tensordot(X, self.parameter_dict['d_gev'], axes=1)  
#         shape = self.parameter_dict['e_gev'] 
#         return tfpd.GeneralizedExtremeValue(location, scale, shape)
        
#     def distr_gev2(self, X, variance):
#         location = self.parameter_dict['a_gev'] + tf.tensordot(X, self.parameter_dict['b_gev'], axes=1)
#         scale = self.parameter_dict['c_gev'] + tf.tensordot(X, self.parameter_dict['d_gev'], axes=1)  + self.parameter_dict['extra_gev'] * variance
#         shape = self.parameter_dict['e_gev'] 
#         return tfpd.GeneralizedExtremeValue(location, scale, shape)

#     def distr_gev3(self, X, variance):
#         location = self.parameter_dict['a_gev'] + tf.tensordot(X, self.parameter_dict['b_gev'], axes=1)
#         scale = self.parameter_dict['c_gev'] + tf.tensordot(X, self.parameter_dict['d_gev'], axes=1)  
#         shape = self.parameter_dict['e_gev'] + 0.001 * variance * self.parameter_dict['extra_gev']
#         return tfpd.GeneralizedExtremeValue(location, scale, shape)

#     class DistributionMixture:
#         def __init__(self, distribution_1, distribution_2, weight):
#             self.distribution_1 = distribution_1
#             self.distribution_2 = distribution_2
#             self.weight = weight

#         def log_prob(self, x):
#             return self.weight * self.distribution_1.log_prob(x) + (1 - self.weight) * self.distribution_2.log_prob(x)

#         def cdf(self, x):
#             return self.weight * self.distribution_1.cdf(x) + (1 - self.weight) * self.distribution_2.cdf(x)

#         def sample(self, n):
#             return self.weight * self.distribution_1.sample(n) + (1 - self.weight) * self.distribution_2.sample(n)    
        
#         def mean(self):
#             return self.weight * self.distribution_1.mean() + (1 - self.weight) * self.distribution_2.mean()
    
        

#     def distr_mixture(self, X, variance):
#         mixture_distr = self.DistributionMixture(self.distribution_1(X, variance), self.distribution_2(X, variance), self.parameter_dict['weight'])
#         return mixture_distr 

#     def distr_mixture_linear(self, X, variance):
#         weight = tf.math.sigmoid(self.parameter_dict['weight_a'] + tf.tensordot(X, self.parameter_dict['weight_b'], axes=1) + self.parameter_dict['weight_c'] * variance)
        
#         mixture_distr = self.DistributionMixture(self.distribution_1(X, variance), self.distribution_2(X, variance), weight)
#         return mixture_distr


    
#     def loss_log_likelihood(self, X, y, variance):
#         """
#         The loss fuction for the log likelihood, based on the forecast distribution and observations.

#         Arguments
#         - X: the input data.
#         - y: the output data.
#         - variance: the variance of the forecast distribution around the grid point.

#         Returns:
#         - the loss value.
#         """
#         forecast_distribution = self.forecast_distribution(X, variance)
#         return -tf.reduce_mean(forecast_distribution.log_prob(y))

#     def loss_CRPS_sample(self, X, y, variance):
#         """
#         The loss function for the CRPS, based on the forecast distribution and observations. We use a sample based approach to estimate the expected value of the CRPS.

#         Arguments:
#         - X: the input data.
#         - y: the output data.
#         - variance: the variance of the forecast distribution around the grid point.

#         Returns:
#         - the loss value.
#         """
#         forecast_distribution = self.forecast_distribution(X, variance)
#         X_1 = forecast_distribution.sample(self.samples)
#         X_2 = forecast_distribution.sample(self.samples)

#         # y will be broadcasted to the shape of X_1 and X_2
#         E_1 = tf.reduce_mean(tf.abs(X_1 - y), axis=0)
#         E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

#         return tf.reduce_mean(E_1 - 0.5 * E_2)
#         # E_1 = tf.norm(X_1 - y, axis=0)
#         # E_2 = tf.norm(X_1 - X_2, axis=0)
        
#         # return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
#     def loss_Brier_score(self, X, y, variance, threshold):
#         forecast_distribution = self.forecast_distribution(X, variance)
#         threshold = tf.constant(threshold, dtype=tf.float32)
#         return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - forecast_distribution.cdf(threshold)))

    
#     def loss_twCRPS_sample(self, X, y, variance):
#         return self.loss_twCRPS_sample_general(X, y, variance, self.chain_function, self.samples)
        
#     def loss_twCRPS_sample_general(self, X, y, variance, chain_function, samples):
#         forecast_distribution = self.forecast_distribution(X, variance)
#         X_1 = forecast_distribution.sample(samples)
#         X_2 = forecast_distribution.sample(samples)
#         vX_1 = chain_function(X_1)
#         vX_2 = chain_function(X_2)
#         # E_1 = tf.norm(vX_1 - chain_function(y), axis=0)
#         # E_2 = tf.norm(vX_2 - vX_1, axis=0)
#         E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(y)), axis=0)
#         E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)

#         #checken of ik ook gemiddelde kan nemen ipv reduce_sum
#         #E_1_ = tf.sqrt(tf.reduce_mean(tf.square(vX_1 - chain_function(y)) + 1.0e-20))
#         #E_2_ = tf.sqrt(tf.reduce_mean(tf.square(vX_2 - vX_1) + 1.0e-20))



#         return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
#         #return tf.reduce_mean(E_1_) - 0.5 * tf.reduce_mean(E_2_)
        

#     def chain_function_indicator(self, y):
#         """
#         Implements the chain function in case the weight function is the indicator function, which is used in weighted loss functions.

#         Arguments:
#         - y: the input value.

#         Returns:
#         - the maximum of y and the threshold.
#         """
#         return tf.maximum(y, self.threshold)
    
#     def chain_function_normal_cdf(self, y):
#         """
#         Implements the chain function in case the weight function is the normal cumulative distribution, with mean and standard deviation in the setup.

#         Arguments:
#         - y: the input value.

#         Returns:
#         - (y - mean) * cdf(y) + std^2 * pdf(y)
#         """
#         first_part = (y - self.chain_function_mean) * self.chain_normal_distr.cdf(y)
#         second_part = self.chain_function_std ** 2 * self.chain_normal_distr.prob(y)
#         return first_part + second_part   
     

#     def fit(self, X, y, variance, steps, printing = True):
#         """
#         Fit the EMOS model to the given data, using the loss function and optimizer specified in the setup.

#         Arguments:
#         - X: the input data.
#         - y: the output data.
#         - variance: the variance of the forecast around the grid point, with grid size neighbourhood_size.
#         - steps: the amount of steps to take with the optimizer.

#         Returns:
#         - hist: a list containing the loss value at each step.
#         """
#         hist = []
#         self.steps_made += steps
#         for step in range(steps):
#             with tf.GradientTape() as tape:
#                 loss_value = self.loss(X, y, variance)
#             grads = tape.gradient(loss_value, [*self.parameter_dict.values()]) 
#             # check if gradient contains nan
#             if tf.math.reduce_any(tf.math.is_nan(grads[0])):
#                 print("Gradient contains NaN")
#                 continue
#             hist.append(loss_value)

           

#             self.optimizer.apply_gradients(zip(grads, self.parameter_dict.values()))
#             if printing:
#                 if 'weight' in self.parameter_dict:
#                     print("Weight: ", self.parameter_dict['weight'].numpy())
#                 print("Step: {}, Loss: {}".format(step, loss_value))
#         return hist
            
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.models.distributions import TruncatedNormal, LogNormal, GEV, Mixture, MixtureLinear, GEV2, GEV3, distribution_name 
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
        self.feature_names = setup['features']
        self.num_features = len(self.feature_names)
        self.neighbourhood_size = setup['neighbourhood_size']

        self.init_loss(setup)
        
        if self.need_chain:
            self.init_chain_function(setup)

        self.init_optimizer(setup)

        
        self.init_forecast_distribution(setup)


        # Optionally we can initialize the feature mean and standard deviation with the given values. Not sure whether this needs to be included
        if setup['feature_mean'] is not None and setup['feature_std'] is not None:
            self.feature_mean = tf.Variable(setup['feature_mean'])
            self.feature_std = tf.Variable(setup['feature_std'])

        # Optionally we can initialize the amount of steps made with the optimizer
        if 'steps_made' in setup:
            self.steps_made = setup['steps_made']
        else:
            self.steps_made = 0

    def init_loss(self, setup):
        self.need_chain = False
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

    def init_chain_function(self, setup):
        try:
            if setup['chain_function'] == 'chain_function_indicator':
                self.chain_function = self.chain_function_indicator
                if 'threshold' not in setup:
                    raise ValueError("Threshold of the chain function not specified")
                else:
                    self.threshold = tf.constant(setup['threshold'], dtype=tf.float32)
            elif setup['chain_function'] == 'chain_function_normal_cdf':
                self.chain_function = self.chain_function_normal_cdf
                if 'chain_function_mean' not in setup:
                    raise ValueError("Mean  of the chain function not specified")
                else:
                    self.chain_function_mean = tf.constant(setup['chain_function_mean'], dtype=tf.float32)
                if 'chain_function_std' not in setup:
                    raise ValueError("Standard deviation of the chain function not specified")
                else:
                    self.chain_function_std = tf.constant(setup['chain_function_std'], dtype=tf.float32)
                self.chain_normal_distr = tfpd.Normal(self.chain_function_mean, self.chain_function_std)
        except AttributeError:
            raise ValueError("Invalid chain function: " + setup['chain_function'])  

    def init_optimizer(self, setup):
        try:
            if 'learning_rate' not in setup:
                raise ValueError("Learning rate not specified")
            self.optimizer = getattr(tf.optimizers, setup['optimizer'])(learning_rate=setup['learning_rate'])
        except AttributeError:
            raise ValueError("Invalid optimizer: " + setup['optimizer']) 
        
    def init_forecast_distribution(self, setup):
        # The setup of the forecast distribution
        if "parameters" in setup:
            parameters = setup["parameters"]
        else:
            parameters = {}

        if "forecast_distribution" not in setup:
            raise ValueError("Forecast distribution not specified")

        if distribution_name(setup["forecast_distribution"]) == "distr_trunc_normal":
            self.forecast_distribution = TruncatedNormal(self.num_features, parameters)
        elif distribution_name(setup["forecast_distribution"]) == "distr_log_normal":
            self.forecast_distribution = LogNormal(self.num_features, parameters)
        elif distribution_name(setup["forecast_distribution"]) == "distr_gev":
            self.forecast_distribution = GEV(self.num_features, parameters)
        elif distribution_name(setup["forecast_distribution"]) == "distr_gev2":
            self.forecast_distribution = GEV2(self.num_features, parameters)
        elif distribution_name(setup["forecast_distribution"]) == "distr_gev3":
            self.forecast_distribution = GEV3(self.num_features, parameters)
        elif distribution_name(setup["forecast_distribution"]) == "distr_mixture":
            if "distribution_1" in setup and "distribution_2" in setup:
                self.forecast_distribution = Mixture(self.num_features, setup["distribution_1"], setup["distribution_2"], parameters)
            else:
                raise ValueError("Please specify the distributions for the mixture")
        elif distribution_name(setup["forecast_distribution"]) == "distr_mixture_linear":
            if "distribution_1" in setup and "distribution_2" in setup:
                self.forecast_distribution = MixtureLinear(self.num_features, setup["distribution_1"], setup["distribution_2"], parameters)
            else:
                raise ValueError("Please specify the distributions for the mixture")
            
        else:
            raise ValueError("Invalid forecast distribution: " + setup['forecast_distribution'])
        
        
        
    
    def __len__(self):
        return len(self.parameter_dict)
    
    def __str__(self):
        # Loss function info
        loss_info = f"Loss function: {self.loss.__name__}"
        if hasattr(self, 'samples'):
            loss_info += f" (Samples: {self.samples})"

        # Optimizer info
        optimizer_info = f"Optimizer: {self.optimizer.__class__.__name__}"
        learning_rate_info = f"Learning rate: {self.optimizer.learning_rate.numpy()}"

        # Forecast distribution info
        forecast_distribution_info = f"Forecast distribution: {self.forecast_distribution.name()}"

        # Feature info
        feature_info = f"Features: {', '.join(self.feature_names)}"
        num_features_info = f"Number of features: {self.num_features}"
        neighbourhood_size_info = f"Neighbourhood size: {self.neighbourhood_size}"

        # Parameters info
        parameters_info = "Parameters:"
        for parameter, value in self.forecast_distribution.get_parameters().items():
            parameters_info += f"\n  {parameter}: {value}"

        # Chaining function info
        chaining_function_info = ""
        if hasattr(self, 'chain_function'):
            chaining_function_info = f"Chaining function: {self.chain_function.__name__}"
            if hasattr(self, 'threshold'):
                chaining_function_info += f" (Threshold: {self.threshold.numpy()})"
            elif hasattr(self, 'chain_function_mean') and hasattr(self, 'chain_function_std'):
                chaining_function_info += f" (Mean: {self.chain_function_mean.numpy()}, Std: {self.chain_function_std.numpy()})"

        distribution_info = ""
        if type(self.forecast_distribution) == Mixture:
            distribution_info = f"Distribution 1: {self.forecast_distribution.distribution_1.name()}\n"
            distribution_info += f"Distribution 2: {self.forecast_distribution.distribution_2.name()}\n"
            distribution_info += f"Mixture weight: {self.forecast_distribution.get_weight()}"
        elif type(self.forecast_distribution) == MixtureLinear:
            distribution_info = f"Distribution 1: {self.forecast_distribution.distribution_1.name()}\n"
            distribution_info += f"Distribution 2: {self.forecast_distribution.distribution_2.name()}\n"
            weight_a, weight_b, weight_c = self.forecast_distribution.get_weights()
            distribution_info += f"Mixture weight a: {weight_a}\n"
            distribution_info += f"Mixture weight b: {weight_b}\n"
            distribution_info += f"Mixture weight c: {weight_c}"


        return (
            f"EMOS Model Information:\n"
            f"{loss_info}\n"
            f"{forecast_distribution_info}\n"
            f"{distribution_info}"
            f"{parameters_info}\n"
            f"{feature_info}\n"
            f"{num_features_info}\n"
            f"{neighbourhood_size_info}\n"
            f"{chaining_function_info}\n"
            f"{optimizer_info}\n"
            f"{learning_rate_info}\n"
        )
    
        
    

    
    def get_parameters(self):
        """
        Return the parameters of the model as a dictionary.
        """
        return self.forecast_distribution.get_parameters()
    
    def set_parameters(self, parameters):
        """
        Set the parameters of the model to the given values.

        Arguments:
        - parameters: a dictionary containing the parameters of the model.
        """
        self.forecast_distribution.set_parameters(parameters)
    
    def to_dict(self):
        """
        Return the model as a dictionary, including the settings and parameters.
        """
        model_dict = {
            'loss' : self.loss.__name__,
            'samples': self.samples,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.learning_rate.numpy(),
            'forecast_distribution': self.forecast_distribution.name(),
            'feature_mean': self.feature_mean.numpy() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.numpy() if self.feature_std is not None else None,
            'steps_made': self.steps_made,
            'features': self.feature_names,
            'neighbourhood_size': self.neighbourhood_size
        }
        model_dict['parameters'] = self.get_parameters()

        if self.need_chain:
            if self.chain_function == self.chain_function_indicator:
                model_dict['chain_function'] = 'chain_function_indicator'
                model_dict['threshold'] = self.threshold.numpy()
            elif self.chain_function == self.chain_function_normal_cdf:
                model_dict['chain_function'] = 'chain_function_normal_cdf'
                model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
                model_dict['chain_function_std'] = self.chain_function_std.numpy()

        if type(self.forecast_distribution) == Mixture or type(self.forecast_distribution) == MixtureLinear:
            model_dict['distribution_1'] = self.forecast_distribution.distribution_1.name()
            model_dict['distribution_2'] = self.forecast_distribution.distribution_2.name()

        return model_dict

        
        # model_dict = {
        #     'loss': self.loss.__name__,
        #     'samples': self.samples,
        #     'optimizer': self.optimizer.__class__.__name__,
        #     'learning_rate': self.optimizer.learning_rate.numpy(),
        #     'forecast_distribution': self.forecast_distribution.__name__,
        #     'feature_mean': self.feature_mean.numpy() if self.feature_mean is not None else None,
        #     'feature_std': self.feature_std.numpy() if self.feature_std is not None else None,
        #     'steps_made': self.steps_made,
        #     'features': self.feature_names,
        #     'neighbourhood_size': self.neighbourhood_size
        # }
        # model_dict['parameters'] = self.get_parameters()
        

        # if self.need_chain:
        #     if self.chain_function == self.chain_function_indicator:
        #         model_dict['chain_function'] = 'chain_function_indicator'
        #         model_dict['threshold'] = self.threshold.numpy()
        #     elif self.chain_function == self.chain_function_normal_cdf:
        #         model_dict['chain_function'] = 'chain_function_normal_cdf'
        #         model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
        #         model_dict['chain_function_std'] = self.chain_function_std.numpy()

        # if hasattr(self, 'distribution_1') and hasattr(self, 'distribution_2'):
        #     model_dict['distribution_1'] = self.distribution_1.__name__
        #     model_dict['distribution_2'] = self.distribution_2.__name__
        #     if self.forecast_distribution == self.distr_mixture:
        #         model_dict['weight'] = self.parameter_dict['weight'].numpy()
        #     elif self.forecast_distribution == self.distr_mixture_linear:
        #         model_dict['weight_a'] = self.parameter_dict['weight_a'].numpy()
        #         model_dict['weight_b'] = self.parameter_dict['weight_b'].numpy()
        #         model_dict['weight_c'] = self.parameter_dict['weight_c'].numpy()
        # return model_dict
    
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
        forecast_distribution = self.forecast_distribution.get_distribution(X, variance)
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
        forecast_distribution = self.forecast_distribution.get_distribution(X, variance)
        X_1 = forecast_distribution.sample(self.samples)
        X_2 = forecast_distribution.sample(self.samples)

        # y will be broadcasted to the shape of X_1 and X_2
        E_1 = tf.reduce_mean(tf.abs(X_1 - y), axis=0)
        E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

        return tf.reduce_mean(E_1 - 0.5 * E_2)
        # E_1 = tf.norm(X_1 - y, axis=0)
        # E_2 = tf.norm(X_1 - X_2, axis=0)
        
        # return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
    def loss_Brier_score(self, X, y, variance, threshold):
        forecast_distribution = self.forecast_distribution.get_distribution(X, variance)
        threshold = tf.constant(threshold, dtype=tf.float32)
        return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - forecast_distribution.cdf(threshold)))

    
    def loss_twCRPS_sample(self, X, y, variance):
        return self.loss_twCRPS_sample_general(X, y, variance, self.chain_function, self.samples)
        
    def loss_twCRPS_sample_general(self, X, y, variance, chain_function, samples):
        forecast_distribution = self.forecast_distribution.get_distribution(X, variance)
        X_1 = forecast_distribution.sample(samples)
        X_2 = forecast_distribution.sample(samples)
        vX_1 = chain_function(X_1)
        vX_2 = chain_function(X_2)
        # E_1 = tf.norm(vX_1 - chain_function(y), axis=0)
        # E_2 = tf.norm(vX_2 - vX_1, axis=0)
        E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(y)), axis=0)
        E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)

        #checken of ik ook gemiddelde kan nemen ipv reduce_sum
        #E_1_ = tf.sqrt(tf.reduce_mean(tf.square(vX_1 - chain_function(y)) + 1.0e-20))
        #E_2_ = tf.sqrt(tf.reduce_mean(tf.square(vX_2 - vX_1) + 1.0e-20))



        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
        #return tf.reduce_mean(E_1_) - 0.5 * tf.reduce_mean(E_2_)
        

    def chain_function_indicator(self, y):
        """
        Implements the chain function in case the weight function is the indicator function, which is used in weighted loss functions.

        Arguments:
        - y: the input value.

        Returns:
        - the maximum of y and the threshold.
        """
        return tf.maximum(y, self.threshold)
    
    def chain_function_normal_cdf(self, y):
        """
        Implements the chain function in case the weight function is the normal cumulative distribution, with mean and standard deviation in the setup.

        Arguments:
        - y: the input value.

        Returns:
        - (y - mean) * cdf(y) + std^2 * pdf(y)
        """
        first_part = (y - self.chain_function_mean) * self.chain_normal_distr.cdf(y)
        second_part = self.chain_function_std ** 2 * self.chain_normal_distr.prob(y)
        return first_part + second_part  

    def compute_loss_and_gradient(self, X, y, variance):
        with tf.GradientTape() as tape:
            loss_value = self.loss(X, y, variance)
        grads = tape.gradient(loss_value, [*self.forecast_distribution.get_parameter_dict().values()])
        return loss_value, grads

     

    def fit(self, X, y, variance, steps, printing = True):
        """
        Fit the EMOS model to the given data, using the loss function and optimizer specified in the setup.

        Arguments:
        - X: the input data.
        - y: the output data.
        - variance: the variance of the forecast around the grid point, with grid size neighbourhood_size.
        - steps: the amount of steps to take with the optimizer.

        Returns:
        - hist: a list containing the loss value at each step.
        """
        hist = []
        self.steps_made += steps
        for step in range(steps):
            loss_value, grads = self.compute_loss_and_gradient(X, y, variance)

            # check if gradient contains nan
            if tf.math.reduce_any(tf.math.is_nan(grads[0])):
                print("Gradient contains NaN")
                continue
            hist.append(loss_value)

            self.optimizer.apply_gradients(zip(grads, self.forecast_distribution.get_parameter_dict().values()))

            if printing:
                if "weight" in self.forecast_distribution.get_parameter_dict():
                    print("Weight: ", self.forecast_distribution.get_parameter_dict()['weight'].numpy())
                print("Step: {}, Loss: {}".format(step, loss_value))
        return hist
            