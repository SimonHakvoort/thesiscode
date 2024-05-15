import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.models.forecast_distributions import TruncatedNormal, LogNormal, GEV, Mixture, MixtureLinear, GEV2, GEV3, distribution_name, initialize_distribution 
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
        self.all_features = copy.deepcopy(setup['all_features'])
        self.location_features = copy.deepcopy(setup['location_features'])
        self.scale_features = copy.deepcopy(setup['scale_features'])

        self.num_features = len(self.all_features)
        
        if 'neighbourhood_size' not in setup:
            self.neighbourhood_size = None
        else:
            self.neighbourhood_size = setup['neighbourhood_size']

        self._init_loss(setup)
        
        if self.need_chain:
            self._init_chain_function(setup)

        self._init_optimizer(setup)
        
        self._init_forecast_distribution(setup)

        self.hist = []
        if 'history' in setup:
            self.hist = setup['history']


        # Optionally we can initialize the feature mean and standard deviation with the given values. Not sure whether this needs to be included
        if 'feature_mean' in setup and 'feature_std' in setup:
            if setup['feature_mean'] is not None and setup['feature_std'] is not None:
                self.feature_mean = tf.Variable(setup['feature_mean'])
                self.feature_std = tf.Variable(setup['feature_std'])
        else:
            self.feature_mean = None
            self.feature_std = None

        # Optionally we can initialize the amount of steps made with the optimizer
        if 'steps_made' in setup:
            self.steps_made = setup['steps_made']
        else:
            self.steps_made = 0

        if 'folds' in setup:
            self.folds = setup['folds']

    def _init_loss(self, setup):
        """
        Setup the loss function of the model.

        Arguments:
        - setup: a dictionary containing the setup for the model.

        The setup should contain the following keys:
        - loss: the loss function used to fit the model
        - samples: the amount of samples used in the loss function in case we use a sample based loss function
        """
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

    def _init_chain_function(self, setup):
        try:
            if setup['chain_function'] == 'chain_function_indicator':
                self.chain_function = self.chain_function_indicator
                if 'chain_function_threshold' not in setup:
                    raise ValueError("Threshold of the chain function not specified")
                else:
                    self.chain_function_threshold = tf.constant(setup['chain_function_threshold'], dtype=tf.float32)
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
            elif setup['chain_function'] == 'chain_function_normal_cdf_plus_constant':
                self.chain_function = self.chain_function_normal_cdf_plus_constant
                if 'chain_function_mean' not in setup:
                    raise ValueError("Mean  of the chain function not specified")
                else:
                    self.chain_function_mean = tf.constant(setup['chain_function_mean'], dtype=tf.float32)
                if 'chain_function_std' not in setup:
                    raise ValueError("Standard deviation of the chain function not specified")
                else:
                    self.chain_function_std = tf.constant(setup['chain_function_std'], dtype=tf.float32)
                self.chain_normal_distr = tfpd.Normal(self.chain_function_mean, self.chain_function_std)
                if 'chain_function_constant' not in setup:
                    raise ValueError("Constant of the chain function not specified")
                else:
                    self.chain_function_constant = tf.constant(setup['chain_function_constant'], dtype=tf.float32)
        except AttributeError:
            raise ValueError("Invalid chain function: " + setup['chain_function'])  

    def _init_optimizer(self, setup):
        try:
            if 'learning_rate' not in setup:
                raise ValueError("Learning rate not specified")
            learning_rate = round(float(setup['learning_rate']), 6)
            self.optimizer = getattr(tf.optimizers, setup['optimizer'])(learning_rate=learning_rate)
        except AttributeError:
            raise ValueError("Invalid optimizer: " + setup['optimizer']) 
        
    def _init_forecast_distribution(self, setup):
        # The setup of the forecast distribution
        if "parameters" in setup:
            parameters = setup["parameters"]
        else:
            parameters = {}

        if "forecast_distribution" not in setup:
            raise ValueError("Forecast distribution not specified")

        if "location_features" not in setup:
            raise ValueError("Location features not specified")
        
        if "scale_features" not in setup:
            raise ValueError("Scale features not specified")

        if "all_features" not in setup:
            raise ValueError("All features not specified")

        random_init = False
        if "random_init" in setup:
            random_init = setup["random_init"]

        distribution_1 = None
        distribution_2 = None

        if setup['forecast_distribution'] == 'distr_mixture' or setup['forecast_distribution'] == 'distr_mixture_linear':
            if 'distribution_1' not in setup or 'distribution_2' not in setup:
                raise ValueError("Please specify the distributions for the mixture")
            else:
                distribution_1 = setup['distribution_1']
                distribution_2 = setup['distribution_2']
        
        self.forecast_distribution = initialize_distribution(
            setup['forecast_distribution'], 
            setup["all_features"], 
            setup["location_features"],
            setup["scale_features"],
            parameters, 
            random_init,
            distribution_1,
            distribution_2)        
        
    
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
        feature_info = f"Features: {', '.join(self.all_features)}"
        location_features = f"Location features: {', '.join(self.location_features)}"
        scale_features = f"Scale features: {', '.join(self.scale_features)}"
        num_features_info = f"Number of features: {self.num_features}"
        neighbourhood_size_info = f"Neighbourhood size: {self.neighbourhood_size}"

        # Parameters info
        parameters_info = "Parameters:"
        for parameter, value in self.forecast_distribution.parameters.items():
            parameters_info += f"\n  {parameter}: {value}"

        # Chaining function info
        chaining_function_info = ""
        if hasattr(self, 'chain_function'):
            chaining_function_info = f"Chaining function: {self.chain_function.__name__}"
            if hasattr(self, 'chain_function_threshold'):
                chaining_function_info += f" (Threshold: {self.chain_function_threshold.numpy()})"
            elif hasattr(self, 'chain_function_mean') and hasattr(self, 'chain_function_std'):
                chaining_function_info += f" (Mean: {self.chain_function_mean.numpy()}, Std: {self.chain_function_std.numpy()})"
                if hasattr(self, 'chain_function_constant'):
                    chaining_function_info += f", Constant: {self.chain_function_constant.numpy()}"

        distribution_info = ""
        if type(self.forecast_distribution) == Mixture:
            distribution_info = f"Distribution 1: {self.forecast_distribution.distribution_1.name()}\n"
            distribution_info += f"Distribution 2: {self.forecast_distribution.distribution_2.name()}\n"
            distribution_info += f"Mixture weight: {self.forecast_distribution.get_weight()}\n"
        elif type(self.forecast_distribution) == MixtureLinear:
            distribution_info = f"Distribution 1: {self.forecast_distribution.distribution_1.name()}\n"
            distribution_info += f"Distribution 2: {self.forecast_distribution.distribution_2.name()}\n"
            #weight_a, weight_b, weight_c = self.forecast_distribution.get_weights()
            weight_a, weight_b = self.forecast_distribution.get_weights()
            distribution_info += f"Mixture weight a: {weight_a}\n"
            distribution_info += f"Mixture weight b: {weight_b}\n"
            #distribution_info += f"Mixture weight c: {weight_c}\n"

        steps_info = ""
        if self.steps_made > 0:
            steps_info = f"Steps made: {self.steps_made}"
        

        return (
            f"EMOS Model Information:\n"
            f"{loss_info}\n"
            f"{forecast_distribution_info}\n"
            f"{distribution_info}"
            f"{parameters_info}\n"
            f"{feature_info}\n"
            f"{location_features}\n"
            f"{scale_features}\n"
            f"{num_features_info}\n"
            f"{neighbourhood_size_info}\n"
            f"{chaining_function_info}\n"
            f"{optimizer_info}\n"
            f"{learning_rate_info}\n"
        )
    
        
    

    
    def get_parameters(self):
        """
        Return the parameters as np.arrays of the model in a dictionary.
        """
        return self.forecast_distribution.parameters
    
    def set_parameters(self, parameters):
        """
        Set the parameters of the model to the given values.

        Arguments:
        - parameters: a dictionary containing the parameters of the model.
        """
        self.forecast_distribution.parameters = parameters
    
    def to_dict(self):
        """
        Return the model as a dictionary, including the settings and parameters.
        Can be used to save the settings and parameters of the model.
        """
        model_dict = {
            'loss' : self.loss.__name__,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.learning_rate.numpy(),
            'forecast_distribution': self.forecast_distribution.name(),
            'feature_mean': self.feature_mean.numpy() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.numpy() if self.feature_std is not None else None,
            'steps_made': self.steps_made,
            'all_features': self.all_features,
            'location_features': self.location_features,
            'scale_features': self.scale_features,
            'neighbourhood_size': self.neighbourhood_size
        }
        model_dict['parameters'] = self.get_parameters()

        if hasattr(self, 'samples'):
            model_dict['samples'] = self.samples

        if self.need_chain:
            if self.chain_function == self.chain_function_indicator:
                model_dict['chain_function'] = 'chain_function_indicator'
                model_dict['chain_function_threshold'] = self.chain_function_threshold.numpy()
            elif self.chain_function == self.chain_function_normal_cdf:
                model_dict['chain_function'] = 'chain_function_normal_cdf'
                model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
                model_dict['chain_function_std'] = self.chain_function_std.numpy()
            elif self.chain_function == self.chain_function_normal_cdf_plus_constant:
                model_dict['chain_function'] = 'chain_function_normal_cdf_plus_constant'
                model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
                model_dict['chain_function_std'] = self.chain_function_std.numpy()
                model_dict['chain_function_constant'] = self.chain_function_constant.numpy()

        if type(self.forecast_distribution) == Mixture or type(self.forecast_distribution) == MixtureLinear:
            model_dict['distribution_1'] = self.forecast_distribution.distribution_1.name()
            model_dict['distribution_2'] = self.forecast_distribution.distribution_2.name()

        model_dict['history'] = self.hist

        return model_dict
    
    def normalize_features(self, X):
        """
        Normalize the features of the model.

        Arguments:
        - X: the input data of shape (n, m), where n is the number of samples and m is the number of features.

        Returns:
        - the normalized input data.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Feature mean and standard deviation not set")
        return (X - self.feature_mean) / self.feature_std

    
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
 
    def loss_log_likelihood(self, X, y):
        """
        The loss fuction for the log likelihood, based on the forecast distribution and observations.

        Arguments:
        - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
        - y (tf.Tensor): the observations of shape (n,).
        - variance (tf.Tensor): the variance of the forecast distribution around the grid point of shape (n,).

        Returns:
        - the loss value.
        """
        forecast_distribution = self.forecast_distribution.get_distribution(X)
        return -tf.reduce_mean(forecast_distribution.log_prob(y))
    
    def get_prob_distribution(self, data):
        # data is a tf.data.Dataset

        # for X, y in data:
        #     distributions = self.forecast_distribution.get_distribution(X['features_emos'])
        #     observations = y
        X, y = next(iter(data))
        distributions = self.forecast_distribution.get_distribution(X['features_emos'])
        observations = y

        return distributions, observations
    
    def CRPS_tfdataset(self, data, samples):
        total_crps = 0
        for X, y in data:
            total_crps += self.CRPS(X['features_emos'], y, samples)
        return total_crps

    
    def CRPS(self, X, y, samples):
        """
        The loss function for the CRPS, based on the forecast distribution and observations.
        We use a sample based approach to estimate the expected value of the CRPS.

        Arguments:
        - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
        - y (tf.Tensor): the observations of shape (n,).
        - variance (tf.Tensor): the variance of the forecast distribution around the grid point of shape (n,).
        - samples (int): the amount of samples used to estimate the expected value of the CRPS.

        Returns:
        - the loss value.
        """
        forecast_distribution = self.forecast_distribution.get_distribution(X)
        #X_1 has shape (samples, n), where n is the number of observations
        X_1 = forecast_distribution.sample(samples)
        X_2 = forecast_distribution.sample(samples)

        # y will be broadcasted to the shape of X_1 and X_2
        E_1 = tf.reduce_mean(tf.abs(X_1 - y), axis=0)
        E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

        return tf.reduce_mean(E_1 - 0.5 * E_2)
    
    def loss_cPIT(self, X, y):
        # return self.calc_pit from t = 0 to t = 15
        return tf.reduce_mean([self.calc_cPIT(X, y, t) for t in range(16)])
    
    def calc_cPIT(self, X, y, t):
        indices = tf.where(y > t)
        indices = tf.reshape(indices, [-1])

        y_greater = tf.gather(y, indices)
        X_greater = tf.gather(X, indices)

        cdf = self.forecast_distribution.get_distribution(X_greater).cdf

        if t == 0:
            probabilities = cdf(y_greater)
        elif t > 0:
            #probabilities = (cdf(observations) - cdf(t)) / (1 - cdf(t))
            upper = cdf(y_greater) - cdf(t)
            lower = 1 - cdf(t)
            # remove the points where lower is 0
            mask = tf.where(lower == 0, False, True)
            upper = tf.boolean_mask(upper, mask)
            lower = tf.boolean_mask(lower, mask)
            probabilities = upper / lower

        probabilities = tf.sort(probabilities)

        return tf.reduce_mean(tf.abs(probabilities - tf.linspace(0.0, 1.0, tf.shape(probabilities)[0])))

    def loss_CRPS_sample(self, X, y):
        return self.CRPS(X, y, self.samples)
        
    
    def Brier_Score(self, X, y, threshold):
        """
        The loss function for the Brier score, based on the forecast distribution and observations.

        Arguments:
        - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
        - y (tf.Tensor): the observations of shape (n,).
        - variance (tf.Tensor): the variance of the forecast distribution around the grid point of shape (n,).
        - threshold (float): the threshold for the Brier score.

        Returns:
        - the Brier score at the given threshold.
        """
        
        cdf_values = self.forecast_distribution.comp_cdf(X, threshold)
        return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - cdf_values))
    
    def Brier_Score_tfdataset(self, data, threshold):
        """
        The loss function for the Brier score, based on the forecast distribution and observations, using a tf.dataset.

        Arguments:
        - data (tf.dataset): the dataset containing the input data and observations.
        - threshold (float): the threshold for the Brier score.

        Returns:
        - the Brier score at the given threshold.
        """
        # take 1 element from the data
        X, y = next(iter(data))
        brier_score = self.Brier_Score(X['features_emos'], y, threshold)
            

        return brier_score



    def twCRPS(self, X, y, threshold, samples):
        chain_function = lambda x: self.chain_function_indicator_general(x, threshold)
        return self.loss_twCRPS_sample_general(X, y, chain_function, samples)
    
    def twCRPS_tfdataset(self, data, threshold, samples):
        chain_function = lambda x: self.chain_function_indicator_general(x, threshold)
        X, y = next(iter(data))
        return self.loss_twCRPS_sample_general(X['features_emos'], y, chain_function, samples)
    
    # def twCRPS_tfdataset(self, data, threshold, samples):
    #     chain_function = lambda x: self.chain_function_indicator_general(x, threshold)
    #     return self.loss_twCRPS_tfdataset_general(data, chain_function, samples)
    
    def loss_twCRPS_sample(self, X, y):
        return self.loss_twCRPS_sample_general(X, y, self.chain_function, self.samples)
        
    def loss_twCRPS_sample_general(self, X, y, chain_function, samples):
        forecast_distribution = self.forecast_distribution.get_distribution(X)
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
    
    # def loss_twCRPS_tfdataset_general(self, data, chain_function, samples):
    #     twcrps = 0
    #     count = 0
    #     for X, y in data:
    #         forecast_distribution = self.forecast_distribution.get_distribution(X)
    #         X_1 = forecast_distribution.sample(samples)
    #         X_2 = forecast_distribution.sample(samples)
    #         vX_1 = chain_function(X_1)
    #         vX_2 = chain_function(X_2)
    #         E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(y)), axis=0)
    #         E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)
    #         twcrps += tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    #         count += 1
    #     return twcrps / count
        

    def chain_function_indicator(self, y):
        """
        Implements the chain function in case the weight function is the indicator function, which is used in weighted loss functions.

        Arguments:
        - y: the input value.

        Returns:
        - the maximum of y and the threshold.
        """
        return self.chain_function_indicator_general(y, self.chain_function_threshold)
    
    def chain_function_indicator_general(self, y, threshold):
        return tf.maximum(y, threshold)
    
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
    
    def chain_function_normal_cdf_plus_constant(self, y):
        """
        Implements the chain function in case the weight function is the normal cumulative distribution, with mean and standard deviation in the setup, and a constant.

        Arguments:
        - y: the input value.

        Returns:
        - (y - mean) * cdf(y) + std^2 * pdf(y) + constant
        """
        first_part = (y - self.chain_function_mean) * self.chain_normal_distr.cdf(y)
        second_part = self.chain_function_std ** 2 * self.chain_normal_distr.prob(y)
        return first_part + second_part + self.chain_function_constant * y

    def _compute_loss_and_gradient(self, X, y):
        """
        Compute the loss and the gradient of the loss with respect to the parameters of the model, 
        which are the parameters of the forecast distribution.

        Arguments:
        - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
        - y (tf.Tensor): the observations of shape (n,).
        - variance (tf.Tensor): the variance of the forecast distribution around the grid point of shape (n,).

        Returns:
        - loss_value: the loss value.
        - grads: the gradients of the loss with respect to the parameters of the model.
        """
        with tf.GradientTape() as tape:
            loss_value = self.loss(X, y)
        grads = tape.gradient(loss_value, [*self.forecast_distribution.parameter_dict.values()])
        return loss_value, grads
    
    #@tf.function
    def _train_step(self, X, y):
        loss_value, grads = self._compute_loss_and_gradient(X, y)
        if tf.math.reduce_any(tf.math.is_nan(grads[0])):
            return -1.0
        self.optimizer.apply_gradients(zip(grads, self.forecast_distribution.parameter_dict.values()))
        return loss_value
    
    def predict(self, X):
        """
        Predict the forecast distribution for the given input data.

        Arguments:
        - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.

        Returns:
        - the forecast distribution.
        """
        return self.forecast_distribution.get_distribution(X)


    # def fit_dataset(self, dataset, steps, printing = True, subset_size = None):
     

    def fit(self, X, y, steps, printing = True, subset_size = None):
        """
        Fit the EMOS model to the given data, using the loss function and optimizer specified in the setup.

        Arguments:
        - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
        - y (tf.Tensor): the observations of shape (n,).
        - steps: the amount of steps to take with the optimizer.
        - printing: whether to print the loss value at each step.
        - subset_size: the size of the random subset used to compute the loss and gradient. If None, the full dataset is used for each step.

        Returns:
        - hist: a list containing the loss value at each step.
        """
        hist = []
        self.steps_made += steps

        if subset_size is not None:
            if subset_size > X.shape[0]:
                raise ValueError("Subset size is larger than the dataset size")
            
        num_steps = steps
        if subset_size is not None:
            num_steps = steps * X.shape[0] // subset_size

        for step in range(num_steps):

            if subset_size is not None:
                indices = np.random.choice(X.shape[0], subset_size, replace=False)
                X_subset = tf.gather(X, indices)
                y_subset = tf.gather(y, indices)
            else:
                X_subset = X
                y_subset = y

            loss_value, grads = self._compute_loss_and_gradient(X_subset, y_subset)

            # check if gradient contains nan
            if tf.math.reduce_any(tf.math.is_nan(grads[0])):
                print("Gradient contains NaN")
                continue
            hist.append(loss_value)

            self.optimizer.apply_gradients(zip(grads, self.forecast_distribution.parameter_dict.values()))

            if printing:
                if "weight" in self.forecast_distribution.parameter_dict:
                    print("Weight: ", self.forecast_distribution.parameter_dict['weight'].numpy())
                print("Step: {}, Loss: {}".format(step, loss_value))
        print("Final loss: ", loss_value.numpy())	
        return hist
    
    ### This function is currently slower than the fit function, except if we use batching
    def fit_tfdataset(self, data, epochs, printing = True):
        for epoch in range(epochs):
            epoch_losses = 0.0
            batch_count = 0.0
            for X, y in data:
                loss_value = self._train_step(X['features_emos'], y)
                epoch_losses += loss_value
                batch_count += 1.0
            epoch_mean_loss = epoch_losses / batch_count
            self.hist.append(epoch_mean_loss)
            if printing:
                tf.print("Epoch: ", epoch, " Loss: ", epoch_mean_loss)
        return self.hist
                
            



class BootstrapEmos():
    def __init__(self, setup):
        self.setup = setup
        self.num_samples = setup['num_samples']
        self.models = [EMOS(setup) for _ in range(self.num_samples)]

    def fit(self, X, y, steps, printing = True, subset_size = None):
        for model in self.models:
            # sample data from X, y
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample = tf.gather(X, indices)
            y_sample = tf.gather(y, indices)
            model.fit(X_sample, y_sample, steps, printing, subset_size)

    # def Brier_Score_Bootstrap(self, X, y, threshold):






