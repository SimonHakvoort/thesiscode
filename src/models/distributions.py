import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

class ForecastDistribution:
    """
    Virtual base class for forecast distributions.

    This class serves as a template for implementing specific EMOS forecast distribution models.
    Subclasses should override the `get_distribution` method to provide functionality for
    generating distribution objects based on input data and variance.

    Attributes:
        num_features (int): Number of features used in the model.
        parameter_dict (dict): Dictionary containing the parameters of the distribution.
    """

    def __init__(self, num_features):
        self.parameter_dict = {}
        self.num_features = num_features

    def get_distribution(self, X, variance):
        """
        Returns a distribution object given the input and variance.

        Args:
        - X (tf.Tensor): Input data
        - variance (tf.Tensor): Variance around each gridpoint

        Returns:
        - tfp.distributions.Distribution/Mixture: The forecast distribution
        """
        pass

    def get_parameter_dict(self):
        """
        Returns the parameters of the distribution as tf.Variables in a dictionary.
        """
        return self.parameter_dict
    
    def get_parameters(self):
        """
        Returns the parameters of the distribution as np.arrays in a dictionary.
        """
        parameters = {}
        for key, value in self.parameter_dict.items():
            parameters[key] = value.numpy()
        return parameters
        
    
    def set_parameters(self, parameters):
        """
        Sets the parameters of the distribution to the given values.

        Args:
        - parameters (dict): Dictionary with parameter names as keys and values as values. 

        Returns:
        - None
        """
        for key, value in parameters.items():
            if key in self.parameter_dict:
                self.parameter_dict[key].assign(value)
                print("Parameter {0} set to {1}".format(key, value))



    
class TruncatedNormal(ForecastDistribution):
    """
    Forecast distribution representing a truncated normal EMOS distribution.

    This class implements a truncated normal distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data and variance.
    It assumes linear relationship between the distribution parameters and the input data.

    Attributes:
        num_features (int): Number of features used in the model.
        parameter_dict (dict): Dictionary containing the parameters of the distribution.
    """
    def __init__(self, num_features, parameters = {}):
        """
        Constructor for the TruncatedNormal class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
        - num_features (int): Number of features used in the model.
        - parameters (dict): Dictionary containing the parameters of the distribution.

        Returns:
        - None
        """
        super().__init__(num_features)
        if "a_tn" in parameters and "b_tn" in parameters and "c_tn" in parameters and "d_tn" in parameters:
            self.parameter_dict["a_tn"] = tf.Variable(parameters["a_tn"], dtype = tf.float32, name="a_tn")
            self.parameter_dict["b_tn"] = tf.Variable(parameters["b_tn"], dtype = tf.float32, name="b_tn")
            self.parameter_dict["c_tn"] = tf.Variable(parameters["c_tn"], dtype = tf.float32, name="c_tn")
            self.parameter_dict["d_tn"] = tf.Variable(parameters["d_tn"], dtype = tf.float32, name="d_tn")
            print("Using given parameters for Truncated Normal distribution")
        else:
            self.parameter_dict['a_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_tn"))
            self.parameter_dict['b_tn'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="b_tn")
            self.parameter_dict['c_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_tn")
            self.parameter_dict['d_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="d_tn")
            print("Using default parameters for truncated normal distribution")

    def get_distribution(self, X, variance):
        mu = self.parameter_dict['a_tn'] + tf.tensordot(X, self.parameter_dict['b_tn'], axes=1)
        sigma = tf.sqrt(tf.abs(self.parameter_dict['c_tn'] + self.parameter_dict['d_tn'] * variance))
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def __str__(self):
        info = "Truncated Normal distribution with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self):
        return "distr_trunc_normal"
    
class LogNormal(ForecastDistribution):
    """
=    Forecast distribution representing a lognormal EMOS distribution.

    This class implements a truncated normal distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data and variance.
    It assumes linear relationship between the distribution parameters and the input data.

    Attributes:
        num_features (int): Number of features used in the model.
        parameter_dict (dict): Dictionary containing the parameters of the distribution.
    """
    def __init__(self, num_features, parameters = {}):
        super().__init__(num_features)
        if "a_ln" in parameters and "b_ln" in parameters and "c_ln" in parameters:
            self.parameter_dict["a_ln"] = tf.Variable(parameters["a_ln"], dtype = tf.float32, name="a_ln")
            self.parameter_dict["b_ln"] = tf.Variable(parameters["b_ln"], dtype = tf.float32, name="b_ln")
            self.parameter_dict["c_ln"] = tf.Variable(parameters["c_ln"], dtype = tf.float32, name="c_ln")
            self.parameter_dict["d_ln"] = tf.Variable(parameters["d_ln"], dtype = tf.float32, name="d_ln")
            print("Using given parameters for Log Normal distribution")
        else:
            self.parameter_dict['a_ln'] = tf.Variable(tf.zeros(1, dtype=tf.float32, name="a_ln"))
            self.parameter_dict['b_ln'] = tf.Variable(tf.zeros(self.num_features, dtype=tf.float32), name="b_ln")
            self.parameter_dict['c_ln'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_ln")
            self.parameter_dict['d_ln'] = tf.Variable(tf.zeros(1, dtype=tf.float32), name="d_ln")
            print("Using default parameters for Log Normal distribution")

    def get_distribution(self, X, variance):
        mu = self.parameter_dict['a_ln'] + tf.tensordot(X, self.parameter_dict['b_ln'], axes=1)
        sigma = tf.sqrt(tf.abs(self.parameter_dict['c_ln'] + self.parameter_dict['d_ln'] * variance))
        return tfpd.LogNormal(mu, sigma)
    
    def __str__(self):
        info = "Log Normal distribution with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self):
        return "distr_log_normal"
    
class GEV(ForecastDistribution):
    """
    Forecast distribution representing a truncated normal EMOS distribution.

    This class implements a generalized extreme value distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data and variance.
    It assumes linear relationship between the distribution parameters and the input data.
    This class does not use the variance in the distribution parameters.

    Attributes:
        num_features (int): Number of features used in the model.
        parameter_dict (dict): Dictionary containing the parameters of the distribution.
    """
    def __init__(self, num_features, parameters = {}):
        super().__init__(num_features)
        if "a_gev" in parameters and "b_gev" in parameters and "c_gev" in parameters and "d_gev" in parameters and "e_gev" in parameters:
            self.parameter_dict["a_gev"] = tf.Variable(parameters["a_gev"], dtype = tf.float32, name="a_gev")
            self.parameter_dict["b_gev"] = tf.Variable(parameters["b_gev"], dtype = tf.float32, name="b_gev")
            self.parameter_dict["c_gev"] = tf.Variable(parameters["c_gev"], dtype = tf.float32, name="c_gev")
            self.parameter_dict["d_gev"] = tf.Variable(parameters["d_gev"], dtype = tf.float32, name="d_gev")
            self.parameter_dict["e_gev"] = tf.Variable(parameters["e_gev"], dtype = tf.float32, name="e_gev")
            print("Using given parameters for Generalized Extreme Value distribution")
        else:
            self.parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_gev"))
            self.parameter_dict['b_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="b_gev")
            self.parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_gev")
            self.parameter_dict['d_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="d_gev")
            self.parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.3, name="e_gev")
            print("Using default parameters for Generalized Extreme Value distribution")

    def get_distribution(self, X, variance):
        location = self.parameter_dict['a_gev'] + tf.tensordot(X, self.parameter_dict['b_gev'], axes=1)
        scale = self.parameter_dict['c_gev'] + tf.tensordot(X, self.parameter_dict['d_gev'], axes=1)  
        shape = self.parameter_dict['e_gev'] 
        return tfpd.GeneralizedExtremeValue(location, scale, shape)

    def __str__(self):
        info = "Generalized Extreme Value distribution with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self):
        return "distr_gev"
    
class GEV2(ForecastDistribution):
    """
    Forecast distribution representing a truncated normal EMOS distribution.

    This class implements a generalized extreme value distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data and variance.
    It assumes linear relationship between the distribution parameters and the input data.
    This class does use the variance in the scale of the distribution.
    Attributes:
        num_features (int): Number of features used in the model.
        parameter_dict (dict): Dictionary containing the parameters of the distribution.
    """
    def __init__(self, num_features, parameters = {}):
        super().__init__(num_features)
        if "a_gev" in parameters and "b_gev" in parameters and "c_gev" in parameters and "d_gev" in parameters and "e_gev" in parameters:
            self.parameter_dict["a_gev"] = tf.Variable(parameters["a_gev"], dtype = tf.float32, name="a_gev")
            self.parameter_dict["b_gev"] = tf.Variable(parameters["b_gev"], dtype = tf.float32, name="b_gev")
            self.parameter_dict["c_gev"] = tf.Variable(parameters["c_gev"], dtype = tf.float32, name="c_gev")
            self.parameter_dict["d_gev"] = tf.Variable(parameters["d_gev"], dtype = tf.float32, name="d_gev")
            self.parameter_dict["e_gev"] = tf.Variable(parameters["e_gev"], dtype = tf.float32, name="e_gev")

            self.parameter_dict["extra_gev"] = tf.Variable(parameters["extra_gev"], dtype = tf.float32, name="extra_gev")
            print("Using given parameters for Generalized Extreme Value distribution 2")
        else:
            self.parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_gev"))
            self.parameter_dict['b_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="b_gev")
            self.parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_gev")
            self.parameter_dict['d_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="d_gev")
            self.parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.3, name="e_gev")

            self.parameter_dict['extra_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="extra_gev")
            print("Using default parameters for Generalized Extreme Value distribution 2")

    def get_distribution(self, X, variance):
        location = self.parameter_dict['a_gev'] + tf.tensordot(X, self.parameter_dict['b_gev'], axes=1)
        scale = self.parameter_dict['c_gev'] + tf.tensordot(X, self.parameter_dict['d_gev'], axes=1) + self.parameter_dict['extra_gev'] * variance
        shape = self.parameter_dict['e_gev'] 
        return tfpd.GeneralizedExtremeValue(location, scale, shape)
    
    def __str__(self):
        info = "Generalized Extreme Value distribution 2 with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self):
        return "distr_gev2"
    
class GEV3(ForecastDistribution):
    """
    Forecast distribution representing a truncated normal EMOS distribution.

    This class implements a generalized extreme value distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data and variance.
    It assumes linear relationship between the distribution parameters and the input data.
    This class does use the variance in the shape of the distribution.

    Attributes:
        num_features (int): Number of features used in the model.
        parameter_dict (dict): Dictionary containing the parameters of the distribution.
    """
    def __init__(self, num_features, parameters = {}):
        super().__init__(num_features)
        if "a_gev" in parameters and "b_gev" in parameters and "c_gev" in parameters and "d_gev" in parameters and "e_gev" in parameters:
            self.parameter_dict["a_gev"] = tf.Variable(parameters["a_gev"], dtype = tf.float32, name="a_gev")
            self.parameter_dict["b_gev"] = tf.Variable(parameters["b_gev"], dtype = tf.float32, name="b_gev")
            self.parameter_dict["c_gev"] = tf.Variable(parameters["c_gev"], dtype = tf.float32, name="c_gev")
            self.parameter_dict["d_gev"] = tf.Variable(parameters["d_gev"], dtype = tf.float32, name="d_gev")
            self.parameter_dict["e_gev"] = tf.Variable(parameters["e_gev"], dtype = tf.float32, name="e_gev")

            self.parameter_dict["extra_gev"] = tf.Variable(parameters["extra_gev"], dtype = tf.float32, name="extra_gev")
            print("Using given parameters for Generalized Extreme Value distribution 3")
        else:
            self.parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_gev"))
            self.parameter_dict['b_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="b_gev")
            self.parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_gev")
            self.parameter_dict['d_gev'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="d_gev")
            self.parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.3, name="e_gev")

            self.parameter_dict['extra_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="extra_gev")
            print("Using default parameters for Generalized Extreme Value distribution 3")

    def get_distribution(self, X, variance):
        location = self.parameter_dict['a_gev'] + tf.tensordot(X, self.parameter_dict['b_gev'], axes=1)
        scale = self.parameter_dict['c_gev'] + tf.tensordot(X, self.parameter_dict['d_gev'], axes=1) 
        shape = self.parameter_dict['e_gev'] + 0.001 * self.parameter_dict['extra_gev'] * variance
        return tfpd.GeneralizedExtremeValue(location, scale, shape)
    
    def __str__(self):
        info = "Generalized Extreme Value distribution 3 with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self):
        return "distr_gev3"

    
class DistributionMixture:
    """
    A class representing a mixture of two distributions.
    
    Attributes:
        distribution_1 (tfp.distributions.Distribution): The first distribution in the mixture
        distribution_2 (tfp.distributions.Distribution): The second distribution in the mixture
        weight (tf.Tensor): The weight of the first distribution in the mixture
    """
    def __init__(self, distribution_1, distribution_2, weight):
        self.distribution_1 = distribution_1
        self.distribution_2 = distribution_2
        self.weight = weight

    def log_prob(self, x):
        return self.weight * self.distribution_1.log_prob(x) + (1 - self.weight) * self.distribution_2.log_prob(x)

    def cdf(self, x):
        return self.weight * self.distribution_1.cdf(x) + (1 - self.weight) * self.distribution_2.cdf(x)

    def sample(self, n):
        return self.weight * self.distribution_1.sample(n) + (1 - self.weight) * self.distribution_2.sample(n)    
    
    def mean(self):
        return self.weight * self.distribution_1.mean() + (1 - self.weight) * self.distribution_2.mean()


### In case a new distribution is added, the following functions need to be updated:
def initialize_distribution(distribution, num_features, parameters, distribution_1 = None, distribution_2 = None):
    """
    Initializes the given distribution based on the input.

    Args:
    - distribution (str): The name of the distribution
    - num_features (int): The number of features used in the model
    - parameters (dict): Dictionary containing the parameters of the distribution, which is optional.

    Returns:
    - ForecastDistribution: The initialized distribution object
    """
    if distribution_name(distribution) == "distr_trunc_normal":
        return TruncatedNormal(num_features, parameters)
    elif distribution_name(distribution) == "distr_log_normal":
        return LogNormal(num_features, parameters)
    elif distribution_name(distribution) == "distr_gev":
        return GEV(num_features, parameters)
    elif distribution_name(distribution) == "distr_gev2":
        return GEV2(num_features, parameters)
    elif distribution_name(distribution) == "distr_gev3":
        return GEV3(num_features, parameters)
    elif distribution_name(distribution) == "distr_mixture":
        return Mixture(num_features, distribution_1, distribution_2, parameters)
    elif distribution_name(distribution) == "distr_mixture_linear":
        return MixtureLinear(num_features, distribution_1, distribution_2, parameters)      
    else:
        raise ValueError("Unknown distribution")
    
def distribution_name(distribution):
    """
    Function to convert the distribution name to a standard name.

    Args:
    - distribution (str): The name of the distribution

    Returns:
    - str: The standard name of the distribution
    """

    if distribution.lower() in ["distr_trunc_normal", "trunc_normal", "truncated_normal", "truncated normal", "truncnormal", "truncatednormal"]:
        return "distr_trunc_normal"
    elif distribution.lower() in ["distr_log_normal", "log_normal", "lognormal", "log normal"]:
        return "distr_log_normal"
    elif distribution.lower() in ["distr_gev", "gev", "generalized extreme value"]:
        return "distr_gev"
    elif distribution.lower() in ["distr_gev2", "gev2"]:
        return "distr_gev2"
    elif distribution.lower() in ["distr_gev3", "gev3"]:
        return "distr_gev3"
    elif distribution.lower() in ["distr_mixture", "mixture"]:
        return "distr_mixture"
    elif distribution.lower() in ["distr_mixture_linear", "mixture_linear", "mixturelinear"]:
        return "distr_mixture_linear"
    else:
        raise ValueError("Unknown distribution")

    
class Mixture(ForecastDistribution):
    """
    Forecast distribution representing a mixture of two distributions. It contains two distributions and a weight parameter.
    The weight parameter is independent of the input data.

    Attributes:
    - num_features (int): Number of features used in the model
    - distribution_1 (ForecastDistribution): The first distribution in the mixture
    - distribution_2 (ForecastDistribution): The second distribution in the mixture
    - parameters (dict): Dictionary containing the parameters of the distribution
    """
    def __init__(self, num_features, distribution_1, distribution_2, parameters = {}):
        super().__init__(num_features)
        if distribution_name(distribution_1) == distribution_name(distribution_2):
            raise ValueError("The two distributions should be different")
        
        self.distribution_1 = initialize_distribution(distribution_1, num_features, parameters)
        
        self.distribution_2 = initialize_distribution(distribution_2, num_features, parameters)
        
        constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)

        if 'weight' in parameters:
            self.parameter_dict['weight'] = tf.Variable(parameters['weight'], dtype = tf.float32, name="weight", trainable=True, constraint=constraint)
        else:
            self.parameter_dict['weight'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.5, dtype=tf.float32, trainable=True, name='weight', constraint=constraint)

        # This create references to the parameters of distribution_1 and distribution_2 in parameter_dict
        self.parameter_dict.update(self.distribution_1.get_parameter_dict())
        self.parameter_dict.update(self.distribution_2.get_parameter_dict())


    def get_distribution(self, X, variance):
        distribution_1 = self.distribution_1.get_distribution(X, variance)
        distribution_2 = self.distribution_2.get_distribution(X, variance)
        return DistributionMixture(distribution_1, distribution_2, self.parameter_dict['weight']) 
    
    # def get_parameter_dict(self):
    #     """
    #     Returns the parameters of distribution_1, distribution_2 and the weight as tf.Variable in a dictionary
    #     """
    #     parameter_dict = self.distribution_1.get_parameter_dict()
    #     parameter_dict.update(self.distribution_2.get_parameter_dict())
    #     parameter_dict.update(self.parameter_dict)
    #     return parameter_dict
    
    # def get_parameters(self):
    #     """
    #     Returns the parameters of distribution_1, distribution_2 and the weight as np.array in a dictionary
    #     """
    #     parameters = self.distribution_1.get_parameters()
    #     parameters.update(self.distribution_2.get_parameters())
    #     parameters.update(self.parameter_dict)
    #     return parameters
    
    # def set_parameters(self, parameters):
    #     """
    #     Sets the parameters of the distribution to the given values.
    #     """
    #     self.distribution_1.set_parameters(parameters)
    #     self.distribution_2.set_parameters(parameters)
    #     for key, value in parameters.items():
    #         if key in self.parameter_dict:
    #             self.parameter_dict[key].assign(value)
    #             print("Parameter {0} set to {1}".format(key, value))

    def __str__(self):
        info = "Mixture distribution with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        info += "Distribution 1:\n"
        info += str(self.distribution_1)
        info += "Distribution 2:\n"
        info += str(self.distribution_2)
        return info
    
    def name(self):
        return "distr_mixture"

    def get_weight(self):
        return self.parameter_dict['weight'].numpy()

    
class MixtureLinear(ForecastDistribution):
    """
    Forecast distribution representing a mixture of two distributions. It contains two distributions and a weight parameter.
    The weight parameter is dependent on the input data.

    Attributes:
    - num_features (int): Number of features used in the model
    - distribution_1 (ForecastDistribution): The first distribution in the mixture
    - distribution_2 (ForecastDistribution): The second distribution in the mixture
    - parameters (dict): Dictionary containing the parameters of the distribution
    """
    def __init__(self, num_features, distribution_1, distribution_2, parameters = {}):
        super().__init__(num_features)
        if distribution_1 == distribution_2:
            raise ValueError("The two distributions should be different")
        
        self.distribution_1 = initialize_distribution(distribution_1, num_features, parameters)
        
        self.distribution_2 = initialize_distribution(distribution_2, num_features, parameters)
        
        if "weight_a" in parameters and "weight_b" in parameters and "weight_c" in parameters:
            self.parameter_dict['weight_a'] = tf.Variable(parameters['weight_a'], dtype = tf.float32, name="weight_a")
            self.parameter_dict['weight_b'] = tf.Variable(parameters['weight_b'], dtype = tf.float32, name="weight_b")
            self.parameter_dict['weight_c'] = tf.Variable(parameters['weight_c'], dtype = tf.float32, name="weight_c")
            print("Using given weight parameters")
        else:
            self.parameter_dict['weight_a'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="weight_a"))
            self.parameter_dict['weight_b'] = tf.Variable(tf.ones(self.num_features, dtype=tf.float32), name="weight_b")
            self.parameter_dict['weight_c'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="weight_c")
            print("Using default weight parameters")

        # This create references to the parameters of distribution_1 and distribution_2 in parameter_dict
        self.parameter_dict.update(self.distribution_1.get_parameter_dict())
        self.parameter_dict.update(self.distribution_2.get_parameter_dict())

    def get_distribution(self, X, variance):
        weight = tf.math.sigmoid(self.parameter_dict['weight_a'] + tf.tensordot(X, self.parameter_dict['weight_b'], axes=1) + self.parameter_dict['weight_c'] * variance)
        
        mixture_distr = self.DistributionMixture(self.distribution_1(X, variance), self.distribution_2(X, variance), weight)
        return mixture_distr
        
    # def get_parameter_dict(self):
    #     parameter_dict = self.distribution_1.get_parameter_dict()
    #     parameter_dict.update(self.distribution_2.get_parameter_dict())
    #     parameter_dict.update(self.parameter_dict)
    #     return parameter_dict
    
    # def get_parameters(self):
    #     parameters = self.distribution_1.get_parameters()
    #     parameters.update(self.distribution_2.get_parameters())
    #     parameters.update(self.parameter_dict)
    #     return parameters
    
    # def set_parameters(self, parameters):
    #     self.distribution_1.set_parameters(parameters)
    #     self.distribution_2.set_parameters(parameters)
    #     for key, value in parameters.items():
    #         if key in self.parameter_dict:
    #             self.parameter_dict[key].assign(value)
    #             print("Parameter {0} set to {1}".format(key, value))

    def __str__(self):
        info = "Mixture Linear distribution with parameters:\n"
        for key, value in self.parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        info += "Distribution 1:\n"
        info += str(self.distribution_1)
        info += "Distribution 2:\n"
        info += str(self.distribution_2)
        return info
    
    def name(self):
        return "distr_mixture_linear"

    def get_weights(self):
        return self.parameter_dict['weight_a'].numpy(), self.parameter_dict['weight_b'].numpy(), self.parameter_dict['weight_c'].numpy()

        
            
