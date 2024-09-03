import collections
from typing import Dict, List
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from abc import ABC, abstractmethod
from src.linreg_emos.probability_distributions import DistributionMixture
tfpd = tfp.distributions


def initialize_distribution(distribution: str, all_features: List[str], location_features: List[str], scale_features: List[str], parameters, random_init: bool, distribution_1 = None, distribution_2 = None):
    """
    Initializes the given distribution based on the input.

    Args:
        distribution (str): The name of the distribution
        location_features (list): List of feature names to be used for location parameter
        scale_features (list): List of feature names to be used for scale parameter
        parameters (dict): Dictionary containing the parameters of the distribution, which is optional.

    Returns:
        ForecastDistribution: The initialized distribution object
    """
    if distribution_name(distribution) == "distr_trunc_normal":
        return TruncatedNormal(all_features, location_features, scale_features, random_init, parameters)
    elif distribution_name(distribution) == "distr_log_normal":
        return LogNormal(all_features, location_features, scale_features, random_init, parameters)
    elif distribution_name(distribution) == "distr_gev":
        return GEV(all_features, location_features, scale_features, random_init, parameters)
    elif distribution_name(distribution) == "distr_mixture":
        return Mixture(all_features, location_features, scale_features, distribution_1, distribution_2, random_init, parameters)
    elif distribution_name(distribution) == "distr_mixture_linear":
        return MixtureLinear(all_features, location_features, scale_features, distribution_1, distribution_2, random_init, parameters) 
    elif distribution_name(distribution) == "distr_trunc_sqrt":
        return TruncatedNormalSqrt(all_features, location_features, scale_features, random_init, parameters)     
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
    elif distribution.lower() in ["distr_trunc_normal_features", "trunc_normal_features", "truncated_normal_features", "truncated normal features", "truncnormalfeatures"]:
        return "distr_trunc_normal_features"
    elif distribution.lower() in ["distr_log_normal", "log_normal", "lognormal", "log normal"]:
        return "distr_log_normal"
    elif distribution.lower() in ["distr_gev", "gev", "generalized extreme value"]:
        return "distr_gev"
    elif distribution.lower() in ["distr_mixture", "mixture"]:
        return "distr_mixture"
    elif distribution.lower() in ["distr_mixture_linear", "mixture_linear", "mixturelinear"]:
        return "distr_mixture_linear"
    elif distribution.lower() in ["distr_trunc_sqrt"]:
        return "distr_trunc_sqrt"
    else:
        raise ValueError("Unknown distribution")
    

class ForecastDistribution(ABC):
    """
    Virtual base class for forecast distributions.

    This class serves as a template for implementing specific EMOS forecast distribution models.
    Subclasses should override the `get_distribution` method to provide functionality for
    generating distribution objects based on input data and variance.
    """

    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str]):
        """
        Initializes the class and selects which features are used to determine the mean and which to determine the variance.

        Arguments:
            all_features (List[str]): a list of all the feature names. The order of the data should be the same as the order of all_features.
            location_features (List[str]): a list of the feature names that should be used to determine the mean.
            scale_features (List[str]): a list of the feature names that should be used to determine the scale.
        """
        self._parameter_dict = {}
        self.location_features = location_features
        self.scale_features = scale_features
        self.all_features = all_features
        
        # We determine the indices of the loc/scale features in all_features, that way we can select the correct features when data is provided.
        self.location_features_indices = [all_features.index(feature) for feature in location_features]
        self.scale_features_indices = [all_features.index(feature) for feature in scale_features]
        
        if not all(feature in all_features for feature in location_features):
            raise ValueError("Location features are not in the list of all features")
        if not all(feature in all_features for feature in scale_features):
            raise ValueError("Scale features are not in the list of all features")
        
        self.location_features_indices = tf.constant(self.location_features_indices, dtype=tf.int32)
        self.scale_features_indices = tf.constant(self.scale_features_indices, dtype=tf.int32)


    @abstractmethod
    def get_distribution(self, X: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Returns a distribution object given the input and variance.

        Args:
            X (tf.Tensor): Input data

        Returns:
            tfp.distributions.Distribution/Mixture: The forecast distribution
        """
        pass
    
    @property
    def parameter_dict(self) -> dict:
        """
        Returns the parameters of the distribution as tf.Variables in a dictionary.
        In case just np.array are needed, the parameters property can be used.
        """
        return self._parameter_dict
    
    @property
    def parameters(self) -> dict:
        """
        Returns the parameters of the distribution as np.arrays in a dictionary.
        """
        parameters = {}
        for key, value in self._parameter_dict.items():
            parameters[key] = value.numpy()
        return parameters
        
    @parameters.setter
    def parameters(self, parameters: dict) -> None:
        """
        Sets the parameters of the distribution to the given values.

        Args:
            parameters (dict): Dictionary with parameter names as keys and values as values. 

        Returns:
            None
        """
        for key, value in parameters.items():
            if key in self._parameter_dict:
                self._parameter_dict[key].assign(value)
                print("Parameter {0} set to {1}".format(key, value))

    
    def contains_gev(self) -> bool:
        """
        Used to check whether the distribution contains the GEV distribution.
        """
        return False
    
    def get_gev_shape(self) -> float:
        """
        Returns the shape of the GEV distribution in case this is part of the parametric distribution.
        """
        return None
    
    def comp_cdf(self, X: tf.Tensor, values: np.ndarray) -> np.ndarray:
        """
        Computes the cumulative distribution function of the forecast distribution for a range of values.

        Args:
            X (tf.Tensor): Input data
            values (np.ndarray): Values for which to compute the cdf

        Returns:
            np.ndarray: The cdf values for each value in the input array
        """
        if not isinstance(values, collections.abc.Iterable):
            values = [values]
            
        output = np.zeros((len(values), X.shape[0]))
        distr = self.get_distribution(X)
        for i, value in enumerate(values):
            output[i] = distr.cdf(value).numpy()

        return output
    
    def print_feature_info(self) -> str:
        """
        Prints the feature information of the distribution.
        """
        print("Features for location parameter:")
        for key, value in self.location_features.items():
            print("{0}: {1}".format(key, value))
        print("\nFeatures for scale parameter:")
        for key, value in self.scale_features.items():
            print("{0}: {1}".format(key, value))

    @abstractmethod
    def distribution_name(self) -> str:
        """
        Prints the name of the distribution.
        """
        pass

    @abstractmethod
    def folder_name(self) -> str:
        """
        Prints a shorter name of the distribution, useful in case we want to save it in a specific folder.
        """
        pass


def random_initialization(x: int, setting: str) -> np.ndarray:
    """
    Function that samples random numbers from a specified distribution.

    Arguments:
        x (int): number of random elements to be selected.
        setting (str): distribution to be sampled from

    Returns:
        np.ndarray containing the random samples.
    """
    if setting == 'standard_uniform':
        return np.random.uniform(size = x)
    elif setting == 'standard_normal':
        return np.random.normal(size = x)
    else:
        raise ValueError("This setting is not supported!")


    
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
    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str], random_init: bool, parameters: Dict[str, float] = {}):
        """
        Constructor for the TruncatedNormal class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
            all_features (List[str]): list of all the feature names.
            location_features (List[str]): list of the features used for the mean.
            scale_features (List[str]): list of features used to determine the scale.
            random_init (bool): boolean to determine if random initialization should be used.
            parameters (Dict[str, float], optional): a dictionary containing the values for the parameters to use for initialization.

        Returns:
            None
        """
        super().__init__(all_features, location_features, scale_features)

        # Initializing the parameters.
        if "a_tn" in parameters and "b_tn" in parameters and "c_tn" in parameters and "d_tn" in parameters:
            # if all the parameters are in the parameters dict, we initialize them to these values.
            self._parameter_dict["a_tn"] = tf.Variable(parameters["a_tn"], dtype = tf.float32, name="a_tn")
            self._parameter_dict["b_tn"] = tf.Variable(parameters["b_tn"], dtype = tf.float32, name="b_tn")
            self._parameter_dict["c_tn"] = tf.Variable(parameters["c_tn"], dtype = tf.float32, name="c_tn")
            self._parameter_dict["d_tn"] = tf.Variable(parameters["d_tn"], dtype = tf.float32, name="d_tn")
            print("Using given parameters for Truncated Normal distribution")
        elif random_init:
            # initialize using random_initialization
            self._parameter_dict['a_tn'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="a_tn")
            self._parameter_dict['b_tn'] = tf.Variable(random_initialization(len(self.location_features_indices), 'standard_normal'), dtype = tf.float32, name="b_tn")
            self._parameter_dict['c_tn'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="c_tn")
            self._parameter_dict['d_tn'] = tf.Variable(random_initialization(len(self.scale_features_indices), 'standard_normal'), dtype = tf.float32, name="d_tn")
            print("Using random initialization for Truncated Normal distribution")
        else:
            # Fixed initialization.
            self._parameter_dict['a_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_tn"))
            self._parameter_dict['b_tn'] = tf.Variable(tf.ones(len(self.location_features_indices), dtype=tf.float32), name="b_tn")
            self._parameter_dict['c_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_tn")
            self._parameter_dict['d_tn'] = tf.Variable(tf.ones(len(self.scale_features_indices), dtype=tf.float32), name="d_tn")
            # print("Using default parameters for truncated normal distribution")

    def get_distribution(self, X: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Based on data computes the forecast distribution.

        Arguments:
            X (tf.Tensor): tensor containing the features.

        Returns:
            A truncated normal distribution with the selected parameters.
        """
        mu = self._parameter_dict['a_tn'] + tf.tensordot(tf.gather(X, self.location_features_indices, axis=1), self._parameter_dict['b_tn'], axes=1)
        sigma_squared = self._parameter_dict['c_tn'] + tf.tensordot(tf.gather(X, self.scale_features_indices, axis=1), self._parameter_dict['d_tn'], axes=1)
        sigma = tf.sqrt(tf.math.softplus(sigma_squared))
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def __str__(self) -> str:
        info = "Truncated Normal distribution with parameters:\n"
        for key, value in self._parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self) -> str:
        return "distr_trunc_normal"
    
    def folder_name(self) -> str:
        return "trunc_normal"
    
    def distribution_name(self) -> str:
        return "tn"
    
class TruncatedNormalSqrt(ForecastDistribution):
    """
    Forecast distribution representing a truncated normal EMOS distribution.

    This class implements a truncated normal distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data.
    It assumes linear relationship between the distribution parameters and the input data.
    """
    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str], random_init: bool, parameters: Dict[str, float] = {}):
        """
        Constructor for the TruncatedNormalSqrt class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
            all_features (List[str]): list of all the feature names.
            location_features (List[str]): list of the features used for the mean.
            scale_features (List[str]): list of features used to determine the scale.
            random_init (bool): boolean to determine if random initialization should be used.
            parameters (Dict[str, float], optional): a dictionary containing the values for the parameters to use for initialization.

        Returns:
            None
        """
        super().__init__(all_features, location_features, scale_features)

        if "a_tn" in parameters and "b_tn" in parameters and "c_tn" in parameters and "d_tn" in parameters:
            # initializing using the chosen parameters
            self._parameter_dict["a_tn"] = tf.Variable(parameters["a_tn"], dtype = tf.float32, name="a_tn")
            self._parameter_dict["b_tn"] = tf.Variable(parameters["b_tn"], dtype = tf.float32, name="b_tn")
            self._parameter_dict["c_tn"] = tf.Variable(parameters["c_tn"], dtype = tf.float32, name="c_tn")
            self._parameter_dict["d_tn"] = tf.Variable(parameters["d_tn"], dtype = tf.float32, name="d_tn")
            print("Using given parameters for Truncated Normal distribution")
        elif random_init:
            # initialize using random_initialization
            self._parameter_dict['a_tn'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="a_tn")
            self._parameter_dict['b_tn'] = tf.Variable(random_initialization(len(self.location_features_indices), 'standard_normal'), dtype = tf.float32, name="b_tn")
            self._parameter_dict['c_tn'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="c_tn")
            self._parameter_dict['d_tn'] = tf.Variable(random_initialization(len(self.scale_features_indices), 'standard_normal'), dtype = tf.float32, name="d_tn")
            print("Using random initialization for Truncated Normal distribution")
        else:
            # fixed initializition
            self._parameter_dict['a_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_tn"))
            self._parameter_dict['b_tn'] = tf.Variable(tf.ones(len(self.location_features_indices), dtype=tf.float32), name="b_tn")
            self._parameter_dict['c_tn'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_tn")
            self._parameter_dict['d_tn'] = tf.Variable(tf.ones(len(self.scale_features_indices), dtype=tf.float32), name="d_tn")
            print("Using default parameters for truncated normal distribution")

    def get_distribution(self, X: tf.Tensor):
        mu = self._parameter_dict['a_tn'] + tf.tensordot(tf.gather(X, self.location_features_indices, axis=1), self._parameter_dict['b_tn'], axes=1)
        sigma = tf.math.softplus(self._parameter_dict['c_tn'] + tf.tensordot(tf.gather(X, self.scale_features_indices, axis=1), self._parameter_dict['d_tn'], axes=1))
        return tfpd.TruncatedNormal(mu, sigma, 0, 1000)
    
    def __str__(self):
        info = "Truncated Normal distribution with parameters:\n"
        for key, value in self._parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self):
        return "distr_trunc_normal"
    
    def folder_name(self):
        return "trunc_normal"
    
    def distribution_name(self):
        return "tn"
    

    
class LogNormal(ForecastDistribution):
    """
    Forecast distribution representing a lognormal EMOS distribution.

    This class implements a truncated normal distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data.
    It assumes linear relationship between the distribution parameters and the input data.
    """
    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str], random_init: bool, parameters: Dict[str, float] = {}):
        """
        Constructor for the LogNormal class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
            all_features (List[str]): list of all the feature names.
            location_features (List[str]): list of the features used for the mean.
            scale_features (List[str]): list of features used to determine the scale.
            random_init (bool): boolean to determine if random initialization should be used.
            parameters (Dict[str, float], optional): a dictionary containing the values for the parameters to use for initialization.

        Returns:
            None
        """
        super().__init__(all_features, location_features, scale_features)

        if "a_ln" in parameters and "b_ln" in parameters and "c_ln" in parameters:
            self._parameter_dict["a_ln"] = tf.Variable(parameters["a_ln"], dtype = tf.float32, name="a_ln")
            self._parameter_dict["b_ln"] = tf.Variable(parameters["b_ln"], dtype = tf.float32, name="b_ln")
            self._parameter_dict["c_ln"] = tf.Variable(parameters["c_ln"], dtype = tf.float32, name="c_ln")
            self._parameter_dict["d_ln"] = tf.Variable(parameters["d_ln"], dtype = tf.float32, name="d_ln")
            print("Using given parameters for Log Normal distribution")
        elif random_init:
            # initialize using random_initialization
            self._parameter_dict['a_ln'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="a_ln")
            self._parameter_dict['b_ln'] = tf.Variable(random_initialization(len(self.location_features_indices), 'standard_normal'), dtype = tf.float32, name="b_ln")
            self._parameter_dict['c_ln'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="c_ln")
            self._parameter_dict['d_ln'] = tf.Variable(random_initialization(len(self.scale_features_indices), 'standard_normal'), dtype = tf.float32, name="d_ln")
            print("Using random initialization for Log Normal distribution")
        else:
            self._parameter_dict['a_ln'] = tf.Variable(tf.ones(1, dtype=tf.float32, name="a_ln"))
            self._parameter_dict['b_ln'] = tf.Variable(tf.ones(len(self.location_features_indices), dtype=tf.float32), name="b_ln")
            self._parameter_dict['c_ln'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_ln")
            self._parameter_dict['d_ln'] = tf.Variable(tf.ones(len(self.scale_features_indices), dtype=tf.float32), name="d_ln")
            # print("Using default parameters for Log Normal distribution")

    def get_distribution(self, X: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Computes the parameters of the log-normal distribution based on the input data.

        Argument:
            X (tf.Tensor): the input data.

        Returns:
            The log-normal distribution with the selected parameters.
        """
        m = self._parameter_dict['a_ln'] + tf.tensordot(tf.gather(X, self.location_features_indices, axis=1), self._parameter_dict['b_ln'], axes=1)
        v = self._parameter_dict['c_ln'] + tf.tensordot(tf.gather(X, self.scale_features_indices, axis=1), self._parameter_dict['d_ln'], axes=1)

        # We transform m and v to ensure the resulting distribution has the correct mean and variance.
        v = tf.math.softplus(v)
        mean = tf.math.log(m ** 2) - 0.5 * tf.math.log(v + m ** 2)

        # We add a small value to avoid numerical instability in the square root.
        sigma = tf.sqrt(tf.math.log(1 + v / m ** 2) + 1e-6)
        return tfpd.LogNormal(mean, sigma)
    
    def __str__(self) -> str:
        info = "Log Normal distribution with parameters:\n"
        for key, value in self._parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self) -> str:
        return "distr_log_normal"
    
    def folder_name(self) -> str:
        return "log_normal"
    
    def distribution_name(self) -> str:
        return "ln"
    

class GEV(ForecastDistribution):
    """
    Forecast distribution representing a truncated normal EMOS distribution.

    This class implements a generalized extreme value distribution model for forecasting.
    It inherits from the ForecastDistribution base class and provides functionality
    for generating truncated normal distribution objects based on input data and variance.
    It assumes linear relationship between the distribution parameters and the input data.
    This class does not use the variance in the distribution parameters.
    """
    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str], random_init: bool, parameters: Dict[str, float] = {}):
        """
        Constructor for the GEV class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
            all_features (List[str]): list of all the feature names.
            location_features (List[str]): list of the features used for the mean.
            scale_features (List[str]): list of features used to determine the scale.
            random_init (bool): boolean to determine if random initialization should be used.
            parameters (Dict[str, float], optional): a dictionary containing the values for the parameters to use for initialization.

        Returns:
            None
        """
        super().__init__(all_features, location_features, scale_features)
        if "a_gev" in parameters and "b_gev" in parameters and "c_gev" in parameters and "d_gev" in parameters and "e_gev" in parameters:
            self._parameter_dict["a_gev"] = tf.Variable(parameters["a_gev"], dtype = tf.float32, name="a_gev")
            self._parameter_dict["b_gev"] = tf.Variable(parameters["b_gev"], dtype = tf.float32, name="b_gev")
            self._parameter_dict["c_gev"] = tf.Variable(parameters["c_gev"], dtype = tf.float32, name="c_gev")
            self._parameter_dict["d_gev"] = tf.Variable(parameters["d_gev"], dtype = tf.float32, name="d_gev")
            self._parameter_dict["e_gev"] = tf.Variable(parameters["e_gev"], dtype = tf.float32, name="e_gev")
            print("Using given parameters for Generalized Extreme Value distribution")
        elif random_init:
            # initialize using random_initialization
            self._parameter_dict['a_gev'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="a_gev")
            self._parameter_dict['b_gev'] = tf.Variable(random_initialization(len(self.location_features_indices), 'standard_normal'), dtype = tf.float32, name="b_gev")
            self._parameter_dict['c_gev'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="c_gev")
            self._parameter_dict['d_gev'] = tf.Variable(random_initialization(len(self.scale_features_indices), 'standard_normal'), dtype = tf.float32, name="d_gev")
            self._parameter_dict['e_gev'] = tf.Variable(random_initialization(1, 'standard_normal'), dtype = tf.float32, name="e_gev")
            print("Using random initialization for Generalized Extreme Value distribution")
        else:
            self._parameter_dict['a_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="a_gev")
            self._parameter_dict['b_gev'] = tf.Variable(tf.ones(len(self.location_features_indices), dtype=tf.float32), name="b_gev")
            self._parameter_dict['c_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32), name="c_gev")
            self._parameter_dict['d_gev'] = tf.Variable(tf.ones(len(self.scale_features_indices), dtype=tf.float32), name="d_gev")
            self._parameter_dict['e_gev'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.3, name="e_gev")
            # print("Using default parameters for Generalized Extreme Value distribution")

    def get_distribution(self, X: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Returns the GEV distribution with the selected parameters based on the input data.

        Arguments:
            X (tf.Tensor): the input data.

        Returns: 
            The GEV distribution.
        """
        location = self._parameter_dict['a_gev'] + tf.tensordot(tf.gather(X, self.location_features_indices, axis=1), self._parameter_dict['b_gev'], axes=1)

        scale = self._parameter_dict['c_gev'] + tf.tensordot(tf.gather(X, self.scale_features_indices, axis=1), self._parameter_dict['d_gev'], axes=1)
        scale = tf.math.softplus(scale)

        shape = self._parameter_dict['e_gev']

        return tfpd.GeneralizedExtremeValue(location, scale, shape)

    def __str__(self) -> str:
        info = "Generalized Extreme Value distribution with parameters:\n"
        for key, value in self._parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        return info
    
    def name(self) -> str:
        return "distr_gev"
    
    def contains_gev(self) -> bool:
        return True
    
    def get_gev_shape(self) -> np.float_:
        """
        Returns the shape of the GEV distribution.
        """
        return self._parameter_dict['e_gev'].numpy()
    
    def comp_cdf(self, X: tf.Tensor, values: np.ndarray) -> np.ndarray:
        """
        In case we want to compute the cdf for a set values, the GEV gives NaN outside of its domain.
        With this method we set these values to either 0 or 1, depending on the shape of the distribution.

        Arguments:
            X (tf.Tensor): tensor containing the features.
            values (np.ndarray): array containing the values on which we want to compute the cdf.

        Returns:
            The cdf values.
        """
        if not isinstance(values, collections.abc.Iterable):
            values = [values]
            
        output = np.zeros((len(values), X.shape[0]))
        distr = self.get_distribution(X)
        shape = distr.concentration.numpy()

        if np.isscalar(shape) or shape.size == 1:
            shape = np.full(X.shape[0], shape)  # Convert scalar to array

        for i, value in enumerate(values):
            cdf_value = distr.cdf(value).numpy()
            nan_indices = np.isnan(cdf_value)  # Find NaN indices
            shape_less_than_zero = shape < 0  # Find shape < 0 indices
            output[i] = cdf_value

            # Update NaN values based on the condition
            output[i, nan_indices & shape_less_than_zero] = 1
            output[i, nan_indices & ~shape_less_than_zero] = 0

        return output
    
    def folder_name(self) -> str:
        return "gev"
    
    def distribution_name(self) -> str:
        return "gev"
    
    
class Mixture(ForecastDistribution):
    """
    Forecast distribution representing a mixture of two distributions. It contains two distributions and a weight parameter.
    The weight parameter is independent of the input data.
    """
    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str], distribution_1, distribution_2, random_init, parameters = {}):
        """
        Constructor for the Mixture class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
            all_features (List[str]): list of all the feature names.
            location_features (List[str]): list of the features used for the mean.
            scale_features (List[str]): list of features used to determine the scale.
            distribution_1 (str): name of the first distribution.
            distribution_2 (str): name of the second distribution.
            random_init (bool): boolean to determine if random initialization should be used.
            parameters (Dict[str, float], optional): a dictionary containing the values for the parameters to use for initialization.

        Returns:
            None
        """
        super().__init__(all_features, location_features, scale_features)
        
        # initialize the parameters of the underlying distributions
        self.distribution_1 = initialize_distribution(distribution_1, all_features, location_features, scale_features, parameters, random_init)
        
        self.distribution_2 = initialize_distribution(distribution_2, all_features, location_features, scale_features, parameters, random_init)
        
        constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)

        if 'weight' in parameters:
            self._parameter_dict['weight'] = tf.Variable(initial_value=parameters['weight'], dtype = tf.float32, name="weight", trainable=True, constraint=constraint)
            print("Using given weight parameter for Mixture distribution")
        else:
            self._parameter_dict['weight'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 0.5, dtype=tf.float32, trainable=True, name='weight', constraint=constraint)
            print("Using default weight parameter for Mixture distribution")

        # This create references to the parameters of distribution_1 and distribution_2 in parameter_dict
        self._parameter_dict.update(self.distribution_1.parameter_dict)
        self._parameter_dict.update(self.distribution_2.parameter_dict)


    def get_distribution(self, X: tf.Tensor) -> tfp.distributions.Distribution:
        distribution_1 = self.distribution_1.get_distribution(X)
        distribution_2 = self.distribution_2.get_distribution(X)
        return DistributionMixture(distribution_1, distribution_2, self._parameter_dict['weight']) 

    def __str__(self):
        info = "Mixture distribution with parameters:\n"
        for key, value in self._parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        info += "Distribution 1:\n"
        info += str(self.distribution_1)
        info += "Distribution 2:\n"
        info += str(self.distribution_2)
        return info
    
    def name(self) -> str:
        return "distr_mixture"

    def get_weight(self) -> np.float_:
        """
        Returns the weight parameter.
        """
        return self._parameter_dict['weight'].numpy()
    
    def contains_gev(self) -> bool:
        return self.distribution_1.contains_gev() or self.distribution_2.contains_gev()
    
    def get_gev_shape(self):
        if self.distribution_1.contains_gev():
            return self.distribution_1.get_gev_shape()
        elif self.distribution_2.contains_gev():
            return self.distribution_2.get_gev_shape()
        else:
            return None
        
    def calc_weights(self, X: tf.Tensor) -> np.ndarray:
        """
        Returns the weight for the input data. This is constant in case of a mixture distribution.

        Arguments:
            X (tf.Tensor): data for which we want to compute the weight

        Returns:
            np.ndarray containing the weight
        """
        return self._parameter_dict['weight']
        
    def comp_cdf(self, X: tf.Tensor, values: np.ndarray) -> np.ndarray:
        """
        Computes the CDF values for the data at the specified values.

        Arguments:
            X (tf.Tensor): the input data containing the samples.
            values (np.ndarray): the values at which the CDF values are computed.

        Returns:
            np.ndarray containing the CDF values.
        """
        cdf_1 = self.distribution_1.comp_cdf(X, values)
        cdf_2 = self.distribution_2.comp_cdf(X, values)
        return self._parameter_dict['weight'].numpy() * cdf_1 + (1 - self._parameter_dict['weight'].numpy()) * cdf_2
    
    def folder_name(self) -> str:
        return "mixture"
    
    def distribution_name(self) -> str:
        return "m" + "_" + self.distribution_1.distribution_name() + "_" + self.distribution_2.distribution_name()
    


    
class MixtureLinear(ForecastDistribution):
    """
    Forecast distribution representing a mixture of two distributions. It contains two distributions and a weight parameter.
    The weight parameter is dependent on the input data.
    """
    def __init__(self, all_features: List[str], location_features: List[str], scale_features: List[str], distribution_1: str, distribution_2: str, random_init: bool, parameters = {}):
        """
        Constructor for the MixtureLinear class. Initializes the parameters of the distribution.
        In case parameters is provided, it sets the parameters to the given values. Otherwise, it
        initializes the parameters to default values.

        Args:
            all_features (List[str]): list of all the feature names.
            location_features (List[str]): list of the features used for the mean.
            scale_features (List[str]): list of features used to determine the scale.
            distribution_1 (str): name of the first distribution.
            distribution_2 (str): name of the second distribution
            random_init (bool): boolean to determine if random initialization should be used.
            parameters (Dict[str, float], optional): a dictionary containing the values for the parameters to use for initialization.

        Returns:
            None
        """
        super().__init__(all_features, location_features, scale_features)
        
        # initialize the parameters of the underlying distribution
        self.distribution_1 = initialize_distribution(distribution_1, all_features, location_features, scale_features, parameters, random_init)
        
        self.distribution_2 = initialize_distribution(distribution_2, all_features, location_features, scale_features, parameters, random_init)

        constraint_a = tf.keras.constraints.MinMaxNorm(min_value=4, max_value=12)
        constraint_b = tf.keras.constraints.MinMaxNorm(min_value=-6, max_value=-0.6)
        
        if "weight_a" in parameters and "weight_b" in parameters: # and "weight_c" in parameters:
            self._parameter_dict['weight_a'] = tf.Variable(parameters['weight_a'], dtype = tf.float32, name="weight_a", constraint=constraint_a)
            self._parameter_dict['weight_b'] = tf.Variable(parameters['weight_b'], dtype = tf.float32, name="weight_b", constraint=constraint_b)
            print("Using given weight parameters for weights in Mixture Linear distribution")
        else:
            self._parameter_dict['weight_a'] = tf.Variable(tf.ones(1, dtype=tf.float32) * 5, name="weight_a", trainable=True, constraint=constraint_a)
            self._parameter_dict['weight_b'] = tf.Variable(tf.ones(1, dtype=tf.float32) * -1, name="weight_b", trainable=True, constraint=constraint_b)
            print("Using default weight parameters for weights in Mixture Linear distribution")

        # This create references to the parameters of distribution_1 and distribution_2 in parameter_dict
        self._parameter_dict.update(self.distribution_1.parameter_dict)
        self._parameter_dict.update(self.distribution_2.parameter_dict)

    def get_distribution(self, X: tf.Tensor) -> DistributionMixture:
        """
        We define the weight as linear function of the first feature of the input data (which is predicted wind speed).
        We then calculate the weight using the sigmoid function to ensure that it is between 0 and 1.

        Inputs:
            X (tf.Tensor): Input data

        Returns:
            DistributionMixture: The forecast distribution
        """
        weight = tf.math.sigmoid(self._parameter_dict['weight_a'] + tf.multiply(X[:,0], self._parameter_dict['weight_b']))
        distribution_1 = self.distribution_1.get_distribution(X)
        distribution_2 = self.distribution_2.get_distribution(X)

        return DistributionMixture(distribution_1, distribution_2, weight)
        
    
    def __str__(self) -> str:
        info = "Mixture Linear distribution with parameters:\n"
        for key, value in self._parameter_dict.items():
            info += "{0}: {1}\n".format(key, value)
        info += "Distribution 1:\n"
        info += str(self.distribution_1)
        info += "Distribution 2:\n"
        info += str(self.distribution_2)
        return info
    
    def name(self) -> str:
        return "distr_mixture_linear"

    def get_weights(self):
        """
        Returns the parameters for determining the weight.
        """
        return self._parameter_dict['weight_a'].numpy(), self._parameter_dict['weight_b'].numpy()
    
    def contains_gev(self) -> bool:
        """
        Checks whether the distribution contains the GEV distribution.
        """
        return self.distribution_1.contains_gev() or self.distribution_2.contains_gev()
    
    def get_gev_shape(self):
        """
        Returns the shape of the GEV distribution in case this is part of the distribution.
        """
        if self.distribution_1.contains_gev():
            return self.distribution_1.get_gev_shape()
        elif self.distribution_2.contains_gev():
            return self.distribution_2.get_gev_shape()
        else:
            return None
        
    def calc_weights(self, X: tf.Tensor) -> np.ndarray:
        """
        Calculates the weight for the samples from data.

        Arguments:
            X (tf.Tensor): the input data.

        Returns:
            np.ndarray with the weights for the samples.
        """
        return tf.math.sigmoid(self._parameter_dict['weight_a'] + tf.multiply(X[:,0], self._parameter_dict['weight_b']))
        
    def comp_cdf(self, X: tf.Tensor, values: np.ndarray) -> np.ndarray:
        """
        Computes the CDF values for the data at the specified values.

        Arguments:
            X (tf.Tensor): the input data containing the samples.
            values (np.ndarray): the values at which the CDF values are computed.

        Returns:
            np.ndarray containing the CDF values.
        """
        weight = self.calc_weights(X)
        cdf_1 = self.distribution_1.comp_cdf(X, values)
        cdf_2 = self.distribution_2.comp_cdf(X, values)
        return weight.numpy() * cdf_1 + (1 - weight.numpy()) * cdf_2
    
    def folder_name(self) -> str:
        return "mixture_linear"
    
    def distribution_name(self) -> str:
        return "ml" + "_" + self.distribution_1.distribution_name() + "_" + self.distribution_2.distribution_name()


        
            
