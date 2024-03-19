import tensorflow as tf
import tensorflow_probability as tfp

class DistributionMixture(tf.Module):
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

        # dummy = tf.reduce_sum(self.weight)
        # ### for this code to work, the self.weight parameter should not be an array, but a single value
        # probs = tf.stack([dummy, 1 - dummy])
        # probs_broadcast = tf.broadcast_to(probs, [self.distribution_1.batch_shape[0], 2])

        # cat = tfp.distributions.Categorical(probs=probs_broadcast)
        # cat2 = tfp.distributions.Categorical(probs=[dummy, 1 - dummy])
        # self.mixture = tfp.distributions.Mixture(
        #     cat=cat2,
        #     components=[distribution_1, distribution_2]
        # )
        # x = 1


    def log_prob(self, x):
        return self.weight * self.distribution_1.log_prob(x) + (1 - self.weight) * self.distribution_2.log_prob(x)

    def cdf(self, x):
        return self.weight * self.distribution_1.cdf(x) + (1 - self.weight) * self.distribution_2.cdf(x)

    def sample(self, n):
        # samples_1 = self.distribution_1.sample(n)
        # samples_2 = self.distribution_2.sample(n)
        # # Create a Categorical distribution with mixing proportions given by self.weight and 1 - self.weight
        # cat = tfp.distributions.Categorical(probs=[self.weight, 1 - self.weight])
        # # Sample from the Categorical distribution
        # samples_cat = cat.sample([n, self.distribution_1.batch_shape[0]])
        # # Use the samples from the Categorical distribution to create a mixture of samples_1 and samples_2
        # return tf.where(samples_cat == 0, samples_1, samples_2)

        # return self.mixture.sample(n)

        samples_1 = self.distribution_1.sample(n)
        samples_2 = self.distribution_2.sample(n)

        uniform_samples = tf.random.uniform([n, self.distribution_1.batch_shape[0]])

        # Create a soft mask using a sigmoid function
        mask = tf.sigmoid((self.weight - uniform_samples) * 10000)

        return mask * samples_1 + (1 - mask) * samples_2

        
        # return self.mixture.sample(n)
        # return self.weight * self.distribution_1.sample(n) + (1 - self.weight) * self.distribution_2.sample(n)    
    
    def mean(self):
        return self.weight * self.distribution_1.mean() + (1 - self.weight) * self.distribution_2.mean()
    
    


class TruncGEV(tfp.distributions.Distribution):
    def __init__(self, loc, scale, shape):
        super(TruncGEV, self).__init__(
            dtype=tf.float32,  # The type of the event samples.
            reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,  # Indicates that samples can be reparameterized.
            validate_args=False,  # When True distribution parameters are checked for validity despite possibly degrading runtime performance
            allow_nan_stats=True  # When True, statistics (e.g., mean, mode, variance) use the value NaN to indicate the result is undefined.
        )
        
        self._loc = loc
        self._scale = scale
        self._shape = shape
        self._low = tf.zeros_like(loc, dtype=tf.float32)
        self._high = tf.ones_like(loc, dtype=tf.float32) * 1000
        self._gev = tfp.distributions.GeneralizedExtremeValue(loc, scale, shape)

        self.cdf_low = self._gev.cdf(self._low)
        # self.cdf_high = self._gev.cdf(self._high)

    def _log_prob(self, x):
        return self._gev.log_prob(x) - (tf.math.log(1 - self.cdf_low))
    
    def _prob(self, x):
        return self._gev.prob(x) / (1 - self.cdf_low)

    def _cdf(self, x):
        return (self._gev.cdf(x) - self.cdf_low) / (1 - self.cdf_low)
    
    def _sample_n(self, n, seed):
        cdf_0 = self._gev.cdf(self._low)

        # check if cdf_0 contains nan. If it contains nan, replace it with 0 in case self._shape > 0 and with 1 in case self._shape < 0
        cdf_0 = tf.where(tf.math.is_nan(cdf_0), tf.where(self._shape > 0, 0.0, 1.0), cdf_0)

        # generate uniform randomnumbers between cdf_0 and 1
        u = tf.random.uniform([n, self._gev.batch_shape[0]], minval=cdf_0, maxval=1, seed=seed)
        # inverse cdf
        return self._gev.quantile(u)

    def _event_shape(self):
        return tf.TensorShape([])
    
    def _batch_shape(self):
        return tf.broadcast_static_shape(self._loc.shape, self._scale.shape)

    
