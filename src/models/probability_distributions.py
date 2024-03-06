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

    def log_prob(self, x):
        return self.weight * self.distribution_1.log_prob(x) + (1 - self.weight) * self.distribution_2.log_prob(x)

    def cdf(self, x):
        return self.weight * self.distribution_1.cdf(x) + (1 - self.weight) * self.distribution_2.cdf(x)

    def sample(self, n):
        return self.weight * self.distribution_1.sample(n) + (1 - self.weight) * self.distribution_2.sample(n)    
    
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
        self.cdf_high = self._gev.cdf(self._high)

    def _log_prob(self, x):
        return self._gev.log_prob(x) - (tf.math.log(self.cdf_high - self.cdf_low))
    
    def _cdf(self, x):
        return (self._gev.cdf(x) - self.cdf_low) / (self.cdf_high - self.cdf_low)
    
    def _sample_n(self, n, seed=None):
        # samples = []
        # while len(samples) < n:
        #     sample = self._gev.sample(n - len(samples), seed)
        #     # remove all the samples that are less than 0
        #     sample = sample[sample > 0]
        #     samples.extend(sample.numpy().tolist())
        # return tf.convert_to_tensor(samples[:n], dtype=tf.float32)
        samples = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        sample_count = 0
        while tf.less(sample_count, n):
            sample = self._gev.sample(n - sample_count, seed)
            mask = tf.greater(sample, 0)
            sample = tf.boolean_mask(sample, mask)
            samples = samples.write(sample_count, sample)
            sample_count += tf.reduce_sum(tf.cast(mask, tf.int32))
        samples = samples.stack()
        return samples[:n]


    
