import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Method for training EMOS model
def train_emos(ensemble, actual, emos, scoring_rule):
    pass
    def objective_function(weights):
        
        ### We should pick a random subset of the data and compute the score of emos over this set.
        subset = ...
        
        emos.SetWeights(weights)
        return scoring_rule.score(emos.cdfFunction(ensemble), actual)
    
    # minimize the objective function

    initial_weights = ...

    result = minimize(objective_function, initial_weights)

    return result


class EMOS():
    def __init__(self, Weights):
        self.Weights = Weights

    def SetWeights(self, Weights):
        self.Weights = Weights
    
    def GetWeights(self):
        return self.Weights
    
    def SetParameters(self, ensemble):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def cdf(x, ensemble):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def pdf(x):
        raise NotImplementedError("Subclass must implement abstract method")
    
    
# We assume the weights are a tuple, where the first element are the weights to determine the mean,
# and the second element are the weights to determine the variance.
# Ensemble is a pandas dataframe, where the columns are the ensemble members
class TruncatedNormal(EMOS):
    def __init__(self, weights):
        super().__init__(weights)
        self.ensemblesize = len(weights[0]) - 1
        self.mean = 0
        self.std = 0

    def SetParameters(self, ensemble):
         # take the inner product of the weights and the ensemble and as bias to compute mean
        self.mean = ensemble.dot(self.Weights[0][1:self.ensemblesize]) + self.Weights[0][0]

        # take estimate of variance
        self.std = np.sqrt(np.var(ensemble) * self.Weights[1][0] + self.Weights[1][1])
    
    def cdfFunction(self, ensemble):
        self.SetParameters(ensemble)
        return lambda x: norm.cdf((x - self.mean) / self.std) / norm.cdf(self.mean / self.std) if x > 0 else 0
    
    def cdf(self, x, ensemble):
        return self.cdfFunction(ensemble)(x) 
    
    def pdfFunction(self, ensemble):
        self.SetParameters(ensemble)
        return lambda x: norm.pdf((x - self.mean) / self.std) / (self.std * norm.cdf(self.mean / self.std)) if x > 0 else 0

    def pdf(self, x, ensemble):
        return self.pdfFunction(ensemble)(x)
    