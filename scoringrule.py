import numpy as heaviside
from scipy.integrate import quad

# Abstract class for scoring rules
class ScoringRules:
    def score(self, CDF, actual) -> float:
        raise NotImplementedError("Subclass must implement abstract method")
    
def Indicator(x1, x2) -> float:
    return 1 if x1 >= x2 else 0

class BrierScore(ScoringRules):
    def __init__(self, benchmark: float):
        self.benchmark = benchmark

    def score(self, CDF, actual: float) -> float:
        return ((1 - CDF(actual)) - Indicator(actual, self.benchmark)) ** 2
    
    

class CRPS(ScoringRules):
    def score(self, CDF, actual) -> float:
        integrand = lambda x : (CDF(x) - Indicator(x, actual)) ** 2

        assert integrand(-10000) <= 1e-10, "CRPS integrand is not 0 at -10000"
        assert integrand(10000) <= 1e-10, "CRPS integrand is not 0 at 10000"

        return quad(integrand, -float('inf'), float('inf'))[0]

    

class WeightedScoringRule(ScoringRules):
    def __init__(self, weightfunction):
        self.weightfunction = weightfunction

class ThresholdWeightedCRPS(WeightedScoringRule):
    def __init__(self, weightfunction: float):
        super().__init__(weightfunction)

    def score(self, CDF, actual) -> float:
        integrand = lambda x: self.weightfunction(x) * (CDF(x) - Indicator(x, actual)) ** 2
        
        return quad(integrand, -float('inf'), float('inf'))[0]