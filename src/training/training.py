import pickle as pkl

import numpy as np

from src.models.emos import LinearEMOS
from src.models.forecast_distributions import Mixture, TruncatedGEV, TruncatedNormal, LogNormal, GEV, Frechet, DistributionMixture, MixtureLinear, distribution_name



def load_model(path):
    with open(path, 'rb') as f:
        model_dict = pkl.load(f)
    
    model = LinearEMOS(model_dict)
    return model

    

    

    