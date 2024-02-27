import numpy as np

def get_gev_initial_params():
    output = {
        'a_gev' : np.array([0.7259205]),
        'b_gev' : np.array([ 0.95463854, -0.15831928, -0.20902145, -0.01082994,  0.11957027]),
        'c_gev' : np.array([-0.08465461]),
        'd_gev' : np.array([-0.01219114,  0.02037963,  0.05282509,  0.00752834,  0.00600949]),
        'e_gev' : np.array([0.6217914])
    }
    return output

def get_trunc_normal_initial_params():
    output = {
        'a_tn' : np.array([0.4743731]),
        'b_tn' : np.array([ 0.9456868,  -0.0903203,  -0.10497134,  0.02873755,  0.09236028]),
        'c_tn' : np.array([0.73862517]),
        'd_tn' : np.array([0.4235527])
    }
    return output

