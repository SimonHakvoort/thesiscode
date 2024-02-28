import numpy as np

def get_gev_initial_params():
    output = {
        'a_gev' : np.array([0.7259205]),
        'b_gev' : np.array([ 0.95463854, -0.15831928, -0.20902145, -0.01082994,  0.11957027]),
        'c_gev' : np.array([0.10003784]),
        'd_gev' : np.array([ 0.08915933,  0.00749009,  0.29409942,  0.4589052,  -0.01932204]),
        'e_gev' : np.array([0.6217914])
    }
    return output

def get_trunc_normal_initial_params():
    output = {
        'a_tn' : np.array([0.4743731]),
        'b_tn' : np.array([ 0.9456868,  -0.0903203,  -0.10497134,  0.02873755,  0.09236028]),
        'c_tn' : np.array([0.83862517]),
        'd_tn' : np.array([0.9235527])
    }
    return output

