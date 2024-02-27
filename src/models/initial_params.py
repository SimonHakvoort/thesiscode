import numpy as np

def get_gev_initial_params():
    output = {
        'a_gev' : np.array([0.4716438]),
        'b_gev' : np.array([ 0.9455215,  -0.09072546, -0.10659732,  0.03168057,  0.09318347]),
        'c_gev' : np.array([0.00019721]),
        'd_gev' : np.array([-3.96872820e-05, -9.66233783e-05,  5.97309445e-05, -1.32532659e-04, 1.04264735e-04]),
        'e_gev' : np.array([0.51883376])
    }
    return output

def get_trunc_normal_initial_params():
    output = {
        'a_tn' : np.array([0.4743731]),
        'b_tn' : np.array([ 0.9456868,  -0.0903203,  -0.10497134,  0.02873755,  0.09236028]),
        'c_tn' : np.array([[0.0002737]]),
        'd_tn' : np.array([-0.00017721])
    }
    return output