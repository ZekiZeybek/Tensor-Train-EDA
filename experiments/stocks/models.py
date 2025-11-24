import numpy as np

def markowitz_obj(x_config, Q_matrix):
    '''
    QUBO objective function to be minimized.
    The function computes the value of the quadratic form x^T Q x,
    where Q is the QUBO matrix and x is the binary configuration vector.
    '''
    return x_config @ Q_matrix @ x_config # x_config.T @ Q_matrix @ x_config same 


def equal_weighted_portf_volat_minim_obj(x_config,sigma, n_min, n_max):
    '''
    Tensor Network Estimation of Distribution Algorithms: https://arxiv.org/pdf/2412.19780v1    
    '''
    n = np.sum(x_config)

    if n < n_min:
        return 100.0 * (n_min - n)
    elif n > n_max:
        return 100.0 * (n - n_max)
    else:
        f = (x_config @ sigma @ x_config)/n**2
        return f