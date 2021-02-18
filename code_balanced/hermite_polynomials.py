import numpy as np
from scipy.special import eval_hermite
from scipy.special import factorial

def approx(k, rho, x, x_prime):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    eigen_vals = (1-rho)*(rho**np.arange(0, k+1))
    eigen_funcs_x = eigen_func(k, rho, x) # output dim: number of datapoints by number of degree
    eigen_funcs_x_prime = eigen_func(k, rho, x_prime)

    all_entries = np.einsum('ij, kj -> ikj', eigen_funcs_x, eigen_funcs_x_prime)
    out = np.einsum('ikj,j -> ik', all_entries, eigen_vals)

    return out, eigen_vals

def eigen_func(k, rho, x):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    orders = np.arange(0, k+1)
    H_k = eval_hermite(orders,x)  # input arguments: degree, where to evaluate at.
                             # output dim: number of datapoints by number of degree
    exp_trm = np.exp(-rho/(1+rho)*(x**2)) # output dim: number of datapoints by 1
    N_k = (2**orders)*(factorial(orders))*np.sqrt((1-rho)/(1+rho))
    eigen_funcs = 1/np.sqrt(N_k)*(H_k*exp_trm) # output dim: number of datapoints by number of degree
    return eigen_funcs