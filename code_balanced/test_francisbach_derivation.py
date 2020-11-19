# author: Mijung Park
# purpose: to better understand the Gaussian kernel approximation using Hermite Polynomials
# theory from https://francisbach.com/hermite-polynomials/
# date: Nov 19 2020

import numpy as np
import kernel as k
from aux import meddistance
from scipy.special import eval_hermite
from scipy.special import factorial
from scipy import optimize

def approx(k, rho, x, x_prime):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    eigen_vals = (1-rho)*(rho**np.arange(0, k+1))
    eigen_funcs_x = eigen_func(k, rho, x) # output dim: number of datapoints by number of degree
    eigen_funcs_x_prime = eigen_func(k, rho, x_prime)
    out = np.mean(np.sum(eigen_vals * eigen_funcs_x * eigen_funcs_x_prime, axis=1))
    return out

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

# first generate data
n_data = 2000
mean = 0
x = mean + np.random.randn(n_data,1)

mean_prime = 10
x_prime = mean_prime + np.random.randn(n_data,1)

# evaluate the kernel function
med = meddistance(np.concatenate((x,x_prime),axis=0))
sigma2 = med**2
Gaussian_kernel = k.KGauss(sigma2=sigma2)
Kxy = np.mean(Gaussian_kernel(x, x_prime))

# # approximate the kernel function
alpha = 1 / (2.0 * sigma2)
# print(sigma2)
# print(alpha)

# from this: alpha = rho / (1- rho**2), identify what rho is
sol = optimize.minimize_scalar(lambda r: (alpha - r / (1-r**2))**2, bounds=(0,1), method='bounded')
rho = sol.x
# print(rho)

n_degree = 27
appr_val = approx(n_degree, rho, x, x_prime)
print('approximate value:', appr_val)
print('true value:', Kxy)

