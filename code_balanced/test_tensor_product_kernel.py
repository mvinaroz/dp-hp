# author: Mijung Park
# purpose: to test the quality of approximation using Hermite Polynomials to the Gaussian kernel
#           where we assume the kernel is defined as a tensor product kernel for computational easiness.
# theory from https://francisbach.com/hermite-polynomials/
# date: 07 Jan 2021

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

    all_entries = np.einsum('ij, kj -> ikj', eigen_funcs_x, eigen_funcs_x_prime)
    out = np.einsum('ikj,j -> ik', all_entries, eigen_vals)

    # out = np.sum(eigen_vals * eigen_funcs_x * eigen_funcs_x_prime, axis=1)

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

# first generate data
n_data = 2000
input_dim = 20 # data dimension

# generate data from two Gaussians
mean = np.zeros(input_dim)
cov = 0.5*np.eye(input_dim)
# cov[0,0] = 0.1 # so that each dimension has different variance
# cov[0,1] = 0.02 # so that the two dimensions are correlated
# cov[1,0] = 0.02
x = np.random.multivariate_normal(mean, cov, n_data)

mean_prime = 5 + np.zeros((input_dim, 1))
x_prime = np.random.multivariate_normal(mean, cov, n_data)

# evaluate the kernel function
med = meddistance(np.concatenate((x,x_prime),axis=0))
sigma2 = med**2
Gaussian_kernel = k.KGauss(sigma2=sigma2)
Kxy = np.mean(Gaussian_kernel(x, x_prime))

""" approximate the kernel function """
# if we use the same length scale per dimenson
alpha = 1 / (2.0 * sigma2)
# from this: alpha = rho / (1- rho**2), identify what rho is
sol = optimize.minimize_scalar(lambda r: (alpha - r / (1-r**2))**2, bounds=(0,1), method='bounded')
rho = sol.x
print(med, alpha, rho)

n_degree = 10
appr_val_mat = np.zeros((n_data, n_data, input_dim))
eigen_val_mat = np.zeros((n_degree+1,input_dim))
for axis in np.arange(input_dim):
    print(axis)
    x_axis = x[:,axis]
    x_axis = x_axis[:,np.newaxis]

    x_prime_axis = x_prime[:,axis]
    x_prime_axis = x_prime_axis[:,np.newaxis]

    appr_val_axis, eigen_vals_axis = approx(n_degree, rho, x_axis, x_prime_axis)
    appr_val_mat[:,:,axis] = appr_val_axis
    eigen_val_mat[:,axis] = eigen_vals_axis

appr_val = np.mean(np.prod(appr_val_mat, axis=2))
# print('eigen_vals', eigen_val_mat)
print('approximate value:', appr_val)
print('true value:', Kxy)

