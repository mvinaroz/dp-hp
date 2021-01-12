#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:33:36 2020

@author: amin

@Detail: 1- we generate two k-dimensional gaussian noises with different means mu1 and mu2
2- We find generalized Hermite polynomials up to sum{nu}=T
3 We calculate lambda and psi (eigenvalue and eigenfunction) of a kernel with C=I and an arbitrary xi
4- Find the vector of mapping using sqrt{lambda} times psi
5- Find MMD and approximate MMD using kernel and the new vector.
6- Find the mean cross term of two distributions.
7- Tune T and plot the decay of lambdas.
8- Plot the error between real and approximate  cross term (or MMD) and plot the iterations to reach them for different dimensions. I suggest you set a value for the error.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sm
from mpl_toolkits.mplot3d import Axes3D
import thewalrus
import kernel as u
from aux import meddistance
from scipy import optimize
from prod_tens_test import appr_iter

def err_cross_iters( k, iters):
    errs    =   np.zeros([iters])
    for i in range(iters):
        print('iteration '+str(i))
        errs[i]    =   err_cross(k)
    return np.average(errs)

def err_cross( k):
    #%% Parameters
    #k   =   1   #Dimension of Gaussians
    mu1     =   np.random.rand(k)
    cov1    =   0.1*np.eye(k)
    mu2     =   np.random.rand(k) 
    cov2    =   0.1*np.eye(k)
    N       =   10

    #%% Generate two Gaussian noises
    g1  =   np.random.multivariate_normal(mu1, cov1, N)
    g2  =   np.random.multivariate_normal(mu2, cov2, N)

    #%%Finding xi
    # evaluate the kernel function
    med = meddistance(np.concatenate((g1,g2),axis=0))
    sigma2 = med**2
    """ approximate the kernel function """
    # if we use the same length scale per dimenson
    alpha = 1 / (2.0 * sigma2)
    # from this: alpha = rho / (1- rho**2), identify what rho is
    # sol = optimize.minimize_scalar(lambda r: (alpha - r / (1-r**2))**2, bounds=(0,1), method='bounded')
    xi = -1/2/alpha+np.sqrt(1/alpha**2+4)/2
    print(xi)
    #%%Find T
    Ts  =   np.arange(0, 15)
    Tis     =   (xi/2)**(Ts)/(sm.factorial(np.floor(Ts/k)))**k
    T   =   np.argmax(Tis<=Thr)-1
    #%%Eigfuncs
    h1      =   np.zeros(np.hstack([N, T*np.ones([ k,], dtype=np.int)]))
    h2      =   np.zeros(np.hstack([N, T*np.ones([ k,], dtype=np.int)]))
    nus     =   np.mgrid[tuple(slice(0, T-1, complex(0, T)) for i in range(k))]
    #%%Gaussian Kernel
    Gaussian_Kernel     =   u.KGauss(sigma2=sigma2)
    

    
    #%% 3D Scatters
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter( g1[:, 0], g1[:, 1], g1[:, 2])
    # ax.scatter( g2[:, 0], g2[:, 1], g2[:, 2])
    
    #%%Hermite polynomials
    
    # I used TheWalrus library and I am not sure if it's the same Hermite that we need
    for i in range(N):
        h1[i]   =   thewalrus.hermite_multidimensional(2*np.eye(k), T, y=g1[i,:])
        
    for i in range(N):
        h2[i]   =   thewalrus.hermite_multidimensional(2*np.eye(k), T, y=g2[i,:])
    print('Found Hermite Polynomials for set1 and set2 of data with T='+str(T)+' and N='+str(N)+' and k='+str(k))
    #%% Eigenvalues and Eigenfunctions
    l1_nus      =   np.sum(nus, axis=0, keepdims=True)   #The first dimension still has 1 dims. However, it could be integrated into other nus expressions.
    facts       =   sm.factorial(nus) 
    eigv        =   ((xi/2)**(l1_nus))/(np.prod(facts, axis=0, keepdims=True))
    eigf1       =   (1-xi**2)**(k/4)*np.expand_dims(np.exp(- xi/(1+xi)*(np.linalg.norm(g1, axis=1, keepdims=True))**2), list(np.arange(2,k+1, dtype=int)))*h1
    eigf2       =   (1-xi**2)**(k/4)*np.expand_dims(np.exp(- xi/(1+xi)*(np.linalg.norm(g2, axis=1, keepdims=True))**2), list(np.arange(2,k+1, dtype=int)))*h2
    # print(np.shape(eigv))
    # print(np.shape(eigf1))
    print('Found Eigenvalues and Eigenfunctions for such T, N, and K')
    #%% Check the approximation
    # First, reshape the eigenfunctions to prepare them for crpss production. We can repeat one array several times.
    eigf2_mod   =   np.kron(np.ones(np.hstack(([N], np.ones([k+1,], dtype=int)))), eigf2) 
    
    cross_term_appr  =   np.sum(eigv*eigf1*eigf2_mod, axis=tuple(np.arange(2,k+2, dtype=int)))
    cross_term       =   Gaussian_Kernel(g1, g2)
    print('Found approximate and real cross-terms for such T, N, and K')
    # one_dim_appr_1    =   np.sum(eigv*h1, axis=tuple(np.arange(1,k+1, dtype=int)))
    # one_dim_1         =   np.exp(xi/2*(np.sum(g1, axis=1))-1/2*xi**2/4*k)
    
    # two_dim_appr    =    (1-xi**2)**(k/2)*np.sum((eigv*h1)*h2, axis=tuple(np.arange(1,k+1, dtype=int)))
    # two_dim         =  np.exp((2*np.sum(g1*g2, axis=1)*xi-(np.linalg.norm(g1, axis=1)**2+np.linalg.norm(g2, axis=1)**2)*(xi**2))/(1-xi**2))
    # # They are not similar to each other. Might be the case that I have not chosen a correct coefficient, etc. However, cross_term_appr does not have the right exponential form as well
    
    #%% Mappings
    
    mean_embedding_cross    =   np.average(cross_term)
    mean_embesdding_appr    =   np.average(cross_term_appr)
    return np.abs(mean_embedding_cross-mean_embesdding_appr)

#%% Parameters
#Ts      =       [5, 5, 5, 5, 5]   #Last power in each dimension of Hermite Polynomial
Ds      =       [2, 3, 4, 5, 6]   #Dimensions of points
Tiis      =       np.zeros([len(Ds)], dtype=int)
It      =       20       #Number of iterations to find the average error
Err     =       np.zeros([len(Ds)])
Err_tens     =       np.zeros([len(Ds)])
xi      =       0.7
Thr     =       1e-7
N       =       10

#%% Find the best xi


#%% Find thresholds of orders using threshold on eig_vals
# Ts  =   np.arange(0, 15)
# for i in range(len(Ds)):
#     D   =   Ds[i]    
#     Tis     =   (xi/2)**(Ts)/(sm.factorial(np.floor(Ts/D)))**D
#     Tiis[i]   =   np.argmax(Tis<=Thr)-1
#%% Find Errors
for k in range(len(Ds)):
    Err[k]  =   err_cross_iters(Ds[k], It)
    Err_tens[k]     =   appr_iter(Ds[k], It)
    print(k)

    
#%%Plot the Result
plt.plot(Ds, Err, label='Generalized Hermite Proxy')
plt.plot(Ds, Err_tens, label='Tensor Product of Hermite Proxy')
plt.xlabel('Dimension of input gaussian data')
plt.ylabel('Average error of the proxy of MMD cross terms')
plt.title('Error analysis of MMD cross terms')
plt.legend()
    