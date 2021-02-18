#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:27:35 2021
Memory Analysis of Tensor product vs. Generalized Hermite Polynomials
@author: amin
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sm

Ds      =       np.arange(1, 10, 1)   #Dimensions of points
Tiis      =       np.zeros([len(Ds)], dtype=int)
Thr     =   1e-7

N   =   1000
#%% Find thresholds of orders using threshold on eig_vals
Ts  =   np.arange(0, 100)
xi  =   0.5
for i in range(len(Ds)):
    D   =   Ds[i]    
    Tis     =   (xi/2)**(Ts)/(sm.factorial(np.floor(Ts/D)))**D
    Tiis[i]   =   np.argmax(Tis<=Thr)-1


min_arrs    =   2*N*Tiis**Ds+(Tiis+1)**Ds/(sm.factorial(Ds))
min_arrs_tens   =   (2*N*Tiis[0]**Ds+(Tiis[0]+1)**Ds/(sm.factorial(Ds)))*Ds
plt.plot(Ds, 4*min_arrs, label='Generalized Hermite Proxy') #Number of Bytes
plt.plot(Ds, 4*min_arrs_tens, label='Tensor Product of Hermite Proxy')
plt.xlabel('Dimension of input gaussian data')
plt.ylabel('Memory in Bytes for Eigenvalue/Eigenfunction values')
plt.title('Memory analysis of MMD cross terms')
plt.legend()
    