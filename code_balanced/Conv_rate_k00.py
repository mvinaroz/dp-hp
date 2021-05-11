#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:16:30 2021
In this program, I will show the convergence rate of Phi(x).Phi(x) to k(0, 0). We are looking for a uniform bound
@author: mcharusaie
"""

from test_francisbach_estimation import approx
import numpy as np
import matplotlib.pyplot as plt

# %% Parameters
T = 60
X_lim = 30
rho = 0.5
X_step = 5
X = np.arange(0, X_lim, X_step)
X = np.reshape(X, [np.size(X), 1])
t = np.arange(1, T)
appr = np.zeros([np.size(X), np.size(t)])

# %% For Loop
for i in range(np.size(t)):
  for j in range(np.size(X)):
    appr[j, i], u = approx(t[i], rho, X[j, :].reshape([1, 1]), X[j, :].reshape([1, 1]))

# %%Plot
for j in range(np.size(X)):
  plt.plot(t, appr[j, :], label='X=' + str(X[j]))

plt.xlabel('Order of Approximation')
plt.ylabel('Approximated K(0, 0) via norm of mean embedding')
plt.title('Convergence Rate on Approxinmated K(0, 0)')
plt.legend()
