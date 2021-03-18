# this script is taken from https://stackoverflow.com/questions/50322833/higher-order-gradients-in-pytorch

import torch
from torch.autograd import grad
import math
from math import factorial
import numpy as np

def nth_derivative(f, wrt, n, rho):

    store_varying_term = torch.ones((wrt.shape[0], n+1))
    for i in range(n):
        # i is the order of HP
        const = rho**(0.5*(i+1))/np.sqrt(factorial(i+1)*2**(i+1))
        print('const', const)
        grads = grad(const*f, wrt, create_graph=True)[0]
        f = grads.sum()/const
        print('order %s th grads are' %(i+1))
        print(grads)
        store_varying_term[:,i+1] = grads*(-1)**(i+1)*torch.exp(-x**2)

    return grads, store_varying_term

# x = torch.arange(4.0, requires_grad=True).reshape(2, 2)
x = torch.tensor([1.0,3.0], requires_grad=True)
# loss = (x ** 4).sum()
loss = (torch.exp(-x**2)).sum()
rho = 0.9
print('x is', x)
grads, store_varying_term = nth_derivative(f=loss, wrt=x, n=3, rho=rho)
