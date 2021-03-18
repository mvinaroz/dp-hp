# this script is taken from https://stackoverflow.com/questions/50322833/higher-order-gradients-in-pytorch

import torch
from torch.autograd import grad

def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
        print('order %s th grads are' %(i+1))
        print(grads)

    return grads

# x = torch.arange(4.0, requires_grad=True).reshape(2, 2)
x = torch.tensor([1.0,3.0], requires_grad=True)
loss = (x ** 4).sum()

print('x is', x)
nth_derivative(f=loss, wrt=x, n=3)