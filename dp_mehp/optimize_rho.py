# given the heuristic sigma2 and a chosen order C, optimize for 'rho' for HP expansion

import torch
import torch.nn as nn
import numpy as np
import kernel as k
from all_aux_files import meddistance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from all_aux_files import find_rho
from all_aux_files import get_dataloaders, flatten_features
from scipy.spatial.distance import cdist
from torch.nn.parameter import Parameter


# def find_rho(sigma2):
#   alpha = 1 / (2.0 * sigma2)
#   rho = -1 / 2 / alpha + torch.sqrt(1 / alpha ** 2 + 4) / 2
#   return rho

def phi_recursion(phi_k, phi_k_minus_1, rho, degree, x_in):
  if degree == 0:
    phi_0 = (1 - rho) ** (0.25) * (1 + rho) ** (0.25) * torch.exp(-rho / (1 + rho) * x_in ** 2)
    return phi_0
  elif degree == 1:
    phi_1 = torch.sqrt(2 * rho) * x_in * phi_k
    return phi_1
  else:  # from degree ==2 (k=1 in the recursion formula)
    k = degree - 1
    first_term = torch.sqrt(rho) / np.sqrt(2 * (k + 1)) * 2 * x_in * phi_k
    second_term = rho / np.sqrt(k * (k + 1)) * k * phi_k_minus_1
    phi_k_plus_one = first_term - second_term
    return phi_k_plus_one


def compute_phi(x_in, n_degrees, rho, device):
  first_dim = x_in.shape[0]
  batch_embedding = torch.empty(first_dim, n_degrees, dtype=torch.float32, device=device)
  # batch_embedding = torch.zeros(first_dim, n_degrees).to(device)
  phi_i_minus_one, phi_i_minus_two = None, None
  for degree in range(n_degrees):
    phi_i = phi_recursion(phi_i_minus_one, phi_i_minus_two, rho, degree, x_in.squeeze())
    batch_embedding[:, degree] = phi_i

    phi_i_minus_two = phi_i_minus_one
    phi_i_minus_one = phi_i

  return batch_embedding


def feature_map_HP(k, x, rho, device):
  # k: degree of polynomial
  # rho: a parameter (related to length parameter)
  # x: where to evaluate the function at

  arg = torch.arange(0, k + 1).to(device)
  eigen_vals = (1 - rho) * (rho ** arg)
  # print("eigenvalues", eigen_vals)
  eigen_vals = eigen_vals.to(device)
  phi_x = compute_phi(x, k + 1, rho, device)

  return phi_x, eigen_vals

class HP_expansion(nn.Module):
    def __init__(self, order, device):
        super(HP_expansion, self).__init__()
        self.parameter = Parameter(0.9*torch.ones(1), requires_grad=True)
        self.order = order
        self.device = device
    def forward(self, x):
        order = self.order
        device = self.device
        n_data, input_dim = x.shape
        # rho = torch.tanh(self.parameter).to(device)
        rho = torch.sigmoid(self.parameter).to(device)
        # rho = find_rho(self.parameter**2).to(device)
        x_flattened = x.view(-1)
        x_flattened = x_flattened[:, None]
        phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, rho, device)
        phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order + 1)
        phi_x = phi_x.type(torch.float)
        phi_x = phi_x / np.sqrt(input_dim)  # because we approximate k(x,x') = \sum_d k_d(x_d, x_d') / input_dim
        return phi_x

def main():

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data='fashion'
    # data = 'digits'
    batch_size = 200
    test_batch_size = 1000

    """Load data"""
    data_pkg = get_dataloaders(data, batch_size, test_batch_size, use_cuda=device, normalize=False,
                               synth_spec_string=None, test_split=None)

    # (1) compute true K using the median heuristic
    sigma2 = 127
    print('median heuristic finds sigma2 as ', sigma2)
    Gaussian_kernel = k.KGauss(sigma2=sigma2)

    epochs = 10
    order = 20
    model = HP_expansion(order, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            n = int(batch_size*0.5)
            x = data[0:n, :]  # Number of samples in x.
            x_prime = data[n:2*n, :]  # number of samples in x_prime.

            K = torch.Tensor(Gaussian_kernel(x.detach().cpu().numpy(), x_prime.detach().cpu().numpy()))

            # (2) compute the approximation using HP
            phi_1 = model(data[0:n,:])
            phi_2 = model(data[n:2*n,:])

            HP = torch.zeros((x.shape[0], x_prime.shape[0]))
            for h in range(x.shape[0]):
                for j in range(x_prime.shape[0]):
                    k_HP = torch.sum(torch.einsum('jk, jk-> jk', phi_1[h, :, :], phi_2[j, :, :]))
                    HP[h, j] = k_HP

            loss = torch.sum((K - HP) ** 2)
            # print('K',K)
            # print('HP', HP)
            # print('loss', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                'Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), data_pkg.n_data,
                                                               loss.item()))
            for param in model.parameters():
                print('estimated rho for the sum kernel is', torch.sigmoid(param.data))
        # end for

        # print(
        #     'Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), data_pkg.n_data, loss.item()))

    # for a given order and sigma2, the best rho is

if __name__ == '__main__':
  main()
