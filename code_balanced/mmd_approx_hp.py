import numpy as np
import torch as pt


def feature_map_hp(k, x, rho, device):
  # k: degree of polynomial
  # rho: a parameter (related to length parameter)
  # x: where to evaluate the function at

  eigen_vals = (1 - rho) * (rho ** pt.arange(0, k + 1))
  # print("eigenvalues", eigen_vals)
  eigen_vals = eigen_vals.to(device)
  phi_x = compute_phi(x, k + 1, rho, device)

  return phi_x, eigen_vals


def hp_mean_embedding(x, order, rho, device, n_training_data):
  n_data, input_dim = x.shape

  # reshape x, such that x is a long vector
  x_flattened = x.view(-1)
  x_flattened = x_flattened[:, None]
  phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_hp(order, x_flattened, rho, device)
  phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order + 1)
  phi_x = phi_x.type(pt.float)

  sum_val = pt.sum(phi_x, axis=0)
  phi_x = sum_val / n_training_data

  phi_x = phi_x / np.sqrt(input_dim)  # because we approximate k(x,x') = \sum_d k_d(x_d, x_d') / input_dim

  phi_x = phi_x.view(-1)  # size: input_dim*(order+1)

  return phi_x


def phi_recursion(phi_k, phi_k_minus_1, rho, degree, x_in):
  if degree == 0:
    phi_0 = (1 - rho) ** (0.25) * (1 + rho) ** (0.25) * pt.exp(-rho / (1 + rho) * x_in ** 2)
    return phi_0
  elif degree == 1:
    phi_1 = np.sqrt(2 * rho) * x_in * phi_k
    return phi_1
  else:  # from degree ==2 (k=1 in the recursion formula)
    k = degree - 1
    first_term = np.sqrt(rho) / np.sqrt(2 * (k + 1)) * 2 * x_in * phi_k
    second_term = rho / np.sqrt(k * (k + 1)) * k * phi_k_minus_1
    phi_k_plus_one = first_term - second_term
    return phi_k_plus_one


def compute_phi(x_in, n_degrees, rho, device):
  first_dim = x_in.shape[0]
  batch_embedding = pt.empty(first_dim, n_degrees, dtype=pt.float32, device=device)
  # batch_embedding = pt.zeros(first_dim, n_degrees).to(device)
  phi_i_minus_one, phi_i_minus_two = None, None
  for degree in range(n_degrees):
    phi_i = phi_recursion(phi_i_minus_one, phi_i_minus_two, rho, degree, x_in.squeeze())
    batch_embedding[:, degree] = phi_i

    phi_i_minus_two = phi_i_minus_one
    phi_i_minus_one = phi_i

  return batch_embedding