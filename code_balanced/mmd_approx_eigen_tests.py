import os
import numpy as np
import torch as pt
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
from mmd_approx_eigen import get_constants, hermite_polynomial_induction, phi_i_fun, sqrt_lambda_i_fun, lambda_phi_induction, \
  debug_phi_induction, hermite_function_induction, batch_data_embedding, balanced_batch_data_embedding, \
  normalized_hermite_polynomial_induction, normalized_batch_data_embedding_debug, \
  normalized_batch_data_embedding_phi_debug, bach_batch_feature_embedding, bach_batch_feature_embedding_debug
from aux import meddistance
import kernel

def base_recursion_test():
  device = 'cpu'
  x_in = pt.tensor(1., device=device)
  max_degree = 10
  hermites = [None, None]

  for degree in range(max_degree + 1):
    h_d = hermite_polynomial_induction(hermites[-1], hermites[-2], degree, x_in)
    hermites.append(h_d)

  hermites = hermites[2:]
  print([k.item() for k in hermites])


def plot_1d_mapping(px_sigma, kernel_length, selected_degrees, probabilists, plot_basis=False, a=None, b=None):
  n_degrees = selected_degrees[-1] + 1
  # plot both the hermite polynomial and its normalized mapping form in the range of [0-1]
  device = 'cpu'
  if a is None and b is None:
    c_tup = get_constants(px_sigma, kernel_length)
  else:
    c_tup = get_constants(a=a, b=b)
  # x_in = pt.arange(0, 1, 0.01)
  x_in = pt.arange(-2, 2, 0.04)

  n_samples = x_in.shape[0]
  embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  val_i_minus_1, val_i_minus_2 = None, None
  for degree in range(n_degrees):
    if plot_basis:
      sqrt_2c = pt.sqrt(pt.tensor(2 * c_tup.c, device=device))
      h_i = hermite_polynomial_induction(val_i_minus_1, val_i_minus_2, degree, sqrt_2c * x_in, probabilists)
      val_i = phi_i_fun(h_i, x_in, c_tup, use_a=True)

    # mapping = lambda_i(degree, c_tup, device=device) * phi_i(h_i, x_in, c_tup, device=device)
    else:
      val_i = lambda_phi_induction(val_i_minus_1, val_i_minus_2, degree, x_in, c_tup, device, probabilists)
    val_i_minus_2 = val_i_minus_1
    val_i_minus_1 = val_i

    embedding[:, degree] = val_i  # multiply features, sum over samples

  embedding = embedding.numpy()
  abs_embedding = np.abs(embedding)
  max_vals = np.max(abs_embedding, axis=0)  # maximum value for each degree
  max_args = np.argmax(abs_embedding, axis=0) / 100

  # selected_vals = embedding[:, selected_degrees]

  if a is None and b is None:
    args_string = f'(sigma={px_sigma}, l={kernel_length})'
  else:
    args_string = f'(a={a}, b={b})'

  map_string = 'phi(x)' if plot_basis else 'sqrt(lambda) phi(x)'
  herm_type_string = 'prob' if probabilists else 'phys'

  plt.figure()
  plt.title(f'log max vals by degree {args_string} [{herm_type_string} hermite]')
  plt.plot(np.arange(n_degrees), np.log(max_vals))
  plt.xlabel('degree')
  plt.ylabel(f'log max_x ({map_string}) by degree')
  plt.savefig('plot_1d_maxvals.png')

  plt.figure()
  plt.title(f'max val args by degree {args_string} [{herm_type_string} hermite]')
  plt.plot(np.arange(n_degrees), max_args)
  plt.xlabel('degree')
  plt.ylabel(f'argmax_x ({map_string}) by degree')
  plt.savefig('plot_1d_maxargs.png')

  plt.figure()
  plt.title(f'mappings for selected degrees {args_string} [{herm_type_string} hermite]')

  for degree in selected_degrees:
    plt.plot(x_in, embedding[:, degree], label=f'degree {degree}')
  plt.xlabel('x')
  plt.ylabel(f'{map_string}')
  if len(selected_degrees) < 10:
    plt.legend()
  plt.savefig('plot_1d_selected.png')


def first_ten_polynomials(x, probabilists):
  if probabilists:
    x = x / pt.sqrt(pt.tensor(2.))
  h_0 = pt.ones(x.shape, device=x.device)
  h_1 = 2*x
  h_2 = 4*x**2    - 2
  h_3 = 8*x**3    - 12*x
  h_4 = 16*x**4   - 48*x**2   + 12
  h_5 = 32*x**5   - 160*x**3  + 120*x
  h_6 = 64*x**6   - 480*x**4  + 720*x**2    - 120
  h_7 = 128*x**7  - 1344*x**5 + 3360*x**3   - 1680*x
  h_8 = 256*x**8  - 3584*x**6 + 13440*x**4  - 13440*x**2 + 1680
  h_9 = 512*x**9  - 9216*x**7 + 48384*x**5  - 80640*x**3 + 30240*x
  polys = [h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9]

  if probabilists:
    for degree in range(10):
      polys[degree] = polys[degree] / 2**(degree/2)

  return pt.stack(polys, dim=-1)


def check_hermite_recursion():
  # function computes
  probabilists = True  # else use physicists Hermite polynomial
  use_pi = True  # use pi in lambda as in zhu et al
  device = 'cpu'
  n_degrees = 5
  x_in = pt.arange(-2, 2, .04)
  eigenfun = True
  c_tup = get_constants(a=1, b=3)  # matching zhu et al

  sqrt_2c = pt.sqrt(pt.tensor(2 * c_tup.c, device=device))

  h_true = first_ten_polynomials(sqrt_2c * x_in, probabilists)
  if n_degrees < 10:
    h_true = h_true[:, :n_degrees]
  lphi_true = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  phi_true = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  for degree in range(n_degrees):
    lambda_i_val = sqrt_lambda_i_fun(degree, c_tup, device, use_pi)
    lphi_true[:, degree] = lambda_i_val * phi_i_fun(h_true[:, degree], x_in, c_tup, use_a=not eigenfun)
    phi_true[:, degree] = phi_i_fun(h_true[:, degree], x_in, c_tup, use_a=not eigenfun)

    print(f'lambda {degree}={lambda_i_val**2}')  # this

  h_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  lphi_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  phi_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  h_i_minus_1, h_i_minus_2 = None, None
  lphi_i_minus_1, lphi_i_minus_2 = None, None
  phi_i_minus_1, phi_i_minus_2 = None, None
  for degree in range(n_degrees):

    h_i = hermite_polynomial_induction(h_i_minus_1, h_i_minus_2, degree, sqrt_2c * x_in, probabilists)
    h_i_minus_2 = h_i_minus_1
    h_i_minus_1 = h_i
    h_recursive[:, degree] = h_i

    debug_phi_i = debug_phi_induction(phi_i_minus_1, phi_i_minus_2, degree, x_in, c_tup, probabilists, eigenfun)
    phi_i_minus_2 = phi_i_minus_1
    phi_i_minus_1 = debug_phi_i
    phi_recursive[:, degree] = debug_phi_i

    lphi_i = lambda_phi_induction(lphi_i_minus_1, lphi_i_minus_2, degree, x_in, c_tup, device, probabilists, use_pi, eigenfun)
    lphi_i_minus_2 = lphi_i_minus_1
    lphi_i_minus_1 = lphi_i
    lphi_recursive[:, degree] = lphi_i


  print(pt.norm(h_true - h_recursive))
  # print((pt.abs(h_true / h_recursive)) <= (1 + 1e-5))
  # print((pt.abs(h_recursive / h_true)) <= (1 + 1e-5))
  print(pt.max(pt.abs(h_recursive / h_true)))
  print(pt.max(pt.abs(h_true / h_recursive)))
  # print(pt.argmax(pt.abs(h_true / h_recursive)))
  # print(h_true)
  # print(h_recursive)
  # print((pt.abs(h_true / h_recursive)).numpy())
  # print((h_true - h_recursive) / h_true)
  # relative error is low. better indicator than absolute error here, as numbers can grow very large.
  # relative error only breaks for H_{2n+1}(0) = 0, as we have a division by 0 in that case

  print(pt.norm(lphi_true - lphi_recursive))
  # print((pt.abs(lphi_true / lphi_recursive)) <= (1 + 1e-4))
  # print((pt.abs(lphi_recursive / lphi_true)) <= (1 + 1e-4))
  print(pt.max(pt.abs(lphi_recursive / lphi_true)[1:, :]))
  print(pt.max(pt.abs(lphi_true / lphi_recursive)[1:, :]))
  # print(pt.argmax(pt.abs(lphi_true / lphi_recursive)[1:, :]))
  # print(lphi_recursive)
  # print(lphi_true)
  # same for the stabilized lphi computation.

  ylim = 1
  # plot all lphi_true in one plot
  plt.figure()
  # plt.ylim(-ylim, ylim)
  for degree in range(n_degrees):
    # plt.plot(x_in, h_true[:, degree])
    plt.plot(x_in, phi_true[:, degree])
    # plt.plot(x_in, lphi_true[:, degree])
  plt.savefig('eigen_approx_debug_plots/check_hermite_lphi_true.png')

  plt.figure()
  # plt.ylim(-ylim, ylim)
  for degree in range(n_degrees):
    # plt.plot(x_in, h_recursive[:, degree])
    plt.plot(x_in, phi_recursive[:, degree])
    # plt.plot(x_in, lphi_recursive[:, degree])
  plt.savefig('eigen_approx_debug_plots/check_hermite_lphi_rec.png')


def check_hermite_function_variant(scaling=1.0, n_degrees=5):
  # function computes
  device = 'cpu'
  xlim = 6 / scaling
  x_in = pt.arange(-xlim, xlim, xlim/50)
  # c_tup = get_constants(a=1, b=3)  # matching zhu et al
  psi_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)

  psi_i_minus_1, psi_i_minus_2 = None, None
  for degree in range(n_degrees):
    psi_i = hermite_function_induction(psi_i_minus_1, psi_i_minus_2, degree, x_in, device, scaling=scaling)
    psi_i_minus_2 = psi_i_minus_1
    psi_i_minus_1 = psi_i
    psi_recursive[:, degree] = psi_i

  ylim = np.sqrt(scaling / np.sqrt(np.pi)) * 1.05
  plt.figure()
  plt.ylim(-ylim, ylim)
  plt.xlim(-xlim, xlim)
  for degree in range(n_degrees):
    plt.plot(x_in, psi_recursive[:, degree])
  plt.savefig(f'eigen_approx_debug_plots/check_hermite_psi_scale={scaling}.png')


def check_hermite_normalized():
  # function computes
  device = 'cpu'
  xlim = 2.
  n_degrees = 15
  use_pi = True
  plot_last_degree = False
  x_in = pt.arange(-xlim, xlim * 51 / 50, xlim / 50)[:, None]
  c_tup = get_constants(a=1, b=3)  # matching zhu et al
  # c_tup = get_constants(px_sigma=1, kernel_l=0.2)
  h_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  phi_reconstructed = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  lphi_reconstructed = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)

  sqrt_two_c = pt.tensor(np.sqrt(2 * c_tup.c))
  fac = np.pi if use_pi else 2 * c_tup.a
  exp_term = pt.exp(-(c_tup.c - c_tup.a) * x_in.flatten() ** 2)
  h_i_minus_1, h_i_minus_2 = None, None
  for degree in range(n_degrees):
    sqrt_lambda_i = pt.tensor((fac / c_tup.big_a) ** 0.25 * c_tup.big_b ** (degree / 2))
    h_i = normalized_hermite_polynomial_induction(h_i_minus_1, h_i_minus_2, degree, x_in * sqrt_two_c)
    # h_i = hermite_polynomial_induction(h_i_minus_1, h_i_minus_2, degree, x_in * sqrt_two_c)
    print(h_i.shape)
    h_i_minus_2 = h_i_minus_1
    h_i_minus_1 = h_i
    h_recursive[:, degree] = h_i.flatten()
    phi_reconstructed[:, degree] = h_i.flatten() * exp_term
    lphi_reconstructed[:, degree] = h_i.flatten() * exp_term * sqrt_lambda_i

  phi_recursive = normalized_batch_data_embedding_phi_debug(x_in, n_degrees, c_tup, device, eigenfun=False)
  phi_eigenfun = normalized_batch_data_embedding_phi_debug(x_in, n_degrees, c_tup, device, eigenfun=True)
  lphi_recursive = normalized_batch_data_embedding_debug(x_in, n_degrees, c_tup, device, use_pi=use_pi)

  degrees_list = list(range(n_degrees - 1, n_degrees)) if plot_last_degree else list(range(n_degrees))

  plt.figure()
  plt.xlim(-xlim, xlim)
  for degree in degrees_list:
    plt.plot(x_in, h_recursive[:, degree])
  plt.savefig(f'eigen_approx_debug_plots/check_hermite_mixed_poly.png')

  plt.figure()
  plt.xlim(0, 15)
  plt.ylim(0., 0.8)
  x_vals = np.arange(0., 15, 1.)
  y_vals = np.asarray([sqrt_lambda_i_fun(k, c_tup, device, use_pi=True)**2 for k in x_vals])
  plt.plot(x_vals, y_vals)
  plt.savefig(f'eigen_approx_debug_plots/zhu_fig_1_a.png')

  if n_degrees >= 6:
    plt.figure()
    plt.xlim(-xlim, xlim)
    plt.ylim(-3., 2.7)
    for degree in range(6):
      plt.plot(x_in, phi_recursive[:, degree])
    plt.savefig(f'eigen_approx_debug_plots/zhu_fig_1_d.png')

    plt.figure()
    plt.ylim(-0.2, 0.2)
    plt.xlim(-xlim, xlim)
    for degree in range(6):
      plt.plot(x_in, phi_eigenfun[:, degree])
    plt.savefig(f'eigen_approx_debug_plots/zhu_fig_1_b.png')

  if n_degrees >= 15:
    plt.figure()
    plt.plot(x_in, phi_recursive[:, 14])
    plt.savefig(f'eigen_approx_debug_plots/zhu_fig_1_c.png')

  # CHECKS OF RECURSIVE VS DIRECT COMPUTATION COMMENTED OUT

  # plt.figure()
  # plt.ylim(-ylim, ylim)
  # phi_two_recursive = phi_recursive[:, 2]
  # phi_two_reconstructed = phi_reconstructed[:, 2]
  # plt.plot(x_in, phi_two_recursive, label='induction')
  # plt.plot(x_in, phi_two_reconstructed, label='reconst')
  # ratio = phi_two_reconstructed / phi_two_recursive
  # normed_ratio = ratio * 0.25 / pt.max(ratio)
  # plt.plot(x_in, normed_ratio, label='ratio')
  # plt.plot(x_in, phi_two_reconstructed - phi_two_recursive, label='difference')
  # plt.legend()
  # plt.savefig(f'check_hermite_phi_comp.png')

  # plt.figure()
  # plt.ylim(-ylim, ylim)
  # for degree in degrees_list:
  #   # for degree in range(n_degrees - 1, n_degrees):
  #   plt.plot(x_in, phi_reconstructed[:, degree])
  # plt.savefig(f'check_hermite_mixed_phi_reconstructed.png')  # matches recursive phi to ensure computation is correct
  #
  # plt.figure()
  # for degree in degrees_list:
  #   # for degree in range(n_degrees - 1, n_degrees):
  #   plt.plot(x_in, lphi_reconstructed[:, degree])
  # plt.savefig(f'check_hermite_mixed_lphi_reconstructed.png')  # matches recursive lphi to ensure computation is correct

  plt.figure()
  plt.xlim(-xlim, xlim)
  for degree in degrees_list:
    plt.plot(x_in, lphi_recursive[:, degree])
  plt.savefig(f'eigen_approx_debug_plots/check_hermite_normalized_lphi.png')




def real_kxy(kernel_length, batch_size, x, y):
  # implementation of non-appriximate kxy mapping mostly for debugging purposes
  # get_squared_dist
  xxt = pt.mm(x, x.t())  # (bs, bs)
  xyt = pt.mm(x, y.t())  # (bs, bs)
  yyt = pt.mm(y, y.t())  # (bs, bs)
  dx = pt.diag(xxt)  # (bs)
  dy = pt.diag(yyt)
  d_xy = pt.nn.functional.relu(dx[:, None] - 2.0 * xyt + dy[None, :])

  # mmd_g
  k_xy = pt.exp(-d_xy / (2.0 * pt.tensor(kernel_length**2)))
  diff = pt.sum(k_xy) - pt.sum((k_xy.diag()))
  # normalizer = batch_size * (batch_size - 1.0)
  e_kxy = diff / (batch_size * (batch_size - 1))
  return e_kxy


def real_kxy_debug(kernel_length, batch_size, x, y, return_kxy=False):
  dist_mat = x - y.transpose(1, 0)
  k_xy = pt.exp(-(dist_mat**2) / (2.0 * pt.tensor(kernel_length**2)))
  e_kxy = pt.sum(k_xy) / batch_size**2
  if return_kxy:
    return e_kxy, k_xy
  else:
    return e_kxy


def check_approx_against_true():
  # sample data from simple distribution
  device = 'cpu'
  n_samples = 100
  px_sigma = 1.5  # increase sigma -> decrease e_kxy
  kernel_length = 1.5  # increase kernel_length -> increase e_kxy
  n_degrees_lphi = 6
  n_degrees_balanced = 100
  use_pi = False
  x_scale = 1.0
  y_scale = 1.0
  x = pt.randn(n_samples, 1, device=device) * x_scale
  y = pt.randn(n_samples, 1, device=device) * y_scale
  # x = pt.abs(pt.randn(n_samples, 1, device=device))
  # y = -pt.abs(pt.randn(n_samples, 1, device=device))
  # correction_factor = pt.tensor(1.5 * (px_sigma / kernel_length)**0.25)
  # correction_factor = pt.tensor(1.5 * (px_sigma / kernel_length) ** 0.33)
  correction_factor = pt.tensor(1.)
  # compute true kyy term
  e_kxy_true = real_kxy(kernel_length, n_samples, x, y)  # copy of old code (I think from Wittawat)
  e_kxy_true_debug, kxy_true = real_kxy_debug(kernel_length, n_samples, x, y, return_kxy=True)  # simple correct version

  # compute approximate term k(x,y) using un-normalized hermite ploynomials as suggested in the paper
  c_tup = get_constants(px_sigma=px_sigma, kernel_l=kernel_length)
  lphi_x = batch_data_embedding(x, n_degrees_lphi, c_tup, device, eigenfun=False, use_pi=use_pi)
  lphi_y = batch_data_embedding(y, n_degrees_lphi, c_tup, device, eigenfun=False, use_pi=use_pi)
  kxy_lphi = lphi_x @ lphi_y.transpose(1, 0)
  e_kxy_lphi = pt.sum(kxy_lphi) / n_samples ** 2

  # compute approximate term k(x,y) using un-normalized hermite ploynomials as suggested in the paper
  lphi_normalized_x = normalized_batch_data_embedding_debug(x, n_degrees_balanced, c_tup, device, use_pi=use_pi, eigenfun=False)
  lphi_normalized_y = normalized_batch_data_embedding_debug(y, n_degrees_balanced, c_tup, device, use_pi=use_pi, eigenfun=False)
  kxy_lphi_normalized = lphi_normalized_x @ lphi_normalized_y.transpose(1, 0)
  e_kxy_lphi_normalized = pt.sum(kxy_lphi) / n_samples ** 2

  kxy_lphi_normalized *= correction_factor
  e_kxy_lphi_normalized *= correction_factor

  # compute approximate term using normalized
  psi_x = balanced_batch_data_embedding(x, n_degrees_balanced, kernel_length, device)
  psi_y = balanced_batch_data_embedding(y, n_degrees_balanced, kernel_length, device)
  e_kxy_psi = pt.sum(psi_x @ psi_y.transpose(1, 0)) / n_samples ** 2

  print(f'e_kxy true: {e_kxy_true}, debug: {e_kxy_true_debug}')
  print(f'e_kxy lphi: {e_kxy_lphi}')
  print(f'e_kxy norm : {e_kxy_lphi_normalized}')
  print(f'e_kxy psi : {e_kxy_psi}')

  print('below: max, min, (mean)')
  plt.figure()
  lphi_diff = kxy_true - kxy_lphi
  print('diff lphi', pt.max(lphi_diff).item(), pt.min(lphi_diff).item())
  plt.hist(lphi_diff.flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_diffs_basic.png')

  plt.figure()
  normalized_diff = kxy_true - kxy_lphi_normalized
  print('diff normalized', pt.max(normalized_diff).item(), pt.min(normalized_diff).item())
  plt.hist(normalized_diff.flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_diffs_normed.png')

  plt.figure()
  # lphi_ratio = kxy_lphi / kxy_true
  lphi_ratio_stable = kxy_lphi / (pt.abs(kxy_true) + 1e-4)
  # print(pt.max(lphi_ratio), pt.min(lphi_ratio))
  print('ratio lphi', pt.max(lphi_ratio_stable).item(), pt.min(lphi_ratio_stable).item())
  plt.hist(lphi_ratio_stable.flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_ratios_basic.png')

  plt.figure()
  normalized_ratio = kxy_lphi_normalized / kxy_true
  print('ratio normalized', pt.max(normalized_ratio).item(), pt.min(normalized_ratio).item(),
        pt.mean(normalized_ratio).item())
  plt.hist((kxy_lphi_normalized / (kxy_true + 1e-6)).flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_ratios_mix_normed.png')


def wittawats_kernel_code():
  # first generate data
  n_data = 2000
  mean = 0
  x = mean + np.random.randn(n_data, 1)
  mean_prime = 0
  x_prime = mean_prime + np.random.randn(n_data, 1)
  # evaluate the kernel function
  from aux import meddistance
  med = meddistance(np.concatenate((x, x_prime), axis=0))
  sigma2 = med ** 2

  (n1, d1) = x.shape
  (n2, d2) = x_prime.shape
  print(x.dot(x_prime.T).shape)
  assert d1 == d2, 'Dimensions of the two inputs must be the same'
  d2 = np.sum(x ** 2, 1)[:, np.newaxis] - 2.0 * x.dot(x_prime.T) + np.sum(x_prime ** 2, 1)
  k_xy = np.mean(np.exp(-d2 / (2.0 * sigma2)))
  print(k_xy)
  e_kxy = real_kxy_debug(med, n_data, pt.tensor(x), pt.tensor(x_prime), return_kxy=False)
  print(e_kxy)
  

def bach_param_conversion(rho=None, alpha=None):
  assert not (alpha is None and rho is None)
  if alpha is None:
    alpha = rho / (1 - rho**2)
  else:
    rho = (alpha * np.sqrt(1/alpha**2 + 4) - 1) / (2 * alpha)

  print(f'alpha={alpha}, rho={rho}')
  return alpha, rho


def check_bach_against_true():
  # sample data from simple distribution
  device = 'cpu'
  n_samples = 10
  # kernel_length = 5.3  # increase kernel_length -> increase e_kxy
  n_degrees_lphi = 5
  x_scale = 1.0
  y_scale = 1.0
  x = pt.randn(n_samples, 1, device=device) * x_scale
  y = pt.randn(n_samples, 1, device=device) * y_scale + 5.
  # compute true kyy term
  # e_kxy_true = real_kxy(kernel_length, n_samples, x, y)  # copy of old code (I think from Wittawat)
  all_samples = pt.cat([x, y]).numpy()
  kernel_length = meddistance(all_samples)
  print(kernel_length, all_samples.shape)
  e_kxy_true, kxy_true = real_kxy_debug(kernel_length, n_samples, x, y, return_kxy=True)  # simple correct version

  Gaussian_kernel = kernel.KGauss(sigma2=kernel_length**2)
  e_kxy_other = np.mean(Gaussian_kernel(x.numpy(), y.numpy()))

  alpha, rho = bach_param_conversion(alpha=1 / (2 * kernel_length**2))
  # compute approximate term k(x,y) using un-normalized hermite ploynomials as suggested in the paper
  lphi_x = bach_batch_feature_embedding(x, n_degrees_lphi, rho, device)
  lphi_y = bach_batch_feature_embedding(y, n_degrees_lphi, rho, device)
  kxy_lphi = pt.squeeze(lphi_x) @ pt.squeeze(lphi_y).transpose(1, 0)
  e_kxy_lphi = pt.sum(kxy_lphi) / n_samples ** 2

  lphi_x_debug = bach_batch_feature_embedding_debug(x, n_degrees_lphi, rho, device)
  lphi_y_debug = bach_batch_feature_embedding_debug(y, n_degrees_lphi, rho, device)
  kxy_lphi_debug = pt.squeeze(lphi_x_debug) @ pt.squeeze(lphi_y_debug).transpose(1, 0)
  e_kxy_lphi_debug = pt.sum(kxy_lphi_debug) / n_samples ** 2


  print(f'e_kxy true: {e_kxy_true}')
  print(f'e_kxy other: {e_kxy_other}')
  print(f'e_kxy lphi: {e_kxy_lphi}')
  print(f'e_kxy debug prob: {e_kxy_lphi_debug}')

  print('below: max, min, (mean)')
  plt.figure()
  lphi_diff = kxy_true - kxy_lphi
  print('diff lphi', pt.max(lphi_diff).item(), pt.min(lphi_diff).item(), pt.median(lphi_diff).item())
  plt.hist(lphi_diff.flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_diffs_bach.png')

  plt.figure()
  # lphi_ratio = kxy_lphi / kxy_true
  lphi_ratio = kxy_lphi / (pt.abs(kxy_true))
  # print(pt.max(lphi_ratio), pt.min(lphi_ratio))
  print('ratio lphi', pt.max(lphi_ratio).item(), pt.min(lphi_ratio).item(), pt.median(lphi_ratio).item())
  plt.hist(lphi_ratio.flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_ratios_bach.png')


  plt.figure()
  # lphi_ratio = kxy_lphi / kxy_true
  lphi_ratio = kxy_lphi_debug / (pt.abs(kxy_true))
  # print(pt.max(lphi_ratio), pt.min(lphi_ratio))
  print('ratio lphi', pt.max(lphi_ratio).item(), pt.min(lphi_ratio).item(), pt.median(lphi_ratio).item())
  plt.hist(lphi_ratio.flatten(), bins=50)
  plt.yscale('log', nonposy='clip')
  plt.savefig('eigen_approx_debug_plots/kxy_ratios_bach_debug.png')

def hermite_induction_vs_numpy():
  from scipy.special import eval_hermite
  x = 1.
  n_degrees = 10
  orders = np.arange(n_degrees)
  H_k = eval_hermite(orders, x)

  batch_embedding = pt.empty(n_degrees, dtype=pt.float32, device='cpu')
  x_tsr = pt.tensor(x)
  h_i_minus_one, h_i_minus_two = None, None
  for degree in range(n_degrees):
    h_i = hermite_polynomial_induction(h_i_minus_one, h_i_minus_two, degree, x_tsr, probabilists=False)
    h_i_minus_two = h_i_minus_one
    h_i_minus_one = h_i
    batch_embedding[degree] = h_i

  print(H_k)
  print(batch_embedding.numpy())


def check_bach_recursion():
  # function computes
  device = 'cpu'
  xlim = 2.
  n_degrees = 10
  rho = 0.6
  x_in = pt.arange(-xlim, xlim * 51 / 50, xlim / 50)[:, None]
  lphi_debug = bach_batch_feature_embedding_debug(x_in, n_degrees, rho, device)
  lphi_recur = bach_batch_feature_embedding(x_in, n_degrees, rho, device)

  degree_select = 1
  plt.figure()
  plt.ylim(-2, 2)
  lphi_two_recur = lphi_recur[:, degree_select]
  lphi_two_debug = lphi_debug[:, degree_select]
  plt.plot(x_in, lphi_two_recur, label='induction')
  plt.plot(x_in, lphi_two_debug, label='debug')
  ratio = lphi_two_debug / lphi_two_recur
  normed_ratio = ratio * 0.25 / pt.max(ratio)
  plt.plot(x_in, normed_ratio, label='ratio')
  plt.plot(x_in, lphi_two_debug - lphi_two_recur, label='difference')
  plt.legend()
  plt.savefig(f'eigen_approx_debug_plots/bach_lphi_comp.png')


if __name__ == '__main__':
  # base_recursion_test()
  # plot_1d_mapping(px_sigma=0.2, kernel_length=0.8, selected_degrees=list(range(50)), probabilists=True)
  # check_hermite_recursion()
  # plot_1d_mapping(None, None, selected_degrees=list(range(16)), probabilists=True, plot_basis=True, a=1, b=3)
  # os.makedirs('eigen_approx_debug_plots/', exist_ok=True)

  check_approx_against_true()  # comparison with true squared exponential kernel.
  # check_hermite_normalized()  # plots using the normalized hermite to reproduce Zhu et al.
  # wittawats_kernel_code()
  # francis_bach_blog_variant(rho=0.7)
  # francis_bach_blog_variant(alpha=1.3725490196078434)
  # check_bach_against_true()
  # hermite_induction_vs_numpy()
  # check_bach_recursion()