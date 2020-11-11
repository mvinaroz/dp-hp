import numpy as np
import torch as pt
from collections import namedtuple
from aux import flat_data
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt


constants_tuple_def = namedtuple('constants', ['a', 'b', 'c', 'big_a', 'big_b'])


def get_constants(px_sigma=None, kernel_l=None, a=None, b=None):
  assert (px_sigma is not None and kernel_l is not None) or (a is not None and b is not None)

  if a is None and b is None:
    a = 1 / (4 * px_sigma**2)
    b = 1 / (2 * kernel_l**2)
  c = np.sqrt(a**2 + 2 * a * b)
  big_a = a + b + c
  big_b = b / big_a
  print(f'c_tup: a={a}, b={b}, c={c}, A={big_a}, B={big_b}')
  return constants_tuple_def(a=a, b=b, c=c, big_a=big_a, big_b=big_b)


def hermite_induction(h_n, h_n_minus_1, degree, x_in, probabilists=True):
  fac = 1 if probabilists else 2
  if degree == 0:
    return pt.tensor(1., dtype=pt.float32, device=x_in.device)
  elif degree == 1:
    return fac * x_in
  else:
    n = degree - 1
    h_n_plus_one = fac*x_in*h_n - fac*n*h_n_minus_1
    return h_n_plus_one


def lambda_phi_induction(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device,
                         probabilists=True, use_pi=False):
  fac = 1 if probabilists else 2
  if degree == 0:
    return non_i_terms(x_in, c_tup, device, use_pi)
  elif degree == 1:
    sqrt_big_b_and_two_c = pt.tensor(np.sqrt(c_tup.big_b * 2 * c_tup.c))
    return fac * x_in * sqrt_big_b_and_two_c * lphi_i_minus_one  # in this cas hb_n_minus_one is non_i_terms
  else:
    sqrt_big_b_and_two_c = pt.tensor(np.sqrt(c_tup.big_b * 2 * c_tup.c))
    term_one = fac * x_in * lphi_i_minus_one * sqrt_big_b_and_two_c
    term_two = fac * (degree - 1) * lphi_i_minus_two * c_tup.big_b
    lphi_i = term_one - term_two
    return lphi_i


def debug_phi_induction(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device, probabilists=True):
  fac = 1 if probabilists else 2
  if degree == 0:
    phi_term = pt.exp(-(c_tup.c - c_tup.a) * x_in ** 2)  # basis function as in both Zhu and GP book
    return phi_term
  elif degree == 1:
    sqrt_two_c = pt.tensor(np.sqrt(2 * c_tup.c))
    return fac * x_in * sqrt_two_c * lphi_i_minus_one  # in this cas hb_n_minus_one is non_i_terms
  else:
    sqrt_two_c = pt.tensor(np.sqrt(2 * c_tup.c))
    term_one = fac * sqrt_two_c * x_in * lphi_i_minus_one
    term_two = fac * (degree - 1) * lphi_i_minus_two
    phi_i = term_one - term_two
    return phi_i


def phi_i(h_i, x_in, c_tup, use_a=True):
  # zhu et al compute eigentfunction without a and basis function with a
  fac = c_tup.c - c_tup.a if use_a else c_tup.c
  return pt.exp(-fac * x_in**2) * h_i


def lambda_i(degree, c_tup, device, use_pi):
  # zhu uses pi, while the GP book uses 2a
  fac = np.pi if use_pi else 2*c_tup.a
  return pt.tensor((fac / c_tup.big_a)**(1/4) * c_tup.big_b**(degree/2), dtype=pt.float32, device=device)


def non_i_terms(x_in, c_tup, device, use_pi):
  fac = np.pi if use_pi else 2 * c_tup.a
  lambda_terms = pt.tensor((fac / c_tup.big_a)**(1/4), dtype=pt.float32, device=device)  # as in GP book
  # lambda_terms = pt.tensor((np.pi / c_tup.big_a) ** (1 / 4), dtype=pt.float32, device=device)  # as in zhu et al
  # phi_terms = pt.exp(-c_tup.c*x_in**2)  # eigenfunction u_k(x) according to Zhu et al.
  phi_terms = pt.exp(-(c_tup.c - c_tup.a) * x_in ** 2)  # basis function as in both Zhu and GP book
  return lambda_terms * phi_terms


def hermite_function_recursion(psi_i_minus_one, psi_i_minus_two, degree, x_in, device, scaling=1.):
  if degree == 0:
    psi_i = pt.tensor(np.sqrt(scaling/np.sqrt(np.pi)), dtype=pt.float32, device=device) * pt.exp(-(scaling**2/2) * x_in**2)
  elif degree == 1:
    psi_i = psi_i_minus_one * pt.tensor(np.sqrt(2) * scaling, dtype=pt.float32, device=device) * x_in
  else:
    term_one = pt.tensor(np.sqrt(2/(degree+1)) * scaling, dtype=pt.float32, device=device) * x_in * psi_i_minus_one
    term_two = pt.tensor(np.sqrt(degree/(degree+1)), dtype=pt.float32, device=device) * psi_i_minus_two
    psi_i = term_one - term_two
  return psi_i


def eigen_mapping(degree, x_in, h_i_minus_1, h_i_minus_2, c_tup, device):
  sqrt_2c = pt.sqrt(pt.tensor(2 * c_tup.c, device=device))
  h_i = hermite_induction(h_i_minus_1, h_i_minus_2, degree, sqrt_2c * x_in)
  mapping = lambda_i(degree, c_tup, device, use_pi=False) * phi_i(h_i, x_in, c_tup)
  return mapping, h_i


def lambda_i_no_b(c_tup, device):
  return pt.tensor(np.sqrt(2*c_tup.a / c_tup.big_a), dtype=pt.float32, device=device)


def batch_data_embedding(x_in, n_degrees, c_tup, device):
  # since the embedding is a scalar operation prior to to taking the product, we compure for increasing degrees,
  # one at a time. separation by label is done outside of this function
  n_samples = x_in.shape[0]
  batch_embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  lphi_i_minus_1, lphi_i_minus_2 = None, None
  for degree in range(n_degrees):
    # mapping, h_i = eigen_mapping(degree, x_in, h_i_minus_1, h_i_minus_2, c_tup, device)
    # mapping, hb_i = stable_eigen_mapping(degree, x_in, hb_i_minus_1, hb_i_minus_2, c_tup, device)
    lphi_i = lambda_phi_induction(lphi_i_minus_1, lphi_i_minus_2, degree, x_in, c_tup, device)

    # print(f'd{degree}: {pt.max(pt.abs(lphi_i)).item()}')
    # print(f'd{degree}: {pt.max(pt.abs(hb_i)).item()}')
    # print(pt.mean(mapping))
    lphi_i_minus_2 = lphi_i_minus_1
    lphi_i_minus_1 = lphi_i

    batch_embedding[:, degree] = pt.prod(lphi_i, dim=1)  # multiply features, sum over samples
  return batch_embedding


def data_label_embedding(data, labels, n_degrees, c_tup,
                         labels_to_one_hot=False, n_labels=None, device=None):
  if labels_to_one_hot:
    batch_size = data.shape[0]
    one_hots = pt.zeros(batch_size, n_labels, device=device)
    one_hots.scatter_(1, labels[:, None], 1)
    labels = one_hots

  data_embedding = batch_data_embedding(data, n_degrees, c_tup, device)
  embedding = pt.einsum('ki,kj->kij', [data_embedding, labels])
  return pt.sum(embedding, 0)


def eigen_dataset_embedding(train_loader, device, n_labels, n_degrees, c_tup, sum_frequency=25):
  emb_acc = []
  n_data = 0

  for data, labels in train_loader:
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    emb_acc.append(data_label_embedding(data, labels, n_degrees, c_tup, labels_to_one_hot=True,
                                        n_labels=n_labels, device=device))
    n_data += data.shape[0]

    if len(emb_acc) > sum_frequency:
      emb_acc = [pt.sum(pt.stack(emb_acc), 0)]

  print('done collecting batches, n_data', n_data)
  emb_acc = pt.sum(pt.stack(emb_acc), 0) / n_data
  print(pt.norm(emb_acc), emb_acc.shape)
  # noise = pt.randn(d_rff, n_labels, device=device)
  # noisy_emb = emb_acc + noise
  return emb_acc


def get_real_kyy(kernel_length, n_labels):

  def real_kyy(gen_enc, gen_labels, batch_size):
    # set gen labels to scalars from one-hot
    _, gen_labels = pt.max(gen_labels, dim=1)
    kyy_sum = 0
    for idx in range(n_labels):
      idx_gen_enc = gen_enc[gen_labels == idx]
      dyy = get_squared_dist(idx_gen_enc)
      kyy_sum += estimate_kyy(dyy, sigma=kernel_length)

    return kyy_sum / batch_size**2

  return real_kyy


def get_squared_dist(y):
    yyt = pt.mm(y, y.t())  # (bs, bs)
    dy = pt.diag(yyt)
    dist_yy = pt.nn.functional.relu(dy[:, None] - 2.0 * yyt + dy[None, :])
    return dist_yy


def estimate_kyy(dist_yy, sigma):
  k_yy = pt.exp(-dist_yy / (2.0 * sigma ** 2))
  # k_yy = pt.exp(-dist_yy / (2.0 * sigma))

  # matrix_mean_wo_diagonal
  diff = pt.sum(k_yy) - pt.sum((k_yy.diag()))
  # normalizer = batch_size * (batch_size - 1.0)
  # e_kyy = diff / normalizer
  # return e_kyy
  return diff


def get_eigen_losses(train_loader, device, n_labels, n_degrees, kernel_length, px_sigma):
  c_tup = get_constants(px_sigma, kernel_length)

  data_emb = eigen_dataset_embedding(train_loader, device, n_labels, n_degrees, c_tup)
  real_kyy_fun = get_real_kyy(kernel_length, n_labels)

  def single_release_loss(gen_features, gen_labels):
    batch_size = gen_features.shape[0]
    gen_emb = data_label_embedding(gen_features, gen_labels, n_degrees, c_tup, device=device)
    print('embedding norms:', pt.norm(data_emb).item(), pt.norm(gen_emb).item() / batch_size)
    cross_term = pt.sum(data_emb * gen_emb) / batch_size  # data_emb is already normalized. -> normalize gen_emb
    gen_term = real_kyy_fun(gen_features, gen_labels, batch_size)  # this term is normalized already
    print(f'L_yy={gen_term}, L_xy={cross_term}')

    approx_loss = gen_term - 2 * cross_term
    # approx_loss = - 2 * cross_term
    return approx_loss

  return single_release_loss, data_emb


def base_recursion_test():
  device = 'cpu'
  x_in = pt.tensor(1., device=device)
  max_degree = 10
  hermites = [None, None]

  for degree in range(max_degree + 1):
    h_d = hermite_induction(hermites[-1], hermites[-2], degree, x_in)
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
      h_i = hermite_induction(val_i_minus_1, val_i_minus_2, degree, sqrt_2c * x_in, probabilists)
      val_i = phi_i(h_i, x_in, c_tup, use_a=True)

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
  n_degrees = 10
  x_in = pt.arange(-2, 2, .04)
  c_tup = get_constants(a=1, b=3)  # matching zhu et al

  sqrt_2c = pt.sqrt(pt.tensor(2 * c_tup.c, device=device))

  h_true = first_ten_polynomials(sqrt_2c * x_in, probabilists)
  if n_degrees < 10:
    h_true = h_true[:, :n_degrees]
  lphi_true = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  phi_true = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  for degree in range(n_degrees):
    lambda_i_val = lambda_i(degree, c_tup, device, use_pi)
    lphi_true[:, degree] = lambda_i_val * phi_i(h_true[:, degree], x_in, c_tup)
    phi_true[:, degree] = phi_i(h_true[:, degree], x_in, c_tup)

    print(f'lambda {degree}={lambda_i_val**2}')  # this

  h_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  lphi_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  phi_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)
  h_i_minus_1, h_i_minus_2 = None, None
  lphi_i_minus_1, lphi_i_minus_2 = None, None
  phi_i_minus_1, phi_i_minus_2 = None, None
  for degree in range(n_degrees):

    h_i = hermite_induction(h_i_minus_1, h_i_minus_2, degree, sqrt_2c * x_in, probabilists)
    h_i_minus_2 = h_i_minus_1
    h_i_minus_1 = h_i
    h_recursive[:, degree] = h_i

    debug_phi_i = debug_phi_induction(phi_i_minus_1, phi_i_minus_2, degree, x_in, c_tup, device, probabilists)
    phi_i_minus_2 = phi_i_minus_1
    phi_i_minus_1 = debug_phi_i
    phi_recursive[:, degree] = debug_phi_i

    lphi_i = lambda_phi_induction(lphi_i_minus_1, lphi_i_minus_2, degree, x_in, c_tup, device, probabilists, use_pi)
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
  plt.ylim(-ylim, ylim)
  for degree in range(n_degrees):
    # plt.plot(x_in, h_true[:, degree])
    # plt.plot(x_in, phi_true[:, degree])
    plt.plot(x_in, lphi_true[:, degree])
  plt.savefig('check_hermite_lphi_true.png')

  plt.figure()
  plt.ylim(-ylim, ylim)
  for degree in range(n_degrees):
    # plt.plot(x_in, h_recursive[:, degree])
    # plt.plot(x_in, phi_recursive[:, degree])
    plt.plot(x_in, lphi_recursive[:, degree])
  plt.savefig('check_hermite_lphi_rec.png')


def check_hermite_normalized(scaling=1.0, n_degrees=5):
  # function computes
  device = 'cpu'
  xlim = 6 / scaling
  x_in = pt.arange(-xlim, xlim, xlim/50)
  # c_tup = get_constants(a=1, b=3)  # matching zhu et al

  psi_recursive = pt.empty(x_in.shape[0], n_degrees, dtype=pt.float32, device=device)

  psi_i_minus_1, psi_i_minus_2 = None, None
  for degree in range(n_degrees):
    psi_i = hermite_function_recursion(psi_i_minus_1, psi_i_minus_2, degree, x_in, device, scaling=scaling)
    psi_i_minus_2 = psi_i_minus_1
    psi_i_minus_1 = psi_i
    psi_recursive[:, degree] = psi_i


  ylim = np.sqrt(scaling / np.sqrt(np.pi)) * 1.05
  plt.figure()
  plt.ylim(-ylim, ylim)
  plt.xlim(-xlim, xlim)
  for degree in range(n_degrees):
    plt.plot(x_in, psi_recursive[:, degree])
  plt.savefig(f'check_hermite_psi_scale={scaling}.png')


if __name__ == '__main__':
  # base_recursion_test()
  # plot_1d_mapping(px_sigma=0.2, kernel_length=0.8, selected_degrees=list(range(50)), probabilists=True)
  # check_hermite_recursion()
  # plot_1d_mapping(None, None, selected_degrees=list(range(16)), probabilists=True, plot_basis=True, a=1, b=3)
  check_hermite_normalized(scaling=0.25)
  check_hermite_normalized(scaling=0.5)
  check_hermite_normalized(scaling=1.0)
  check_hermite_normalized(scaling=2.0)
