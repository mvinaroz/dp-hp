import numpy as np
import torch as pt
from collections import namedtuple
from aux import flat_data


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


def hermite_polynomial_induction(h_n, h_n_minus_1, degree, x_in, probabilists=True):
  fac = 1 if probabilists else 2
  if degree == 0:
    return pt.tensor(1., dtype=pt.float32, device=x_in.device)
  elif degree == 1:
    return fac * x_in
  else:
    n = degree - 1
    h_n_plus_one = fac*x_in*h_n - fac*n*h_n_minus_1
    return h_n_plus_one


def normalized_hermite_polynomial_induction(h_n, h_n_minus_1, degree, x_in):
  if degree == 0:
    return pt.tensor(np.pi**(-1/4), dtype=pt.float32, device=x_in.device) + pt.zeros(x_in.shape)
  elif degree == 1:
    return x_in * h_n * pt.sqrt(pt.tensor(2.))
  else:
    n = degree - 1
    h_n_plus_one = pt.sqrt(pt.tensor(2/degree))*x_in*h_n - pt.sqrt(pt.tensor(n/degree))*h_n_minus_1
    return h_n_plus_one


def lambda_phi_induction(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device,
                         probabilists=True, use_pi=False, eigenfun=False):
  fac = 1 if probabilists else 2
  if degree == 0:
    fac = np.pi if use_pi else (2 * c_tup.a)
    lambda_term = pt.tensor((fac / c_tup.big_a) ** (1 / 4), dtype=pt.float32, device=device)  # as in GP book
    # lambda_term = pt.tensor((np.pi / c_tup.big_a) ** (1 / 4), dtype=pt.float32, device=device)  # as in zhu et al
    # phi_term = pt.exp(-c_tup.c*x_in**2)  # eigenfunction u_k(x) according to Zhu et al.
    if eigenfun:
      phi_term = pt.exp(-c_tup.c * x_in ** 2)  # eigenfunction as in both Zhu and GP book
    else:
      phi_term = pt.exp(-(c_tup.c - c_tup.a) * x_in ** 2)  # basis function as in both Zhu and GP book
    return lambda_term * phi_term
  elif degree == 1:
    sqrt_big_b_and_two_c = pt.tensor(np.sqrt(c_tup.big_b * 2 * c_tup.c))
    return fac * x_in * sqrt_big_b_and_two_c * lphi_i_minus_one  # in this cas hb_n_minus_one is non_i_terms
  else:
    sqrt_big_b_and_two_c = pt.tensor(np.sqrt(c_tup.big_b * 2 * c_tup.c))
    term_one = fac * x_in * lphi_i_minus_one * sqrt_big_b_and_two_c
    term_two = fac * (degree - 1) * lphi_i_minus_two * c_tup.big_b
    lphi_i = term_one - term_two
    return lphi_i


def debug_phi_induction(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, probabilists=True, eigenfun=False):
  fac = 1 if probabilists else 2
  if degree == 0:
    if eigenfun:
      phi_term = pt.exp(-c_tup.c * x_in ** 2)  # eigenfunction as in both Zhu and GP book
    else:
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


def phi_i_fun(h_i, x_in, c_tup, use_a=True):
  # zhu et al compute eigentfunction without a and basis function with a
  fac = c_tup.c - c_tup.a if use_a else c_tup.c
  return pt.exp(-fac * x_in**2) * h_i


def lambda_i_fun(degree, c_tup, device, use_pi=False):
  # zhu uses pi, while the GP book uses 2a
  fac = np.pi if use_pi else (2*c_tup.a)
  return pt.tensor((fac / c_tup.big_a)**(1/4) * c_tup.big_b**(degree/2), dtype=pt.float32, device=device)


def hermite_function_induction(psi_i_minus_one, psi_i_minus_two, degree, x_in, device, scaling=1.):
  if degree == 0:
    psi_i = pt.tensor(np.sqrt(scaling/np.sqrt(np.pi)), dtype=pt.float32, device=device) * pt.exp(-(scaling**2/2) * x_in**2)
  elif degree == 1:
    psi_i = psi_i_minus_one * pt.tensor(np.sqrt(2) * scaling, dtype=pt.float32, device=device) * x_in
  else:
    term_one = pt.tensor(np.sqrt(2/(degree+1)) * scaling, dtype=pt.float32, device=device) * x_in * psi_i_minus_one
    term_two = pt.tensor(np.sqrt(degree/(degree+1)), dtype=pt.float32, device=device) * psi_i_minus_two
    psi_i = term_one - term_two
  return psi_i


def mixed_function_induction(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device, use_pi=False, eigenfun=False):
  if degree == 0:
    pi_fac = np.pi if use_pi else (2 * c_tup.a)
    lambda_term = pt.tensor((pi_fac / c_tup.big_a) ** 0.25, dtype=pt.float32, device=device)
    exp_fac = -c_tup.c if eigenfun else -(c_tup.c - c_tup.a)
    phi_term = pt.exp(exp_fac * x_in ** 2) #  / pt.tensor(np.pi ** 0.25)
    return lambda_term * phi_term
  elif degree == 1:
    sqrt_big_b_and_c = pt.tensor(np.sqrt(c_tup.big_b * c_tup.c))
    return 2 * sqrt_big_b_and_c * x_in * lphi_i_minus_one  # in this cas hb_n_minus_one is non_i_terms
  else:
    factor_one = pt.tensor(2 * np.sqrt(c_tup.big_b * c_tup.c/degree), dtype=pt.float32, device=device) * x_in
    factor_two = pt.tensor(c_tup.big_b * np.sqrt((degree-1)/degree), dtype=pt.float32, device=device)
    return factor_one * lphi_i_minus_one - factor_two * lphi_i_minus_two


def mixed_function_induction_phi_debug(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device, eigenfun=False):
  if degree == 0:
    exp_fac = -c_tup.c if eigenfun else -(c_tup.c - c_tup.a)
    phi_term = pt.exp(exp_fac * x_in ** 2)  # / pt.tensor(np.pi ** 0.25)
    if eigenfun:
      phi_term /= pt.tensor(2 * np.pi)  # for some reason scaling in Zhu et al is off by this factor
    return phi_term
  elif degree == 1:
    sqrt_c = pt.tensor(np.sqrt(c_tup.c))
    return 2 * sqrt_c * x_in * lphi_i_minus_one  # in this cas hb_n_minus_one is non_i_terms
  else:
    factor_one = pt.tensor(2 * np.sqrt(c_tup.c/degree), dtype=pt.float32, device=device) * x_in
    factor_two = pt.tensor(np.sqrt((degree - 1)/degree), dtype=pt.float32, device=device)
    return factor_one * lphi_i_minus_one - factor_two * lphi_i_minus_two

# def eigen_mapping(degree, x_in, h_i_minus_1, h_i_minus_2, c_tup, device):
#   sqrt_2c = pt.sqrt(pt.tensor(2 * c_tup.c, device=device))
#   h_i = hermite_induction(h_i_minus_1, h_i_minus_2, degree, sqrt_2c * x_in)
#   mapping = lambda_i(degree, c_tup, device, use_pi=False) * phi_i(h_i, x_in, c_tup)
#   return mapping, h_i
#
#
# def lambda_i_no_b(c_tup, device):
#   return pt.tensor(np.sqrt(2*c_tup.a / c_tup.big_a), dtype=pt.float32, device=device)


def batch_data_embedding(x_in, n_degrees, c_tup, device, eigenfun, use_pi):
  # since the embedding is a scalar operation prior to to taking the product, we compure for increasing degrees,
  # one at a time. separation by label is done outside of this function
  n_samples = x_in.shape[0]
  batch_embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  lphi_i_minus_1, lphi_i_minus_2 = None, None
  for degree in range(n_degrees):
    lphi_i = lambda_phi_induction(lphi_i_minus_1, lphi_i_minus_2, degree, x_in, c_tup, device,
                                  eigenfun=eigenfun, use_pi=use_pi)

    lphi_i_minus_2 = lphi_i_minus_1
    lphi_i_minus_1 = lphi_i

    batch_embedding[:, degree] = pt.prod(lphi_i, dim=1)  # multiply features, sum over samples
  return batch_embedding


def balanced_batch_data_embedding(x_in, n_degrees, kernel_length, device):
  # since the embedding is a scalar operation prior to to taking the product, we compure for increasing degrees,
  # one at a time. separation by label is done outside of this function
  n_samples = x_in.shape[0]
  batch_embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  psi_i_minus_one, psi_i_minus_two = None, None
  for degree in range(n_degrees):
    psi_i = hermite_function_induction(psi_i_minus_one, psi_i_minus_two, degree, x_in, device, kernel_length)
    psi_i_minus_two = psi_i_minus_one
    psi_i_minus_one = psi_i
    batch_embedding[:, degree] = pt.prod(psi_i, dim=1)  # multiply features, sum over samples
  return batch_embedding


def mixed_batch_data_embedding(x_in, n_degrees, c_tup, device, use_pi, eigenfun=False):
  # since the embedding is a scalar operation prior to to taking the product, we compure for increasing degrees,
  # one at a time. separation by label is done outside of this function
  n_samples = x_in.shape[0]
  batch_embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  lphi_i_minus_one, lphi_i_minus_two = None, None
  for degree in range(n_degrees):
    lphi_i = mixed_function_induction(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device,
                                      use_pi=use_pi, eigenfun=eigenfun)
    lphi_i_minus_two = lphi_i_minus_one
    lphi_i_minus_one = lphi_i
    batch_embedding[:, degree] = pt.prod(lphi_i, dim=1)  # multiply features, sum over samples
  return batch_embedding


def mixed_batch_data_embedding_phi_debug(x_in, n_degrees, c_tup, device, eigenfun):
  # since the embedding is a scalar operation prior to to taking the product, we compure for increasing degrees,
  # one at a time. separation by label is done outside of this function
  n_samples = x_in.shape[0]
  batch_embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  phi_i_minus_one, phi_i_minus_two = None, None
  for degree in range(n_degrees):
    phi_i = mixed_function_induction_phi_debug(phi_i_minus_one, phi_i_minus_two, degree, x_in, c_tup, device, eigenfun)
    phi_i_minus_two = phi_i_minus_one
    phi_i_minus_one = phi_i
    batch_embedding[:, degree] = pt.prod(phi_i, dim=1)  # multiply features, sum over samples
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
