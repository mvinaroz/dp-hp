import numpy as np
import torch as pt
from collections import namedtuple
from aux import flat_data
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt


constants_tuple_def = namedtuple('constants', ['a', 'b', 'c', 'big_a', 'big_b'])


def get_constants(px_sigma, kernel_l):
  a = 1 / (4 * px_sigma**2)
  b = 1 / (2 * kernel_l**2)
  c = np.sqrt(a**2 + 2 * a * b)
  big_a = a + b + c
  big_b = b / big_a
  print(f'c_tup: a={a}, b={b}, c={c}, A={big_a}, B={big_b}')
  return constants_tuple_def(a, b, c, big_a, big_b)


def hermite_fibonacci_recursion(h_n, h_n_minus_1, degree, x_in):

  if degree == 0:
    return pt.tensor(1., dtype=pt.float32, device=x_in.device)
  elif degree == 1:
    return 2 * x_in
  else:
    n = degree - 1
    h_n_plus_one = 2*x_in*h_n - 2*n*h_n_minus_1
    # print(f'd{degree}: {pt.max(h_n_plus_one).item()}')
    return h_n_plus_one


def lambda_phi_recursion(lphi_i_minus_one, lphi_i_minus_two, degree, x_in, c_tup, device, probabilists=False):
  fac = 1 if probabilists else 2
  if degree == 0:
    return non_i_terms(x_in, c_tup, device)
  elif degree == 1:
    sqrt_big_b = pt.tensor(np.sqrt(c_tup.big_b))
    return fac * x_in * sqrt_big_b * lphi_i_minus_one  # in this cas hb_n_minus_one is non_i_terms
  else:
    sqrt_big_b = pt.tensor(np.sqrt(c_tup.big_b))
    lphi_i = fac*x_in*lphi_i_minus_one*sqrt_big_b - fac*(degree - 1)*lphi_i_minus_two*c_tup.big_b
    # print(f'd{degree}: {pt.max(h_n_plus_one).item()}')
    return lphi_i


def phi_i(h_i, x_in, c_tup, device):
  return pt.exp(-(c_tup.c - c_tup.a)*x_in**2) * h_i * pt.tensor(np.sqrt(2*c_tup.c), dtype=pt.float32, device=device) * x_in


def lambda_i(degree, c_tup, device):
  return pt.tensor((2*c_tup.a / c_tup.big_a)**(1/4) * c_tup.big_b**(degree/2), dtype=pt.float32, device=device)


def non_i_terms(x_in, c_tup, device):
  lambda_terms = pt.tensor((2*c_tup.a / c_tup.big_a)**(1/4), dtype=pt.float32, device=device)
  phi_terms = pt.exp(-(c_tup.c - c_tup.a)*x_in**2) * pt.tensor(np.sqrt(2*c_tup.c), dtype=pt.float32, device=device) * x_in
  return lambda_terms * phi_terms


def eigen_mapping(degree, x_in, h_i_minus_1, h_i_minus_2, c_tup, device):
  h_i = hermite_fibonacci_recursion(h_i_minus_1, h_i_minus_2, degree, x_in)
  mapping = lambda_i(degree, c_tup, device=device) * phi_i(h_i, x_in, c_tup, device=device)
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
    lphi_i = lambda_phi_recursion(lphi_i_minus_1, lphi_i_minus_2, degree, x_in, c_tup, device)

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
    h_d = hermite_fibonacci_recursion(hermites[-1], hermites[-2], degree, x_in)
    hermites.append(h_d)

  hermites = hermites[2:]
  print([k.item() for k in hermites])


def plot_1d_mapping(px_sigma, kernel_length, selected_degrees):
  n_degrees = selected_degrees[-1] + 1
  # plot both the hermite polynomial and its normalized mapping form in the range of [0-1]
  device = 'cpu'
  c_tup = get_constants(px_sigma, kernel_length)
  x_in = pt.arange(0, 1, 0.01)

  n_samples = x_in.shape[0]
  embedding = pt.empty(n_samples, n_degrees, dtype=pt.float32, device=device)
  lphi_i_minus_1, lphi_i_minus_2 = None, None
  for degree in range(n_degrees):
    # h_i = hermite_fibonacci_recursion(h_i_minus_1, h_i_minus_2, degree, x_in)
    # mapping = lambda_i(degree, c_tup, device=device) * phi_i(h_i, x_in, c_tup, device=device)
    lphi_i = lambda_phi_recursion(lphi_i_minus_1, lphi_i_minus_2, degree, x_in, c_tup, device, probabilists=False)
    lphi_i_minus_2 = lphi_i_minus_1
    lphi_i_minus_1 = lphi_i

    embedding[:, degree] = lphi_i  # multiply features, sum over samples

  embedding = embedding.numpy()
  abs_embedding = np.abs(embedding)
  max_vals = np.max(abs_embedding, axis=0)  # maximum value for each degree
  max_args = np.argmax(abs_embedding, axis=0) / 100

  # selected_vals = embedding[:, selected_degrees]

  plt.figure()
  plt.title(f'log max vals by degree (sigma={px_sigma}, l={kernel_length})')
  plt.plot(np.arange(n_degrees), np.log(max_vals))
  plt.xlabel('degree')
  plt.ylabel('log max_x (lambda phi(x)) by degree')
  plt.savefig('plot_1d_maxvals.png')

  plt.figure()
  plt.title(f'max val args by degree (sigma={px_sigma}, l={kernel_length})')
  plt.plot(np.arange(n_degrees), max_args)
  plt.xlabel('degree')
  plt.ylabel('argmax_x (lambda phi(x)) by degree')
  plt.savefig('plot_1d_maxargs.png')

  plt.figure()
  plt.title(f'mappings for selected degrees (sigma={px_sigma}, l={kernel_length})')

  for degree in selected_degrees:
    plt.plot(x_in, embedding[:, degree], label=f'degree {degree}')
  plt.xlabel('x')
  plt.ylabel('lambda * phi(x)')
  if len(selected_degrees) < 10:
    plt.legend()
  plt.savefig('plot_1d_selected.png')


if __name__ == '__main__':
  # base_recursion_test()
  plot_1d_mapping(px_sigma=0.19, kernel_length=0.41, selected_degrees=list(range(50)))
