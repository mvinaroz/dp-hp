import torch as pt

def mmd_loss(data_enc, data_labels, gen_enc, gen_labels, n_labels, sigma2):
    # set gen labels to scalars from one-hot
    _, gen_labels = pt.max(gen_labels, dim=1)
    _, data_labels = pt.max(data_labels, dim=1)

    # for each label, take the associated encodings
    # print('label shapes:', data_labels.shape, gen_labels.shape)
    mmd_sum = 0
    for idx in range(n_labels):
      idx_data_enc = data_enc[data_labels == idx]
      idx_gen_enc = gen_enc[gen_labels == idx]
      # print('sample selection shapes:', idx_data_enc.shape, idx_gen_enc.shape)
      # then for that label compute mmd:
      dxx, dxy, dyy = get_squared_dist(idx_data_enc, idx_gen_enc)
      mmd_sum += mmd_g(dxx, dxy, dyy, sigma=pt.sqrt(pt.tensor(sigma2)))

    return mmd_sum


def get_squared_dist(x, y=None):
    """
    This function calculates the pairwise distance between x and x, x and y, y and y
    Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead
    :param x: batch_size-by-d matrix
    :param y: batch_size-by-d matrix
    :return:
    """

    xxt = pt.mm(x, x.t())  # (bs, bs)
    xyt = pt.mm(x, y.t())  # (bs, bs)
    yyt = pt.mm(y, y.t())  # (bs, bs)

    dx = pt.diag(xxt)    # (bs)
    dy = pt.diag(yyt)

    dist_xx = pt.nn.functional.relu(dx[:, None] - 2.0 * xxt + dx[None, :])
    dist_xy = pt.nn.functional.relu(dx[:, None] - 2.0 * xyt + dy[None, :])
    dist_yy = pt.nn.functional.relu(dy[:, None] - 2.0 * yyt + dy[None, :])

    return dist_xx, dist_xy, dist_yy


def mmd_g(dist_xx, dist_xy, dist_yy, sigma, upper_bound=None, lower_bound=None):
  """This function calculates the maximum mean discrepancy with Gaussian distribution kernel
  The kernel is taken from following paper:
  Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
  MMD GAN: Towards Deeper Understanding of Moment Matching Network.
  :param dist_xx:
  :param dist_xy:
  :param dist_yy:
  :param batch_size:
  :param sigma:
  :param upper_bound: bounds for pairwise distance in mmd-g.
  :param lower_bound:
  :return:
  """

  if lower_bound is None:
    k_xx = pt.exp(-dist_xx / (2.0 * sigma ** 2))
    k_yy = pt.exp(-dist_yy / (2.0 * sigma ** 2))
  else:
    k_xx = pt.exp(-pt.max(dist_xx, lower_bound) / (2.0 * sigma ** 2))
    k_yy = pt.exp(-pt.max(dist_yy, lower_bound) / (2.0 * sigma ** 2))

  if upper_bound is None:
    k_xy = pt.exp(-dist_xy / (2.0 * sigma ** 2))
  else:
    k_xy = pt.exp(-pt.min(dist_xy, upper_bound) / (2.0 * sigma ** 2))

  e_kxx = matrix_mean_wo_diagonal(k_xx)
  e_kxy = matrix_mean_wo_diagonal(k_xy)
  e_kyy = matrix_mean_wo_diagonal(k_yy)

  mmd = e_kxx + e_kyy - 2.0 * e_kxy
  return mmd


def matrix_mean_wo_diagonal(matrix):

    num_row, num_col = matrix.shape

    if num_row == num_col:
        diff = pt.sum(matrix) - pt.sum((matrix.diag()))
        normalizer = num_row * (num_row-1) # unbiased estimator
    else:
        diff = pt.sum(matrix)
        normalizer = num_row*num_col

    return diff / normalizer
