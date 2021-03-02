import torch as pt
import numpy as np

def mmd_loss(data_enc, data_labels, gen_enc, gen_labels, n_labels, sigma2, method):

 # set gen labels to scalars from one-hot
 _, gen_labels = pt.max(gen_labels, dim=1)
 batch_size = data_enc.shape[0]
 mmd_sum = 0
 for idx in range(n_labels):
  idx_data_enc = data_enc[data_labels == idx]
  idx_gen_enc = gen_enc[gen_labels == idx]
  if method=='sum_kernel':
     mmd_sum += mmd_per_class(idx_data_enc, idx_gen_enc, pt.sqrt(sigma2))
  else:
     dxx, dxy, dyy = get_squared_dist(idx_data_enc, idx_gen_enc)
     mmd_sum += mmd_g(dxx, dxy, dyy, batch_size, sigma=pt.sqrt(sigma2))
 return mmd_sum


def get_squared_dist(x, y=None):
 """
 This function calculates the pairwise distance between x and x, x and y, y and y
 Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead
 :param x: batch_size-by-d matrix
 :param y: batch_size-by-d matrix
 :return:
 """

 xxt = pt.mm(x, x.t()) # (bs, bs)
 xyt = pt.mm(x, y.t()) # (bs, bs)
 yyt = pt.mm(y, y.t()) # (bs, bs)

 dx = pt.diag(xxt) # (bs)
 dy = pt.diag(yyt)

 dist_xx = pt.nn.functional.relu(dx[:, None] - 2.0 * xxt + dx[None, :])
 dist_xy = pt.nn.functional.relu(dx[:, None] - 2.0 * xyt + dy[None, :])
 dist_yy = pt.nn.functional.relu(dy[:, None] - 2.0 * yyt + dy[None, :])

 return dist_xx, dist_xy, dist_yy


def mmd_per_class(x, y, sigma2):
 """
 size(x) = mini_batch by feature_dimension
 size(y) = mini_batch by feature_dimension
 dist_xx = mini_batch by feature_dimension
 dist_yy = mini_batch by feature_dimension
 dist_xy = mini_batch by mini_batch by feature_dimension
 """

 m,feat_dim = x.shape
 n = y.shape[0]
 if m<1:
  m_div = 1e-3
 else:
  m_div = m
 if n<1:
  n_div = 1e-3
 else:
  n_div = n

 xx = pt.einsum('id,dj -> ijd', x, x.t()) # m by m by feature_dimension
 yy = pt.einsum('id,dj -> ijd', y, y.t()) # n by n by feature_dimension
 xy = pt.einsum('id,dj -> ijd', x, y.t()) #  m by n by feature_dimension

 # print('shape of xx', xx.shape)
 # print('shape of yy', yy.shape)
 # print('shape of xy', xy.shape)

 x2 = x**2 # m by feat_dim
 x2_extra_dim1 = x2[:,None,:]
 x2_extra_dim2 = x2[None,:,:]
 y2 = y**2
 y2_extra_dim1 = y2[:,None,:]
 y2_extra_dim2 = y2[None,:,:]

 # print('shape of x2', x2.shape)
 # print('shape of y2', y2.shape)

 # first term: sum_d sum_i sum_j k(x_i, x_j)
 dist_xx = pt.abs(x2_extra_dim1.repeat(1,m,1) - 2.0*xx + x2_extra_dim2.repeat(m,1,1))
 kxx = pt.sum(pt.exp(-dist_xx/(2.0*sigma2**2)))/(m_div**2)/feat_dim

 # second term: sum_d sum_i sum_j k(y_i, y_j)
 dist_yy = pt.abs(y2_extra_dim1.repeat(1,n,1) - 2.0*yy + y2_extra_dim2.repeat(n,1,1))
 kyy = pt.sum(pt.exp(-dist_yy/(2.0*sigma2**2)))/(n_div**2)/feat_dim

 # third term: sum_d sum_i sum_j k(x_i, y_j)
 dist_xy = pt.abs(x2_extra_dim1.repeat(1,n,1) - 2.0*xy + y2_extra_dim2.repeat(m,1,1))
 kxy = pt.sum(pt.exp(-dist_xy/(2.0*sigma2**2)))/(m_div*n_div)/feat_dim

 mmd = kxx + kyy - 2.0*kxy

 return mmd


def mmd_g(dist_xx, dist_xy, dist_yy, batch_size, sigma, upper_bound=None, lower_bound=None):
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

 # m = tf.constant(batch_size, tf.float32)
 e_kxx = matrix_mean_wo_diagonal(k_xx, batch_size)
 e_kxy = matrix_mean_wo_diagonal(k_xy, batch_size)
 e_kyy = matrix_mean_wo_diagonal(k_yy, batch_size)

 mmd = e_kxx + e_kyy - 2.0 * e_kxy
 return mmd


def matrix_mean_wo_diagonal(matrix, num_row, num_col=None):
 """ This function calculates the mean of the matrix elements not in the diagonal
 2018.4.9 - replace tf.diag_part with tf.matrix_diag_part
 tf.matrix_diag_part can be used for rectangle matrix while tf.diag_part can only be used for square matrix
 :param matrix:
 :param num_row:
 :type num_row: float
 :param num_col:
 :type num_col: float
 :param name:
 :return:
 """
 diff = pt.sum(matrix) - pt.sum((matrix.diag()))
 normalizer = num_row * (num_row - 1.0) if num_col is None else (num_row * num_col - min(num_col, num_row))
 return diff / normalizer
