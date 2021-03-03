import torch as pt
import numpy as np

def mmd_loss(data_enc, data_labels, gen_enc, gen_labels, n_labels, sigma2, method):

 # set gen labels to scalars from one-hot
 _, gen_labels = pt.max(gen_labels, dim=1)
 batch_size = data_enc.shape[0]
 # feature_dim = data_enc.shape[1]
 mmd_sum = 0
 # mmd_sum1 = 0
 for idx in range(n_labels):
  idx_data_enc = data_enc[data_labels == idx]
  idx_gen_enc = gen_enc[gen_labels == idx]
  # print('# generated samples: %s for class %s' %(idx_gen_enc.shape[0], idx))
  # print('# real samples: %s for this class%s' %(idx_data_enc.shape[0],idx))
  # m = idx_data_enc.shape[0]
  # n = idx_gen_enc.shape[0]
  if method=='sum_kernel':
     mmd_sum += mmd_per_class(idx_data_enc, idx_gen_enc, pt.sqrt(sigma2), batch_size)

     # for dim in np.arange(0, feature_dim):
     #  print('dimension', dim)
     #  dxx, dxy, dyy = get_squared_dist(idx_data_enc[:, dim].unsqueeze(1), idx_gen_enc[:, dim].unsqueeze(1))
     #  mmd_sum1 += mmd_g(dxx, dxy, dyy, pt.sqrt(sigma2), batch_size)
  else:
     dxx, dxy, dyy = get_squared_dist(idx_data_enc, idx_gen_enc)
     mmd_sum += mmd_g(dxx, dxy, dyy, pt.sqrt(sigma2), batch_size)
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

 # dist_xx = pt.nn.functional.relu(dx[:, None] - 2.0 * xxt + dx[None, :])
 # dist_xy = pt.nn.functional.relu(dx[:, None] - 2.0 * xyt + dy[None, :])
 # dist_yy = pt.nn.functional.relu(dy[:, None] - 2.0 * yyt + dy[None, :])

 dist_xx = pt.abs(dx[:, None] - 2.0 * xxt + dx[None, :])
 dist_xy = pt.abs(dx[:, None] - 2.0 * xyt + dy[None, :])
 dist_yy = pt.abs(dy[:, None] - 2.0 * yyt + dy[None, :])

 return dist_xx, dist_xy, dist_yy


def mmd_per_class(x, y, sigma2, batch_size):
 """
 size(x) = mini_batch by feature_dimension
 size(y) = mini_batch by feature_dimension
 dist_xx = mini_batch by feature_dimension
 dist_yy = mini_batch by feature_dimension
 dist_xy = mini_batch by mini_batch by feature_dimension
 """

 m,feat_dim = x.shape
 n = y.shape[0]

 xx = pt.einsum('id,jd -> ijd', x, x) # m by m by feature_dimension
 yy = pt.einsum('id,jd -> ijd', y, y) # n by n by feature_dimension
 xy = pt.einsum('id,jd -> ijd', x, y) #  m by n by feature_dimension

 x2 = x**2 # m by feat_dim
 x2_extra_dim1 = x2[:,None,:]
 x2_extra_dim2 = x2[None,:,:]
 y2 = y**2
 y2_extra_dim1 = y2[:,None,:]
 y2_extra_dim2 = y2[None,:,:]

 # first term: sum_d sum_i sum_j k(x_i, x_j)
 dist_xx = pt.abs(x2_extra_dim1.repeat(1,m,1) - 2.0*xx + x2_extra_dim2.repeat(m,1,1))
 kxx = pt.sum(pt.exp(-dist_xx/(2.0*sigma2**2)))/(batch_size**2)

 # second term: sum_d sum_i sum_j k(y_i, y_j)
 dist_yy = pt.abs(y2_extra_dim1.repeat(1,n,1) - 2.0*yy + y2_extra_dim2.repeat(n,1,1))
 kyy = pt.sum(pt.exp(-dist_yy/(2.0*sigma2**2)))/(batch_size**2)

 # third term: sum_d sum_i sum_j k(x_i, y_j)
 dist_xy = pt.abs(x2_extra_dim1.repeat(1,n,1) - 2.0*xy + y2_extra_dim2.repeat(m,1,1))
 kxy = pt.sum(pt.exp(-dist_xy/(2.0*sigma2**2)))/(batch_size*batch_size)

 mmd = kxx + kyy - 2.0*kxy

 return mmd

def mmd_g(dist_xx, dist_xy, dist_yy, sigma, batch_size):

 # We compute the biased MMD estimator
 k_xx = pt.exp(-dist_xx / (2.0 * sigma ** 2))
 k_yy = pt.exp(-dist_yy / (2.0 * sigma ** 2))
 k_xy = pt.exp(-dist_xy / (2.0 * sigma ** 2))

 e_kxx = pt.sum(k_xx)/(batch_size**2)
 e_kyy = pt.sum(k_yy)/(batch_size**2)
 e_kxy = pt.sum(k_xy)/(batch_size**2)

 mmd = e_kxx + e_kyy - 2.0 * e_kxy
 return mmd

