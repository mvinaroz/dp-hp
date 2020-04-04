import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from aux import plot_mnist_batch, NamedArray

import torch as pt
from backpack import extend, backpack
from backpack.extensions import BatchGrad, BatchL2Grad


def dp_sgd_example(model, input, target, loss_function, optimizer, clip_norm, noise_factor, cuda_device):
  # first we enable the model parameters to produce extra gradients. this can be done once outside the training loop
  model = extend(model)

  # you can also do the same for the loss but it's not necessary for 1st order gradient stats.
  # only use for builtin loss classes like nn.MSELoss(). it would look like this:
  # loss_function = extend(loss_function)

  # now the training loop would start:

  loss = loss_function(input, target)

  # now tell backward to produce (1) sample-wise grad and (2) squared norm of sample-wise grad
  with backpack(BatchGrad(), BatchL2Grad()):
    loss.backward()  # extra grads are stored under parameter.grad_batch and parameter.batch_l2

  # now we need the global sample-wise gradient norm to compute the clipping factors for each sample-wise gradient
  squared_param_norms = [p.batch_l2 for p in model.parameters()]  # first we get all the squared parameter norms...
  bp_global_norms = pt.sqrt(pt.sum(pt.stack(squared_param_norms), dim=0))  # ...then compute the global norms...
  global_clips = pt.clamp_max(clip_norm / bp_global_norms, 1.)  # ...and finally get a vector of clipping factors

  for idx, param in enumerate(model.parameters()):
    # for each parameter in the model, we now take the sample-wise grads and multply with the clipping factors
    clipped_sample_grads = param.grad_batch * expand_vector(global_clips, param.grad_batch)
    clipped_grad = pt.mean(clipped_sample_grads, dim=0)  # after clipping we average over the batch

    batch_size = clipped_sample_grads.shape[0]  # because we averaged, the sensitivity is divided by the batch-size
    noise_sdev = (2 * noise_factor * clip_norm / batch_size)  # gaussian noise standard deviation is computed...
    clipped_grad = clipped_grad + pt.rand_like(clipped_grad, device=cuda_device) * noise_sdev  # ...and applied
    param.grad = clipped_grad  # now we set the parameter gradient to what we just computed

  # and after this is done for all parameters, we update.
  optimizer.step()


def expand_vector(v, tgt_vec):
  # expand v to the number of dimensions of tgt_vec. I'm sure there is a nice way to do this but this works as well
  tgt_dims = len(tgt_vec.shape)
  if tgt_dims == 2:
    return v[:, None]
  elif tgt_dims == 3:
    return v[:, None, None]
  elif tgt_dims == 4:
    return v[:, None, None, None]
  elif tgt_dims == 5:
    return v[:, None, None, None, None]
  elif tgt_dims == 6:
    return v[:, None, None, None, None, None]
  else:
    return ValueError


def named_array_test():
  a = np.asarray(list(range(125)))
  a = np.reshape(a, (5, 5, 5))
  name_ids = [str(k) for k in range(5)]
  dim_names = ['a', 'b', 'c']
  idx_names = {'a': name_ids, 'b': name_ids, 'c': name_ids}
  named_arr1 = NamedArray(a, dim_names, idx_names)

  q1 = {'a': ['4', '2'], 'b': ['0', '1'], 'c': ['3', '4']}
  get1 = named_arr1.get(q1)

  q2 = {'b': ['1', '3', '0'], 'c': ['1', '4']}
  get2 = named_arr1.get(q2)

  print(a)
  print(get1)
  print(get2)

  a = np.asarray(list(range(1000, 1060)))
  a = np.reshape(a, (5, 3, 4))
  dim_names = ['a', 'b', 'c']
  idx_names = {'a': ['1', '2', '3', '7', '9'], 'b': ['10', '11', '12'], 'c': ['0', '1', '2', '3']}
  named_arr2 = NamedArray(a, dim_names, idx_names)

  merged_array = named_arr1.merge(named_arr2, merge_dim='b')
  print(merged_array.idx_names)
  print(merged_array.array)


def plot_dpcgan_data():
  # mat = np.load('dp-cgan-synth-mnist-eps1.0.npz')
  mat = np.load('dmnist-sig5-eps1.0.npz')
  data = mat['data']
  labels = mat['labels']
  mat_select = []
  for idx in range(10):
    ids = np.where(labels[:, idx] == 1)[0][:10]
    print(ids)
    mat_select.append(data[ids])
  mat_select = np.concatenate(mat_select, axis=0)
  print(mat_select.shape)
  plot_mnist_batch(mat_select, 10, 10, 'dmnist-sig5-plot', save_raw=False)


if __name__ == '__main__':
  # 'dpmerf-high-eps-f0'
  # mat = np.load('logs/gen/dpmerf-high-eps-d4/synth_eval/sub0.1_bagging_log.npz')
  # mat = np.load('logs/gen/dpmerf-high-eps-f4/synth_eval/sub0.1_bagging_log.npz')
  plot_dpcgan_data()
  # print(mat['accuracies'])

