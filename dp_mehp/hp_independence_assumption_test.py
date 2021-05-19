import torch
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from all_aux_files import FCCondGen, ConvCondGen, find_rho, ME_with_HP, get_mnist_dataloaders
from all_aux_files import get_dataloaders, log_args, test_results_subsampling_rate
from all_aux_files import synthesize_data_with_uniform_labels, flatten_features, log_gen_data
from all_aux_files import heuristic_for_length_scale, plot_mnist_batch
from all_aux_files_tab_data import ME_with_HP_tab
from collections import namedtuple
import faulthandler
import matplotlib

matplotlib.use('Agg')
from autodp import privacy_calibrator

faulthandler.enable()  # prints stacktrace in case of segmentation fault

train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--data', type=str, default='digits', help='options are digits, fashion')

  parser.add_argument('--embed-batch-size', '-ebs', type=int, default=1000)

  parser.add_argument('--method', type=str, default='sum_kernel', help='')
  parser.add_argument('--order-hermite', type=int, default=100, help='')
  parser.add_argument('--kernel-length', type=float, default=0.001, help='')

  # parser.add_argument('--debug-data', type=str, default=None, choices=['flip', 'flip_binary', 'scramble_per_label'])

  ar = parser.parse_args()

  preprocess_args(ar)
  # log_args(ar.log_dir, ar)
  return ar


def preprocess_args(ar):
  if ar.seed is None:
    ar.seed = np.random.randint(0, 1000)
  assert ar.data in {'digits', 'fashion'}


def get_full_data_embedding(data_pkg, order, rho, embed_batch_size, device, data_key, separate_kernel_length,
                            debug_data):
  embedding_train_loader, _ = get_mnist_dataloaders(embed_batch_size, embed_batch_size,
                                                    use_cuda=device, dataset=data_key,
                                                    debug_data=debug_data)

  # summing at the end uses unnecessary memory - leaving previous version in in case of errors with this one
  data_embedding = torch.zeros(data_pkg.n_features * (order + 1), data_pkg.n_labels, device=device)
  for batch_idx, (data, labels) in enumerate(embedding_train_loader):
    data, labels = flatten_features(data.to(device)), labels.to(device)
    for idx in range(data_pkg.n_labels):
      idx_data = data[labels == idx]
      if separate_kernel_length:
        phi_data = ME_with_HP_tab(idx_data, order, rho, device, data_pkg.n_data)
      else:
        phi_data = ME_with_HP(idx_data, order, rho, device, data_pkg.n_data)
      data_embedding[:, idx] += phi_data

  del embedding_train_loader
  return data_embedding


def main():
  """Load settings"""
  ar = get_args()
  print(ar)
  torch.manual_seed(ar.seed)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  """Load data"""
  data_pkg = get_dataloaders(ar.data, ar.embed_batch_size, ar.embed_batch_size, use_cuda=device,
                             normalize=False, synth_spec_string=None, test_split=None,
                             debug_data=None)

  sigma2 = ar.kernel_length
  print('sigma2 is', sigma2)
  rho = find_rho(sigma2, False)
  order = ar.order_hermite

  dataset_embedding = get_full_data_embedding(data_pkg, order, rho, ar.embed_batch_size, device,
                                              ar.data, False,
                                              debug_data=None)

  scrambled_embedding = get_full_data_embedding(data_pkg, order, rho, ar.embed_batch_size, device,
                                                ar.data, False,
                                                debug_data='scramble_per_label')
  norm_1 = torch.norm(dataset_embedding)
  norm_2 = torch.norm(scrambled_embedding)
  norm_diff = torch.norm(dataset_embedding - scrambled_embedding)
  print('if HP indeed fails to model correlation, the two embeddings should be equal. (and thus the norm 0)')
  print(f'data embedding norm: {norm_1}, scrambled data norm: {norm_2}, norm of difference: {norm_diff}')


if __name__ == '__main__':
  main()
