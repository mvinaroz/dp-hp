import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
sys.path.insert(0, '../code/mnist/')
from synth_data_benchmark import test_passed_gen_data, datasets_colletion_def, load_mnist_data, subsample_data


def plot_mnist_batch(mnist_mat, n_rows, n_cols, save_path):
  bs = mnist_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  mnist_mat = np.reshape(mnist_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows*i:n_rows*(i+1)], n_rows) for i in range(n_cols)]
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

  plt.imsave(save_path + '.png', mnist_mat_flat, cmap=cm.gray, vmin=0., vmax=1.)


def first_look():
  mat = np.load('samples_d1.npy')
  print(mat.shape)
  print(np.max(mat), np.min(mat))

  mat = np.reshape(mat, (10, 6000, 784))
  print(mat.shape)

  mat1 = np.reshape(mat[:, :10, :], (100, 784))
  plot_mnist_batch(mat1, 10, 10, save_path='gs_test_plot')

  x_gen = np.reshape(mat, (60000, 784))
  y_gen = np.concatenate([np.zeros(6000, dtype=np.int) + k for k in range(10)])

  data_key = 'digits'
  x_real_train, y_real_train, x_real_test, y_real_test = load_mnist_data(data_key, False, base_dir='../code/mnist/data')

  if len(y_gen.shape) == 2:  # remove onehot
    if y_gen.shape[1] == 1:
      y_gen = y_gen.ravel()
    elif y_gen.shape[1] == 10:
      y_gen = np.argmax(y_gen, axis=1)
    else:
      raise ValueError


  rand_perm = np.random.permutation(y_gen.shape[0])
  x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]

  subsample = 0.1

  if subsample < 1.:
    x_gen, y_gen = subsample_data(x_gen, y_gen, subsample, balance_classes=True)
    x_real_train, y_real_train = subsample_data(x_real_train, y_real_train, subsample, balance_classes=True)


  data_col = datasets_colletion_def(x_gen, y_gen, x_real_train, y_real_train, x_real_test, y_real_test)
  test_passed_gen_data('gs_first_look', data_col, '.', subsample=subsample, custom_keys='mlp')


def first_look_dpgan():
  # mat = np.load('../dpgan-alternative/synth_data/apr17_sigma1.5_test/synthetic_mnist.npy')
  mat = np.load('../dpgan-alternative/synth_data/apr18_sigma1.5_baseline/synthetic_mnist.npz')
  # print(mat['data'])
  data = mat['data']
  targets = mat['labels']
  print(data.shape)
  print(targets.shape)
  print(np.max(data), np.min(data))

  data = np.reshape(data, (-1, 784))
  print(data.shape)

  mat1 = np.reshape(data[:100, :], (100, 784))
  plot_mnist_batch(mat1, 10, 10, save_path='dpgan_test_plot')

  mat2 = data[6000:6100, :]
  plot_mnist_batch(mat2, 10, 10, save_path='dpgan_test_zeros')


def gs_wgan_eval(load_path, save_dir, data_key, run_slow_models, subsample=1.0, custom_keys=None):
  mat = np.load(load_path)
  mat = np.reshape(mat, (10, 6000, 784))
  mat1 = np.reshape(mat[:, :10, :], (100, 784))

  os.makedirs(save_dir, exist_ok=True)
  plot_mnist_batch(mat1, 10, 10, save_path=os.path.join(save_dir, 'sample_plot'))

  x_gen = np.reshape(mat, (60000, 784))
  y_gen = np.concatenate([np.zeros(6000, dtype=np.int) + k for k in range(10)])

  # data_key = 'digits'
  x_real_train, y_real_train, x_real_test, y_real_test = load_mnist_data(data_key, False, base_dir='../code/mnist/data')

  rand_perm = np.random.permutation(y_gen.shape[0])
  x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]

  if subsample < 1.:
    x_gen, y_gen = subsample_data(x_gen, y_gen, subsample, balance_classes=True)
    x_real_train, y_real_train = subsample_data(x_real_train, y_real_train, subsample, balance_classes=True)

  data_col = datasets_colletion_def(x_gen, y_gen, x_real_train, y_real_train, x_real_test, y_real_test)

  test_passed_gen_data(None, data_col, save_dir, log_results=True, subsample=subsample, custom_keys=custom_keys,
                       skip_slow_models=not run_slow_models, only_slow_models=run_slow_models)


if __name__ == '__main__':
  # first_look()

  parser = argparse.ArgumentParser()
  parser.add_argument('--load-path', type=str, default=None)
  parser.add_argument('--save-dir', type=str, default=None)
  parser.add_argument('--data-key', type=str, default=None)
  parser.add_argument('--run-slow-models', action='store_true', default=True)

  ar = parser.parse_args()

  gs_wgan_eval(ar.load_path, ar.save_dir, ar.data_key, ar.run_slow_models, subsample=1.0, custom_keys=None)
