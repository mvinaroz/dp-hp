import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np


def make_data_1d(center_a=-1., center_b=1.):
  n_data_per_class = 5000

  sigma = 0.5
  data_a = np.random.normal(center_a, sigma, size=(n_data_per_class, 1))
  data_b = np.random.normal(center_b, sigma, size=(n_data_per_class, 1))

  perm = np.random.permutation(2*n_data_per_class)
  data = np.concatenate([data_a, data_b])[perm]
  labels = np.concatenate([np.zeros(n_data_per_class, dtype=np.int), np.ones(n_data_per_class, dtype=np.int)])[perm]
  return data, labels


def plot_data_1d(data, labels, save_str):
  n_classes = int(np.max(labels)) + 1

  while len(data.shape) > 1:
    if data.shape[-1] == 1 and len(data.shape) == 2:
      data = data.flatten()
    else:
      data = np.sqrt(np.sum(data**2, axis=-1))

  plt.figure()
  data_by_class = [data[labels == c_idx].flatten() for c_idx in range(n_classes)]
  class_labels = [str(k) for k in range(n_classes)]
  plt.hist(data_by_class, bins=30, label=class_labels)
  # for d, l in zip(data_by_class, class_labels):
  #   plt.hist(d, label=l)

  plt.legend()
  plt.savefig(f'{save_str}.png')


if __name__ == '__main__':
  data, labels = make_data_1d()
  plot_data_1d(data, labels, save_str='1d_data_test')