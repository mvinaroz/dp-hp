import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from aux import plot_mnist_batch, NamedArray



if __name__ == '__main__':
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