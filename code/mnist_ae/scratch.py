import numpy as np
from aux import plot_mnist_batch, save_img
from torchvision import datasets


def dpcgan_plot():
  # loads = np.load('dp_cgan_synth_mnist_eps9.6.npz')
  loads = np.load('reference_dpcgan1_9.6.npz')
  # loads = np.load('ref_dpcgan_fashion5-eps9.6.npz')
  data, labels = loads['data'], loads['labels']

  print(np.sum(labels, axis=0))
  print(np.max(data), np.min(data))

  rand_perm = np.random.permutation(data.shape[0])
  data = data[rand_perm]
  labels = np.argmax(labels[rand_perm], axis=1)

  data_ids = [[], [], [], [], [], [], [], [], [], []]
  n_full = 0
  for idx in range(data.shape[0]):
    l = labels[idx]
    if len(data_ids[l]) < 10:
      data_ids[l].append(idx)
      # print(l)
      if len(data_ids[l]) == 10:
        n_full += 1
        if n_full == 10:
          break

  data_ids = np.asarray(data_ids)
  data_ids = np.reshape(data_ids, (100,))
  plot_mat = data[data_ids]
  plot_mnist_batch(plot_mat, 10, 10, 'dp_cgan_digit_plot', denorm=False, save_raw=False)


def dpgan_plot():
  data = np.load('dpgan_data.npy')

  rand_perm = np.random.permutation(data.shape[0])
  data = data[rand_perm] / 255.

  data = data[:100]
  print(np.max(data), np.min(data))
  plot_mnist_batch(data, 10, 10, 'dpgan_digit_plot', denorm=False, save_raw=False)


def direct_gen_redo_plot():
  data = np.load('MNISTsamples_ep5_raw.npy')

  mnist_mean = 0.1307
  mnist_sdev = 0.3081
  data = np.clip((data - mnist_mean) / mnist_sdev, a_min=0., a_max=1.)
  # data = data[:100]
  print(data.shape)
  print(np.max(data), np.min(data))
  # plot_mnist_batch(data, 10, 10, 'fmnist_direct_plot', denorm=False, save_raw=False)
  save_img('dmnist_direct_plot.png', data)


def data_plot():
  # loads = np.load('dp_cgan_synth_mnist_eps9.6.npz')

  # loads = np.load('ref_dpcgan_fashion5-eps9.6.npz')

  # train_data = datasets.MNIST('../../data', train=True)
  train_data = datasets.FashionMNIST('../../data', train=True)

  data, labels = train_data.data.numpy(), train_data.targets.numpy()
  data = np.reshape(data, (-1, 784)) / 255

  print(np.sum(labels, axis=0))
  print(np.max(data), np.min(data))

  # rand_perm = np.random.permutation(data.shape[0])
  # data = data[rand_perm]
  # labels = np.argmax(labels[rand_perm], axis=1)

  data_ids = [[], [], [], [], [], [], [], [], [], []]
  n_full = 0
  for idx in range(data.shape[0]):
    l = labels[idx]
    if len(data_ids[l]) < 10:
      data_ids[l].append(idx)
      # print(l)
      if len(data_ids[l]) == 10:
        n_full += 1
        if n_full == 10:
          break

  data_ids = np.asarray(data_ids)
  data_ids = np.reshape(data_ids, (100,))
  plot_mat = data[data_ids]
  plot_mnist_batch(plot_mat, 10, 10, 'data_fashion_plot', denorm=False, save_raw=False)


if __name__ == '__main__':
  # dpcgan_plot()
  # dpgan_plot()
  # direct_gen_redo_plot()
  data_plot()