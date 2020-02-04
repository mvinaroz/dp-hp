import numpy as np
from aux import plot_mnist_batch

# loads = np.load('dp_cgan_synth_mnist_eps9.6.npz')
# loads = np.load('dp-cgan-synth-mnist-eps9.60.npz')
loads = np.load('ref_dpcgan_fashion1-eps9.6.npz')
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
plot_mnist_batch(plot_mat, 10, 10, 'dp_cgan_plot', denorm=False, save_raw=False)
