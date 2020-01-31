import numpy as np
from aux import plot_mnist_batch

loads = np.load('dp_cgan_synth_mnist_eps9.6.npz')
data, labels = loads['data'], loads['labels']

print(np.sum(labels, axis=0))
print(np.max(data), np.min(data))

plot_mat = data[:100]
plot_mnist_batch(plot_mat, 10, 10, 'dp_cgan_plot.png', denorm=False, save_raw=False)