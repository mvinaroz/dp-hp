import torch as pt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from real_mmd_loss import get_real_mmd_loss
from dp_kmeans_balcan import alg4_private_clustering


def get_kmeans_mmd_loss(train_loader, n_labels, tgt_epsilon, n_means, mmd_sigma, batch_size, encoding_dim):
  means, labels = extract_kmeans(train_loader, n_labels, n_means, tgt_epsilon, encoding_dim)

  mmd_loss = get_real_mmd_loss(mmd_sigma, n_labels, batch_size)

  def kmeans_loss(gen_data, gen_labels):
    return mmd_loss(means, labels, gen_data, gen_labels)

  return kmeans_loss


def extract_kmeans(train_loader, n_labels, n_means, tgt_epsilon, encoding_dim):
  # go through full dataset, sort by label
  data_acc, label_acc = [], []
  for data, labels in train_loader:
    data_acc.append(data)
    label_acc.append(labels)

  data = pt.cat(data_acc, dim=0).numpy()
  data = np.reshape(data, (data.shape[0], -1))
  labels = pt.cat(label_acc, dim=0).numpy()

  if tgt_epsilon is None:
    delta, data_radius = None, None
  else:
    delta = 0.01
    data_radius = np.max(np.linalg.norm(data, axis=1))  # dp_kmeans requires centering of data for better performance

  means = []
  new_labels = []
  print('running kmeans per label')
  for l in range(n_labels):
    # select out label
    data_l = data[labels == l]
    if tgt_epsilon is None:
      means_l = MiniBatchKMeans(n_clusters=n_means).fit(data_l).cluster_centers_
    else:
      means_l, _ = alg4_private_clustering(data_l, tgt_epsilon, delta, n_means, data_radius, encoding_dim)
    print(f'label {l} with {data_l.shape[0]} samples done')
    means.append(means_l)
    new_labels.append(np.ones(means_l.shape[0]) * l)

  means = pt.tensor(np.concatenate(means), dtype=pt.float32)
  new_labels = pt.tensor(np.concatenate(new_labels), dtype=pt.int)
  return means, new_labels


