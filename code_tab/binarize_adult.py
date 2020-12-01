import os
import numpy as np
from collections import OrderedDict

def binarize_data(data):
  n_samples, n_features = data.shape
  bin_features = []
  mapping_info = OrderedDict()
  for feat in range(n_features):
    bin_feat, map_tuple = binarize_feature(data[:, feat])
    bin_features.append(bin_feat)
    mapping_info[feat] = map_tuple
  bin_data = np.concatenate(bin_features, axis=1)
  return bin_data, mapping_info


def binarize_feature(feature):
  f_min, f_max = np.min(feature), np.max(feature)
  n_bin_features = int(np.ceil(np.log2(f_max - f_min)))  # make log(domain) columns
  bin_features = np.zeros((feature.shape[0], n_bin_features))
  mapping_base = np.arange(f_min, f_max+1)
  mapping_bin = np.zeros((mapping_base.shape[0], n_bin_features))
  for idx in range(n_bin_features):
    bin_features[:, idx] = feature % idx**2
    mapping_bin[:, idx] = mapping_base % idx**2
  return bin_features, (n_bin_features, mapping_base, mapping_bin)


def un_binarize_data(bin_data, mapping_info):
  bin_data_idx = 0
  unbin_features = []
  for feat_idx, (n_bin_features, mapping_base, mapping_bin) in mapping_info.items():
    bin_data_chunk = bin_data[:, bin_data_idx:bin_data_idx+1]