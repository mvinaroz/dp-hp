#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:38:59 2021

@author: amin2


Goal: In this program I want to load a trained model and find out the evaluation of that
"""
import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import FCCondGen, ConvCondGen
from aux import plot_mnist_batch, log_args, flatten_features, log_final_score
from data_loading import get_dataloaders
from mmd_approx_rff import get_rff_losses
from mmd_approx_eigen import get_eigen_losses
from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def
from mmd_real import get_real_mmd_loss
from kmeans import get_kmeans_mmd_loss
from mmd_HP_pytorch import get_hp_losses
from synth_data_2d import plot_data
from synth_data_1d import plot_data_1d


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits, fashion and 2d')
  parser.add_argument('--create-dataset', action='store_true', default=True, help='if true, make 60k synthetic code_balanced')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=500)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--gen-batch-size', '-gbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=5)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  # parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
  parser.add_argument('--conv-gen', action='store_true', default=True, help='use convolutional generator')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='500,500', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=1000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=str, default=None, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=5.0, help='privacy noise parameter')

  # ALTERNATE MODES
  parser.add_argument('--single-release', action='store_true', default=True, help='get 1 data mean embedding only')

  parser.add_argument('--loss-type', type=str, default='rff', help='how to approx mmd',
                      choices=['rff', 'kmeans', 'real_mmd', 'eigen', 'hermite'])
  # parser.add_argument('--real-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')
  # parser.add_argument('--kmeans-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')

  parser.add_argument('--n-means', type=int, default=10, help='number of means to find per class')
  parser.add_argument('--dp-kmeans-encoding-dim', type=int, default=10, help='dimension the data is projected to')
  parser.add_argument('--tgt-epsilon', type=float, default=1.0, help='privacy epsilon for dp k-means')
  parser.add_argument('--kmeans-delta', type=float, default=0.01, help='soft failure probability in dp k-means')
  parser.add_argument('--mmd-type', type=str, default='sphere', help='how to approx mmd', choices=['sphere', 'r+r'])

  parser.add_argument('--center-data', action='store_true', default=False, help='k-means requires centering')

  # synth_d2 data
  # parser.add_argument('--synth-spec-string', type=str, default='norm_k5_n10000_row5_col5_noise0.2', help='')
  parser.add_argument('--synth-spec-string', type=str, default='norm_k5_n50000_row5_col5_noise0.2', help='')
  parser.add_argument('--test-split', type=float, default=0.1, help='only relevant for synth_2d so far')

  # eigen-approximation
  parser.add_argument('--kernel-length', type=float, default=0.1, help='')
  parser.add_argument('--px-sigma', type=float, default=0.3, help='')
  parser.add_argument('--n-eigen-degrees', type=int, default=1000, help='')

  parser.add_argument('--skip-downstream-model', action='store_true', default=False, help='')
  parser.add_argument('--order-hermite', type=int, default=20, help='')

  ar = parser.parse_args()

  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar


def preprocess_args(ar):
  if ar.log_dir is None:
    assert ar.log_name is not None
    ar.log_dir = ar.base_log_dir + ar.log_name + '/'
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)

  if ar.seed is None:
    ar.seed = np.random.randint(0, 1000)
  assert ar.data in {'digits', 'fashion', '2d', '1d'}
  if ar.rff_sigma is None:
    ar.rff_sigma = '105' if ar.data == 'digits' else '127'

  if ar.loss_type == 'kmeans' and ar.tgt_epsilon > 0.0:
    assert ar.center_data, 'dp kmeans requires centering of data'

  if ar.data in {'2d', '1d'}:
    ar.conv_gen = False
  else:
    ar.conv_gen = True
    
def synthesize_data_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  if n_data % gen_batch_size != 0:
    assert n_data % 100 == 0
    gen_batch_size = n_data // 100
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()

  
  
def test_results(data_key, log_name, log_dir, data_tuple, eval_func, skip_downstream_model):
  if data_key in {'digits', 'fashion'}:
    if not skip_downstream_model:
      final_score = test_gen_data(log_name, data_key, subsample=0.1, custom_keys='logistic_reg')
      log_final_score(log_dir, final_score)
  elif data_key == '2d':
    if not skip_downstream_model:
      final_score = test_passed_gen_data(log_name, data_tuple, log_save_dir=None, log_results=False,
                                         subsample=.1, custom_keys='mlp', compute_real_to_real=True)
      log_final_score(log_dir, final_score)
    eval_score = eval_func(data_tuple.x_gen, data_tuple.y_gen.flatten())
    print(f'Score of evaluation function: {eval_score}')
    with open(os.path.join(log_dir, 'eval_score'), 'w') as f:
      f.writelines([f'{eval_score}'])

    plot_data(data_tuple.x_real_train, data_tuple.y_real_train.flatten(), os.path.join(log_dir, 'plot_train'),
              center_frame=True)
    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_sub0.2'), subsample=0.2)
    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_centered'),
              center_frame=True)

    plot_data_1d(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_norms_hist'))
  elif data_key == '1d':
    plot_data_1d(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
    plot_data_1d(data_tuple.x_real_test, data_tuple.y_real_test.flatten(), os.path.join(log_dir, 'plot_data'))




def main():
    print('Ent')
    # load settings
    ar = get_args()
    pt.manual_seed(ar.seed)
    use_cuda = pt.cuda.is_available()
    device = pt.device("cuda" if use_cuda else "cpu")
    
      # load data
    data_pkg = get_dataloaders(ar.data, ar.batch_size, ar.test_batch_size, use_cuda, ar.center_data,
                               ar.synth_spec_string, ar.test_split)
    
    
    # init model
    if ar.conv_gen:
      gen = ConvCondGen(ar.d_code, ar.gen_spec, data_pkg.n_labels, ar.n_channels, ar.kernel_sizes).to(device)
    else:
      use_sigmoid = ar.data in {'digits', 'fashion'}
      gen = FCCondGen(ar.d_code, ar.gen_spec, data_pkg.n_features, data_pkg.n_labels, use_sigmoid=use_sigmoid,
                      batch_norm=True).to(device)
    
      gen.load_state_dict(pt.load(ar.log_dir + 'gen.pt'))
      
      data_id     =   "gen_data"
      syn_data, syn_labels = synthesize_data_with_uniform_labels(gen, device, gen_batch_size=ar.gen_batch_size,
                                                                 n_data=data_pkg.n_data,
                                                                 n_labels=data_pkg.n_labels)
    
      data_tuple = datasets_colletion_def(syn_data, syn_labels,
                                          data_pkg.train_data.data, data_pkg.train_data.targets,
                                          data_pkg.test_data.data, data_pkg.test_data.targets)
      
      test_results(ar.data, ar.log_name, ar.log_dir, data_tuple, data_pkg.eval_func, ar.skip_downstream_model)
    
if __name__ == '__main__':
  main()
