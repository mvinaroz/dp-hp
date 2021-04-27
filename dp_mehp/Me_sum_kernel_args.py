import torch
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from all_aux_files import FCCondGen, ConvCondGen, find_rho, find_order, ME_with_HP, get_mnist_dataloaders
from all_aux_files import get_dataloaders, log_args, datasets_colletion_def, test_results_subsampling_rate
from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data, flatten_features, log_gen_data
from all_aux_files import heuristic_for_length_scale
from all_aux_files_tab_data import ME_with_HP_tab
#from autodp import privacy_calibrator
from collections import namedtuple
from torch.autograd import grad
import math
from math import factorial
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.autograd import grad
from autodp import privacy_calibrator

train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits, fashion and 2d')
  parser.add_argument('--create-dataset', action='store_true', default=True, help='if true, make 60k synthetic code_balanced')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=1000)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--gen-batch-size', '-gbs', type=int, default=1000)
  parser.add_argument('--embed-batch-size', '-gbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=100)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')
  
  # MODEL DEFINITION
  parser.add_argument('--model_name', type=str, default='CNN', help='you can use CNN of FC')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')
  parser.add_argument('--gen-spec', type=str, default='500,500', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')

  # ALTERNATE MODES
  parser.add_argument('--single-release', action='store_true', default=True, help='get 1 data mean embedding only')
  parser.add_argument('--report_intermediate', action='store_true', default=False, help='')
  parser.add_argument('--loss-type', type=str, default='MEHP', help='how to approx mmd')
  parser.add_argument('--method', type=str, default='sum_kernel', help='')
  parser.add_argument('--sampling_rate_synth', type=float, default=0.1,  help='')
  parser.add_argument('--skip-downstream-model', action='store_false', default=False, help='')
  parser.add_argument('--order-hermite', type=int, default=100, help='')
  parser.add_argument('--heuristic-sigma', action='store_true', default=False)
  parser.add_argument("--separate-kernel-length", action='store_true', default=False) # heuristic-sigma has to be "True", to enable separate-kernel-length
  parser.add_argument('--kernel-length', type=float, default=0.001, help='')

  # DP SPEC
  parser.add_argument('--is-private', default=False, help='produces a DP mean embedding of data')
  parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
  parser.add_argument('--delta', type=float, default=1e-5, help='delta in (epsilon, delta)-DP')
  
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

    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)
        assert ar.data in {'digits', 'fashion'}

    
def main():
    """Load settings"""
    ar = get_args()
    print(ar)
    torch.manual_seed(ar.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    """Load data"""
    data_pkg = get_dataloaders(ar.data, ar.batch_size, ar.test_batch_size, use_cuda=device,
                               normalize=False, synth_spec_string=None, test_split=None)

    """ Define a generator """
    if ar.model_name == 'FC':
        model = FCCondGen(ar.d_code, ar.gen_spec, data_pkg.n_features, data_pkg.n_labels, use_sigmoid=True, batch_norm=True, use_clamp=False).to(device)
    elif ar.model_name == 'CNN':
        model = ConvCondGen(ar.d_code, ar.gen_spec, data_pkg.n_labels, ar.n_channels, ar.kernel_sizes,
                            use_sigmoid=True, batch_norm=True).to(device)

    """ set the scale length """
    #num_iter = np.int(data_pkg.n_data / ar.batch_size)

    if ar.heuristic_sigma:

        print('we use the median heuristic for length scale')
        sigma = heuristic_for_length_scale(data_pkg.train_loader, data_pkg.n_features, ar.batch_size, data_pkg.n_data, device)
        if ar.separate_kernel_length:
            print('we use a separate length scale on each coordinate of the data')
            sigma2=np.mean(sigma**2, axis=0)
            sigma2[sigma2==0]=1e-6

        else:
            sigma2 = np.median(sigma**2)

    else:
        sigma2 = ar.kernel_length


    print('sigma2 is', sigma2)
    rho = find_rho(sigma2, ar.separate_kernel_length)
    order = ar.order_hermite

    # ev_thr = 1e-6  # eigen value threshold, below this, we wont consider for approximation
    # order = find_order(rho, ev_thr)
    # or_thr = ar.order_hermite
    # if order>or_thr:
    #     order = or_thr
    #     print('chosen order is', order)

    print('computing mean embedding of data')
    # old_data_embedding = torch.zeros(data_pkg.n_features*(order+1), data_pkg.n_labels, num_iter, device=device)
    #
    # for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
    #     # print(batch_idx)
    #     data, labels = data.to(device), labels.to(device)
    #     data = flatten_features(data)
    #     for idx in range(data_pkg.n_labels):
    #         idx_data = data[labels == idx]
    #         phi_data = ME_with_HP(idx_data, order, rho, device, data_pkg.n_data)
    #         old_data_embedding[:, idx, batch_idx] = phi_data
    # old_data_embedding = torch.sum(old_data_embedding, axis=2)
    embedding_train_loader, _ = get_mnist_dataloaders(ar.embed_batch_size, ar.test_batch_size,
                                                      use_cuda=device, dataset=ar.data)

    # summing at the end uses unnecessary memory - leaving previous version in in case of errors with this one
    data_embedding = torch.zeros(data_pkg.n_features * (order + 1), data_pkg.n_labels, device=device)
    for batch_idx, (data, labels) in enumerate(embedding_train_loader):
        data, labels = flatten_features(data.to(device)), labels.to(device)
        for idx in range(data_pkg.n_labels):
            idx_data = data[labels == idx]
            if ar.separate_kernel_length:
                phi_data = ME_with_HP_tab(idx_data, order, rho, device, data_pkg.n_data )
            else:
                phi_data = ME_with_HP(idx_data, order, rho, device, data_pkg.n_data )
            data_embedding[:, idx] += phi_data

    # print('DEBUG embedding diff:', torch.norm(old_data_embedding - data_embedding))  # it's around 2e-6, so almost 0
    print('done with computing mean embedding of data')

    if ar.is_private:
        k = 1
        epsilon = ar.epsilon
        delta = ar.delta
        privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
        print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])
        # print('we add noise to the data mean embedding as the private flag is true')
        std = (2 * privacy_param['sigma'] * np.sqrt(data_pkg.n_features) / data_pkg.n_data)
        noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

        print('before perturbation, mean and variance of data mean embedding are %f and %f ' %(torch.mean(data_embedding), torch.std(data_embedding)))
        data_embedding = data_embedding + noise
        print('after perturbation, mean and variance of data mean embedding are %f and %f ' % (torch.mean(data_embedding), torch.std(data_embedding)))
    else:
        print('we do not add noise to the mean embedding as is_private is set to False.')

      
    """ Training """
    optimizer = torch.optim.Adam(list(model.parameters()), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    score_mat = np.zeros(ar.epochs)

    print('start training the generator')
    for epoch in range(1, ar.epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            gen_code, gen_labels = model.get_code(ar.batch_size, device)
            gen_samples = model(gen_code)  # batch_size by 784

            if ar.single_release:
                synth_data_embedding = torch.zeros((data_pkg.n_features * (order+1), data_pkg.n_labels), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(data_pkg.n_labels):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    if ar.separate_kernel_length:
                        synth_data_embedding[:, idx] = ME_with_HP_tab(idx_synth_data, order, rho, device, ar.batch_size)
                    else:
                        synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, ar.batch_size)
                    
            else:
                synth_data_embedding = torch.zeros((data_pkg.n_features * (order+1), data_pkg.n_labels), device=device)
                data_embedding = torch.zeros((data_pkg.n_features * (order+1), data_pkg.n_labels), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(data_pkg.n_labels):
                    idx_data = data[labels == idx]
                    if ar.separate_kernel_length:
                        synth_data_embedding[:, idx] = ME_with_HP_tab(idx_synth_data, order, rho, device, ar.batch_size)
                    else:
                        synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, ar.batch_size)
                    
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, ar.batch_size)
                    
            loss = torch.sum((data_embedding - synth_data_embedding)**2)
            
            optimizer.zero_grad()
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
        # end for
        
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), data_pkg.n_data, loss.item()))

        log_gen_data(model, device, epoch, data_pkg.n_labels, ar.log_dir)
        scheduler.step()

    #     end if
    # end for

    if ar.report_intermediate:
        max_score = np.max(score_mat)

        dir_max_score = ar.log_dir + ar.data + '/max_score'
        np.save(dir_max_score, max_score)
        print('max score among the training runs is', max_score)

    #########################################################################
    """ Once we have a trained generator, we store synthetic data from it and test them on logistic regression """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=ar.batch_size,
                                                               n_data=data_pkg.n_data,
                                                               n_labels=data_pkg.n_labels)

    dir_syn_data = ar.log_dir + ar.data + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
#    final_score = test_gen_data(ar.log_name + '/' +  ar.data, ar.data, subsample=ar.sampling_rate_synth, custom_keys='logistic_reg')
#    data_tuple = datasets_colletion_def(syn_data, syn_labels,
#                                        data_pkg.train_data.data, data_pkg.train_data.targets,
#                                        data_pkg.test_data.data, data_pkg.test_data.targets)
    test_results_subsampling_rate(ar.data, ar.log_name + '/' + ar.data, ar.log_dir, ar.skip_downstream_model, ar.sampling_rate_synth)
    
    
#    dir_score = ar.log_dir + '/' + ar.data + '/score_60k'
#    np.save(dir_score, final_score)
#    print('score with 60k samples is', final_score)

  
if __name__ == '__main__':
  main()
