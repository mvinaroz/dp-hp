import torch
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from all_aux_files import FCCondGen, ConvCondGen, flatten_features, meddistance
from all_aux_files import eval_hermite_pytorch, get_mnist_dataloaders
from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data, flatten_features, log_gen_data
from collections import namedtuple
from autodp import privacy_calibrator
import math
from math import factorial
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])

def get_dataloaders(dataset_key, batch_size, test_batch_size, use_cuda, normalize, synth_spec_string, test_split):
  if dataset_key in {'digits', 'fashion'}:
    train_loader, test_loader, trn_data, tst_data = get_mnist_dataloaders(batch_size, test_batch_size, use_cuda,
                                                                          dataset=dataset_key, normalize=normalize,
                                                                          return_datasets=True)
    n_features = 784
    n_data = 60_000
    n_labels = 10
    eval_func = None
  else:
    raise ValueError

  return train_data_tuple_def(train_loader, test_loader, trn_data, tst_data, n_features, n_data, n_labels, eval_func)

def load_data(data_name, batch_size):
    transform_digits = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    transform_fashion = transforms.Compose([transforms.ToTensor()])

    if data_name == 'digits':
        train_dataset = datasets.MNIST(root='data', train=True, transform=transform_digits, download=True)
    elif data_name == 'fashion':
        train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform_fashion, download=True)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=0,shuffle=True)
    return train_loader

def find_rho(sigma2):
    alpha = 1 / (2.0 * sigma2)
    rho = -1 / 2 / alpha + np.sqrt(1 / alpha ** 2 + 4) / 2
    rho_1 = -1 / 2 / alpha - np.sqrt(1 / alpha ** 2 + 4) / 2

    if rho<1: # rho is always non-negative
        print('rho is less than 1. so we take this value.')
    elif rho>1:
        print('rho is larger than 1. Mehler formula does not hold')
        if rho_1>-1: # rho_1 is always negative
            print('rho_1 is larger than -1. so we take this value.')
            rho = rho_1
        else: # if rho_1 <-1,
            print('rho_1 is smaller than -1. Mehler formula does not hold')
            sys.exit('no rho values satisfy the Mehler formulas. We have to stop the run')

    return rho

def find_order(rho,eigen_val_threshold):
    k = 100
    eigen_vals = (1 - rho) * (rho ** np.arange(0, k + 1))
    idx_keep = eigen_vals > eigen_val_threshold
    keep_eigen_vals = eigen_vals[idx_keep]
    print('keep_eigen_vals are ', keep_eigen_vals)
    order = len(keep_eigen_vals)
    print('The number of orders for Hermite Polynomials is', order)
    return order


def feature_map_HP(k, x, rho, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    # print('device', device)
    eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1))
    eigen_vals = eigen_vals.to(device)
    # print('eigen_vals', eigen_vals)
    eigen_funcs_x = eigen_func(k, rho, x, device)  # output dim: number of datapoints by number of degree
    # print('eigen_funcs', eigen_funcs_x)
    sqrt_eig_vals = torch.sqrt(torch.abs(eigen_vals))

    if (sqrt_eig_vals!=sqrt_eig_vals).any():
        print('sqrt eig values are nan', sqrt_eig_vals)
    phi_x = torch.einsum('ij,j-> ij', eigen_funcs_x, sqrt_eig_vals)  # number of datapoints by order
    # n_data = eigen_funcs_x.shape[0]
    # mean_phi_x = torch.sum(phi_x,0)/n_data

    if (phi_x != phi_x).any():
      print('phi_x has nan', phi_x)

    return phi_x, eigen_vals

def ME_with_HP(x, order, rho, device, n_training_data):
    n_data, input_dim = x.shape

    # reshape x, such that x is a long vector
    x_flattened = x.view(-1)
    x_flattened = x_flattened[:,None]
    phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, rho, device)
    # if (phi_x_axis_flattened != phi_x_axis_flattened).any():
    #     print('phi_x_axis_flattened inside ME_with_HP is nan', phi_x_axis_flattened)
    phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order+1)
    # if (phi_x != phi_x).any():
    #     print('reshaped phi_x_axis_flattened inside ME_with_HP is nan', phi_x)
    # phi_x = torch.mean(phi_x, axis=0)
    # phi_x2 = torch.tensor(phi_x, dtype=float)
    phi_x = phi_x.type(torch.float)
    sum_val = torch.sum(phi_x, axis=0)
    phi_x = sum_val / n_training_data
    # # if (sum_val !=sum_val).any():
    # #     print('sum val is nan', sum_val)
    #
    # if n_data ==0:
    #     phi_x = sum_val/(1+n_data)
    #     print('n_data is zero, so we are adding 1 to n_data to avoid nan')
    # else:
    #     phi_x = sum_val/n_data

    # if (phi_x != phi_x).any():
    #     print('mean embedding inside ME_with_HP is nan', phi_x)
    #     print('n_data is ', n_data)
    phi_x = phi_x.view(-1) # size: input_dim*(order+1)

    # phi_x_mat = torch.zeros(input_dim*(order+1))
    # for axis in torch.arange(input_dim):
    #     # print(axis)
    #     x_axis = x[:, axis]
    #     x_axis = x_axis[:, None]
    #     phi_x_axis, eigen_vals_axis = feature_map_HP(order, x_axis, torch.tensor(rho), device)
    #     me = torch.mean(phi_x_axis,axis=0)
    #     phi_x_mat[axis*(order+1):(axis+1)*(order+1)] = me
    #     # phi_x_mat.append(me[:,None])
    # phi_x_mat = phi_x_mat[:,None]
    # phi_x_mat = phi_x_mat.to(device)

    return phi_x

def eigen_func(k, rho, x, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at, size: number of data points by input_dimension
    # print('device', device)
    # orders = torch.arange(0, k + 1, device=device, dtype=torch.float32)
    orders = torch.arange(0, k + 1, device=device)
    H_k = eval_hermite_pytorch(x, k + 1, device, return_only_last_term=False)
    H_k = H_k[:, :, 0]

    if torch.isnan(H_k).any():
        print('H_k has nan')
    # H_k = eval_hermite(orders, x)  # input arguments: degree, where to evaluate at.
    # output dim: number of datapoints by number of degree
    rho = torch.tensor(rho, dtype=torch.float32, device=device)
    # print('Device of Rho is ', rho.device, 'and device of x is ', x.device)
    exp_trm = torch.exp(-rho / (1 + rho) * (x ** 2))  # output dim: number of datapoints by 1
    N_k = (2 ** orders) * ((orders + 1).to(torch.float32).lgamma().exp()) * torch.sqrt(torch.abs((1 - rho) / (1 + rho)))


    # N_k1 = 2**torch.arange(0, k + 1, device=device, dtype=torch.float32)
    #
    # ord = np.arange(0, k + 1)
    # N_k2 = [factorial(num) for num in ord]
    # N_k2 = np.array(N_k2, dtype=np.int)
    # N_k2 = torch.tensor(N_k2, dtype=torch.float, device=device)
    #
    # N_k3 = torch.sqrt(torch.abs((1 - rho) / (1 + rho)))
    # N_k = N_k1*N_k2*N_k3


    if torch.isnan(1 / torch.sqrt(N_k)).any():
        print('oops, 1/N_k has nan')
    eigen_funcs = 1 / torch.sqrt(N_k) * (H_k * exp_trm)  # output dim: number of datapoints by number of degree
    if (eigen_funcs != eigen_funcs).any():
      print('eigen_funcs has nan', eigen_funcs)

    return eigen_funcs


def main():

    torch.manual_seed(0)
    data_name = 'digits' # 'digits' or 'fashion'
    method = 'sum_kernel' # sum_kernel or a_Gaussian_kernel
    loss_type = 'MEHP'
    single_release = True
    private = False # this flag can be true or false only when single_release is true.
    if private:
        epsilon=1.0
        delta = 1e-5
    model_name = 'FC' # CNN or FC
    report_intermidiate_result = True
    subsampling_rate_for_synthetic_data = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Load data to test """
    batch_size = 100
    # train_loader = load_data(data_name, batch_size)
    test_batch_size = 200
    data_pkg = get_dataloaders(data_name, batch_size, test_batch_size, True, False, [], [])
    train_loader = data_pkg.train_loader
    n_train_data = 60000

    """ Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    n_epochs = 20
    if model_name == 'FC':
        model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=False, batch_norm=True).to(device)
    elif model_name == 'CNN':
        model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)


    """ set the scale length """
    num_iter = np.int(n_train_data / batch_size)
    if data_name=='digits':
        sigma2_arr = 0.05
    elif data_name=='fashion':
        sigma2_arr = 0.07
    # # these are values we computed from the script below
    # sigma2_arr = np.zeros((np.int(num_iter), feature_dim))
    # for batch_idx, (data, labels) in enumerate(train_loader):
    #     # print('batch idx', batch_idx)
    #     data, labels = data.to(device), labels.to(device)
    #     data = flatten_features(data)  # minibatch by feature_dim
    #     data_numpy = data.detach().cpu().numpy()
    #     for dim in np.arange(0, feature_dim):
    #         med = meddistance(np.expand_dims(data_numpy[:, dim], axis=1))
    #         sigma2 = med ** 2
    #         sigma2_arr[batch_idx, dim] = sigma2

    sigma2 = np.mean(sigma2_arr)
    print('sigma2 is', sigma2)
    rho = find_rho(sigma2)
    ev_thr = 0.001  # eigen value threshold, below this, we wont consider for approximation
    order = find_order(rho, ev_thr)
    or_thr = 5
    if order>or_thr:
        order = or_thr
        print('chosen order is', order)
    if single_release:
        print('single release is', single_release)
        print('computing mean embedding of data')
        data_embedding = torch.zeros(feature_dim*(order+1), n_classes, num_iter, device=device)
        for batch_idx, (data, labels) in enumerate(train_loader):
            # print(batch_idx)
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)
            for idx in range(n_classes):
                idx_data = data[labels == idx]
                phi_data = ME_with_HP(idx_data, order, rho, device, n_train_data)
                data_embedding[:,idx, batch_idx] = phi_data
        data_embedding = torch.mean(data_embedding,axis=2)
        print('done with computing mean embedding of data')

        if private:
            print('we add noise to the data mean embedding as the private flag is true')
            k = 1 # how many compositions we do, here it's just once.
            privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
            privacy_param = privacy_param['sigma']
            print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param)
            std = (2 * privacy_param * np.sqrt(feature_dim) / n_train_data)
            noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

            print('before perturbation, mean and variance of data mean embedding are %f and %f ' %(torch.mean(data_embedding), torch.std(data_embedding)))

            data_embedding = data_embedding + noise

            print('after perturbation, mean and variance of data mean embedding are %f and %f ' % (
            torch.mean(data_embedding), torch.std(data_embedding)))

        else:
            print('we do not add noise to the data mean embedding as the private flag is false')


    """ name the directories """
    base_dir = 'logs/gen/'
    log_dir = base_dir + data_name + '_' + method + '_' + loss_type + '_' + model_name + '_' + 'single_release' + '_' +str(single_release) \
              + '_' + 'order_' + '_' +str(order) + '_' + 'private' + '_' + str(private) + '/'
    log_dir2 = data_name + '_' + method + '_' + loss_type + '_' + model_name + '_' + 'single_release' + '_' +str(single_release) \
              + '_' + 'order_' + '_' +str(order) + '_' + 'private' + '_' + str(private) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    """ Training """
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    score_mat = np.zeros(n_epochs)

    print('start training the generator')
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            if single_release:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, batch_size)
            else:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_data = data[labels == idx]
                    # if len(idx_data.shape)==0:
                    #     print('there is no real data samples for this class')
                    # if torch.isnan(idx_data).any():
                    #     print('true data has nan')
                    data_embedding[:, idx] = ME_with_HP(idx_data, order, rho, device, batch_size)
                    # if torch.isnan(data_embedding[:,idx]).any():
                    #     print('data mean embedding of selected datapoints has nan')

                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    # if len(idx_synth_data.shape)==0:
                    #     print('there is no generated samples for this class')
                    # if torch.isnan(idx_synth_data).any():
                    #     print('true data has nan')
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, batch_size)
                    # if torch.isnan(synth_data_embedding[:,idx]).any():
                    #     print('data mean embedding of generated datapoints has nan')

            loss = torch.norm(data_embedding - synth_data_embedding)**2
            # if torch.isnan(loss):
            #     print('loss is nan')
            #     print(loss)
            # print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_train_data, loss.item()))
            # print('syn_data_me', torch.mean(synth_data_embedding))
            # print('batch_idx:%s, loss:%f' %(batch_idx, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # end for

        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_train_data, loss.item()))

        log_gen_data(model, device, epoch, n_classes, log_dir)
        scheduler.step()

        if report_intermidiate_result:
            """ now we save synthetic data and test them on logistic regression """
            syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                                       n_data=n_train_data,
                                                                       n_labels=n_classes)

            dir_syn_data = log_dir + data_name + '/synthetic_mnist'
            if not os.path.exists(dir_syn_data):
                os.makedirs(dir_syn_data)

            np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
            final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=subsampling_rate_for_synthetic_data, custom_keys='logistic_reg')
            print('on logistic regression, accuracy is', final_score)
            score_mat[epoch - 1] = final_score

    #     end if
    # end for

    #########################################################################3
    """ Once we have a trained generator, we store synthetic data from it and test them on logistic regression """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=n_train_data,
                                                               n_labels=n_classes)

    dir_syn_data = log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
    final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=subsampling_rate_for_synthetic_data, custom_keys='logistic_reg')
    print('on logistic regression, accuracy is', final_score)

    max_score = np.max(score_mat)
    max_score = np.max([max_score, final_score])

    dir_max_score = log_dir + data_name + '/max_score'
    np.save(dir_max_score+'max_score', max_score)
    print('max score is', max_score)


if __name__ == '__main__':
    main()
