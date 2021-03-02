# We train a generator by minimizing the sum of pixel-wise MMD

# from __future__ import print_function
import torch
import numpy as np
import os
# from torch.optim.lr_scheduler import StepLR
from all_aux_files import FCCondGen, ConvCondGen, meddistance, flatten_features, get_mnist_dataloaders, log_gen_data
from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data
# test_passed_gen_data, datasets_colletion_def, test_results
from mmd_sum_kernel import mmd_loss
from all_aux_files import eval_hermite_pytorch
from collections import namedtuple

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

def find_order(rho,eigen_val_threshold):
    k = 100
    eigen_vals = (1 - rho) * (rho ** np.arange(0, k + 1))
    idx_keep = eigen_vals > eigen_val_threshold
    keep_eigen_vals = eigen_vals[idx_keep]
    print('keep_eigen_vals are ', keep_eigen_vals)
    order = len(keep_eigen_vals)
    print('The number of orders for Hermite Polynomials is', order)
    return order

def find_rho(sigma2):
    alpha = 1 / (2.0 * sigma2)
    # from this: alpha = rho / (1- rho**2), identify what rho is
    # sol = optimize.minimize_scalar(lambda r: (alpha - r / (1 - r ** 2)) ** 2, bounds=(0, 1),
    #                                method='bounded')
    # rho = sol.x
    # print('rho from optimization', rho)
    rho = -1 / 2 / alpha + np.sqrt(1 / alpha ** 2 + 4) / 2
    # print('rho from closed form solution', rho)
    return rho


def ME_with_HP(x, order, rho, device):
    n_data, input_dim = x.shape
    phi_x_mat = []
    for axis in torch.arange(input_dim):
        # print(axis)
        x_axis = x[:, axis]
        x_axis = x_axis[:, None]
        phi_x_axis, eigen_vals_axis = feature_map_HP(order, x_axis, torch.tensor(rho), device)
        me = torch.mean(phi_x_axis,axis=0)
        phi_x_mat.append(me[:,None])

    return phi_x_mat


def feature_map_HP(k, x, rho, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    print('device', device)
    eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1))
    print('eigen_vals', eigen_vals)
    eigen_funcs_x = eigen_func(k, rho, x, device)  # output dim: number of datapoints by number of degree
    print('eigen_funcs', eigen_funcs_x)

    phi_x = torch.einsum('ij,j-> ij', eigen_funcs_x, torch.sqrt(eigen_vals))  # number of datapoints by order
    # n_data = eigen_funcs_x.shape[0]
    # mean_phi_x = torch.sum(phi_x,0)/n_data

    return phi_x, eigen_vals


def eigen_func(k, rho, x, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at, size: number of data points by input_dimension
    print('device', device)
    orders = torch.arange(0, k + 1, device=device)
    H_k = eval_hermite_pytorch(x, k + 1, device, return_only_last_term=False)
    H_k = H_k[:, :, 0]
    # H_k = eval_hermite(orders, x)  # input arguments: degree, where to evaluate at.
    # output dim: number of datapoints by number of degree
    rho = torch.tensor(rho, dtype=float, device=device)
    # print('Device of Rho is ', rho.device, 'and device of x is ', x.device)
    exp_trm = torch.exp(-rho / (1 + rho) * (x ** 2))  # output dim: number of datapoints by 1
    N_k = (2 ** orders) * ((orders + 1).to(torch.float).lgamma().exp()) * torch.sqrt(((1 - rho) / (1 + rho)))
    eigen_funcs = 1 / torch.sqrt(N_k) * (H_k * exp_trm)  # output dim: number of datapoints by number of degree
    return eigen_funcs


def main():

    data_name = 'digits' # 'digits' or 'fashion'
    method = 'sum_kernel' # sum_kernel or a_Gaussian_kernel
    """ this code only works for MMDest, for ME with HP, use ME_sum_kernel.py"""
    loss_type = 'MMDest' # ME (mean-embedding with HP) or MMDest (MMD estimate)
    single_release = True
    model_name = 'CNN' # CNN or FC
    report_intermidiate_result = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Load data to test """
    batch_size = 100
    test_batch_size = 200
    data_pkg = get_dataloaders(data_name, batch_size, test_batch_size, True, False, [], [])
    train_loader = data_pkg.train_loader
    # test_loader = data_pkg.test_loader
    n_train_data = data_pkg.n_data

    """ Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    n_epochs = 20
    if model_name == 'FC':
        model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=False, batch_norm=True).to(device)
        # for MNIST, with FC, logistic accuracy 81 percent (20 epochs) for full MMD
        # for Fashion-MNIST, with FC, logistic accuracy 0.687 (20 epochs) for full MMD
    elif model_name == 'CNN':
        model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
        # for MNIST, with CNN, logistic accuracy 0.78 (20 epochs) for full MMD
        # for Fashion-MNIST, with CNN, logistic accuracy 0.689 (20 epochs) for full MMD

    """ Training """
    # set the scale length
    num_iter = n_train_data/batch_size
    if method=='sum_kernel':
        if data_name=='digits':
            sigma2_arr = 0.05
        else:
            sigma2_arr = np.zeros((np.int(num_iter), feature_dim))
            for batch_idx, (data, labels) in enumerate(train_loader):
                # print('batch idx', batch_idx)
                data, labels = data.to(device), labels.to(device)
                data = flatten_features(data) # minibatch by feature_dim
                data_numpy = data.detach().cpu().numpy()
                for dim in np.arange(0,feature_dim):
                    med = meddistance(np.expand_dims(data_numpy[:,dim],axis=1))
                    sigma2 = med ** 2
                    sigma2_arr[batch_idx, dim] = sigma2

    elif method=='a_Gaussian_kernel':
        sigma2_arr = np.zeros(np.int(num_iter))
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)  # minibatch by feature_dim
            data_numpy = data.detach().cpu().numpy()
            med = meddistance(data_numpy)
            sigma2 = med ** 2
            sigma2_arr[batch_idx] = sigma2

    sigma2 = torch.tensor(np.mean(sigma2_arr))
    print('length scale', sigma2)


    base_dir = 'logs/gen/'
    log_dir = base_dir + data_name + method + model_name + '/'
    log_dir2 = data_name + method + model_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    if loss_type=='ME':
        rho = find_rho(sigma2)
        ev_thr = 1e-3  # eigen value threshold, below this, we wont consider for approximation
        order = find_order(rho, ev_thr)
        if single_release:
            data_embedding = torch.zeros((feature_dim*order, n_classes))
            for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
                data, labels = data.to(device), labels.to(device)
                data = flatten_features(data)
                for idx in range(n_classes):
                    idx_data = data[labels == idx]
                    data_embedding[:,idx] = 0.5*(data_embedding[:,idx] + ME_with_HP(idx_data,order,rho,device))

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            optimizer.zero_grad()
            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            if loss_type == 'MMDest':
                loss = mmd_loss(data, labels, gen_samples, gen_labels, n_classes, sigma2, method)
            elif loss_type == 'ME':
                if single_release:
                    synth_data_embedding = torch.zeros((feature_dim * order, n_classes))
                    _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                    for idx in range(n_classes):
                        idx_synth_data = gen_samples[gen_labels_numerical == idx]
                        synth_data_embedding[:, idx] = 0.5 * (synth_data_embedding[:, idx] + ME_with_HP(idx_synth_data, order, rho, device))
                else:
                    synth_data_embedding = torch.zeros((feature_dim * order, n_classes))
                    data_embedding = torch.zeros((feature_dim * order, n_classes))
                    _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                    for idx in range(n_classes):
                        idx_data = data[labels == idx]
                        data_embedding[:, idx] = 0.5 * (data_embedding[:, idx] + ME_with_HP(idx_data, order, rho, device))
                        idx_synth_data = gen_samples[gen_labels_numerical == idx]
                        synth_data_embedding[:, idx] = 0.5 * (synth_data_embedding[:, idx] + ME_with_HP(idx_synth_data, order, rho, device))

                loss = torch.mean((data_embedding - synth_data_embedding) ** 2)

            loss.backward()
            optimizer.step()
        # end for

        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_train_data, loss.item()))

        log_gen_data(model, device, epoch, data_pkg.n_labels, log_dir)
        # scheduler.step()

        if report_intermidiate_result:
            """ now we save synthetic data and test them on logistic regression """
            syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                                       n_data=data_pkg.n_data,
                                                                       n_labels=data_pkg.n_labels)

            dir_syn_data = log_dir + data_name + '/synthetic_mnist'
            if not os.path.exists(dir_syn_data):
                os.makedirs(dir_syn_data)

            np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
            final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=0.1, custom_keys='logistic_reg')
            print('on logistic regression, accuracy is', final_score)

        # end if
    # end for

    #########################################################################3
    """ Once we have a trained generator, we store synthetic data from it and test them on logistic regression """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=data_pkg.n_data,
                                                               n_labels=data_pkg.n_labels)

    dir_syn_data = log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
    final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=0.1, custom_keys='logistic_reg')
    print('on logistic regression, accuracy is', final_score)


if __name__ == '__main__':
    main()
