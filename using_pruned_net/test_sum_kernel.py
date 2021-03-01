# We train a generator by minimizing the sum of pixel-wise MMD

from __future__ import print_function
import torch
from test_transfer_learning import load_trained_model, data_loader
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append('/is/ei/mpark/DPDR/code_balanced')
from models_gen import FCCondGen, ConvCondGen
from aux import meddistance
# from full_mmd import mmd_loss
from mmd_sum_kernel import mmd_loss
from aux import flatten_features
import torch.optim as optim
import numpy as np
from data_loading import get_dataloaders
import os
from gen_balanced import log_gen_data, test_results
from gen_balanced import synthesize_data_with_uniform_labels
from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def

def main():

    data_name = 'digits' # 'digits' or 'fashion'
    method = 'sum_kernel' # sum_kernel or a_Gaussian_kernel
    model_name = 'FC' # CNN or FC
    report_intermidiate_result = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Load data to test """
    batch_size = 100
    test_batch_size = 200
    use_cuda = torch.cuda.is_available()
    data_pkg = get_dataloaders(data_name, batch_size, test_batch_size, use_cuda, False, [], [])

    """ Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    n_epochs = 20
    if model_name == 'FC':
        model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=False, batch_norm=True).to(device)
        # with FC, logistic accuracy 81 percent (20 epochs) for full MMD
    elif model_name == 'CNN':
        model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
        # with CNN, logistic accuracy 73 percent (20 epochs) for full MMD

    """ Training """
    # set the scale length
    num_iter = data_pkg.n_data/batch_size
    if method=='sum_kernel':
        sigma2_arr = np.zeros((np.int(num_iter), feature_dim))
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data) # minibatch by feature_dim
            data_numpy = data.detach().cpu().numpy()
            for dim in np.arange(0,feature_dim):
                med = meddistance(np.expand_dims(data_numpy[:,dim],axis=1))
                sigma2 = med ** 2
                sigma2_arr[batch_idx, dim] = sigma2

        # avg_sigma2 = np.mean(sigma2_arr, axis=0)
        # print('avg sigma2 is', avg_sigma2)
        # print(avg_sigma2.shape)

    elif method=='a_Gaussian_kernel':
        sigma2_arr = np.zeros(np.int(num_iter))
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
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
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            data, labels = data.to(device), labels.to(device)
            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            data = flatten_features(data)
            loss = mmd_loss(data, labels, gen_samples, gen_labels, n_classes, sigma2, method)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # end for

        n_data = len(data_pkg.train_loader.dataset)
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))

        log_gen_data(model, device, epoch, data_pkg.n_labels, log_dir)
        scheduler.step()

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
