# We train a generator by minimizing the sum of pixel-wise MMD without HP approximations

# from __future__ import print_function
import torch
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from all_aux_files import FCCondGen, ConvCondGen, meddistance, flatten_features, get_mnist_dataloaders, log_gen_data
from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data
# test_passed_gen_data, datasets_colletion_def, test_results
from mmd_sum_kernel import mmd_loss
from collections import namedtuple
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

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

def main():

    torch.manual_seed(0)
    data_name = 'fashion' # 'digits' or 'fashion'
    # method = 'a_Gaussian_kernel' # sum_kernel or a_Gaussian_kernel
    method = 'sum_kernel'
    """ this code only works for MMDest, for ME with HP, use ME_sum_kernel.py"""
    loss_type = 'MMDest'
    if loss_type == 'MMDest':
        order = 'infty'
        single_release = False

    model_name = 'FC' # CNN or FC
    report_intermidiate_result = True
    subsampling_rate_for_synthetic_data = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    """ Load data to test """
    batch_size = 100
    test_batch_size = 200
    # train_loader = load_data(data_name, batch_size)
    data_pkg = get_dataloaders(data_name, batch_size, test_batch_size, True, False, [], [])
    train_loader = data_pkg.train_loader

    # train_loader = load_data(data_name, batch_size)
    n_train_data = 60000

    """ Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    n_epochs = 20
    if model_name == 'FC':
        model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=False, batch_norm=True).to(device)
        # for MNIST, with FC, logistic accuracy 81 percent (20 epochs) for full MMD
        # for Fashion-MNIST, with FC, logistic accuracy 0.687 (20 epochs) for full MMD
        # for MNIST, with FC, 0.778 (sum kernel)
        # for Fashion-MNIST, with FC,  0.727  (sum kernel)
    elif model_name == 'CNN':
        model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
        # for MNIST, with CNN, logistic accuracy 0.78 (20 epochs) for full MMD
        # for Fashion-MNIST, with CNN, logistic accuracy 0.689 (20 epochs) for full MMD
        # for MNIST, with CNN, 0.7649 (sum kernel)
        # for Fashion-MNIST, with CNN, 0.732 (sum kernel)

    """ Training """
    # set the scale length
    num_iter = n_train_data/batch_size
    if method=='sum_kernel':
        if data_name=='digits':
            sigma2_arr = 0.05
            # sigma2_arr = 0.1
        else:
            sigma2_arr = 0.07
            # sigma2_arr = np.zeros((np.int(num_iter), feature_dim))
            # for batch_idx, (data, labels) in enumerate(train_loader):
            #     # print('batch idx', batch_idx)
            #     data, labels = data.to(device), labels.to(device)
            #     data = flatten_features(data) # minibatch by feature_dim
            #     data_numpy = data.detach().cpu().numpy()
            #     for dim in np.arange(0,feature_dim):
            #         med = meddistance(np.expand_dims(data_numpy[:,dim],axis=1))
            #         sigma2 = med ** 2
            #         sigma2_arr[batch_idx, dim] = sigma2

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
    # log_dir = base_dir + data_name + method + model_name + '/'
    # log_dir2 = data_name + method + model_name + '/'
    log_dir = base_dir + data_name + '_' + method + '_' + loss_type + '_' + model_name + '_' + 'single_release' + '_' +str(single_release) \
              + '_' + 'order_threshold' + '_' +str(order) + '/'
    log_dir2 = data_name + '_' + method + '_' + loss_type + '_' + model_name + '_' + 'single_release' + '_' +str(single_release) \
              + '_' + 'order_threshold' + '_' +str(order) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    score_mat = np.zeros(n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            optimizer.zero_grad()
            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            loss = mmd_loss(data, labels, gen_samples, gen_labels, n_classes, sigma2, method)
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
            score_mat[epoch-1] = final_score

        # end if
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
