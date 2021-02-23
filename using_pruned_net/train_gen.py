# We train a generator by minimizing the MMD computed on the features of ResNet.
# (1) load a trained model to use it as a feature extractor
# (2) train a generator by reducing MMD using extracted features of the generated samples and real samples

from __future__ import print_function
import torch
from test_transfer_learning import load_trained_model, data_loader
from torch.optim.lr_scheduler import StepLR
import sys
# sys.path.append('/home/mpark/DPDR/code_balanced')
sys.path.append('/is/ei/mpark/DPDR/code_balanced')
from models_gen import FCCondGen
from aux import meddistance
from full_mmd import mmd_loss
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
    model_name = 'ResNet'
    transfer = True
    reg_L2norm = 0.005 # regularization constant for L2 norm constraint
    print('%s dataset with regularization constant %f' % (data_name, reg_L2norm))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ 1. Load the trained model and its architecture """
    model2load = 'Trained_ResNet'
    features = load_trained_model(model2load, model_name, device)

    """ 2. Load data to test """
    batch_size = 100
    test_batch_size = 200
    use_cuda = torch.cuda.is_available()
    # train_loader, test_loader = data_loader(batch_size, model_name, data_name)
    data_pkg = get_dataloaders(data_name, batch_size, test_batch_size, use_cuda, False, [], [])

    """ 3. Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    n_epochs = 50
    model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=True, batch_norm=True).to(device)

    """ 4. Training """
    # set the scale length
    num_iter = data_pkg.n_data/batch_size
    sigma2_arr = np.zeros(np.int(num_iter))
    for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
        data, labels = data.to(device), labels.to(device)
        data = flatten_features(data)
        med = meddistance(data.detach().cpu().numpy())
        sigma2 = med**2
        sigma2_arr[batch_idx] = sigma2
        # print(sigma2)

    # log_dir = 'logs/gen'+str(reg_L2norm)+'/'
    base_dir = 'logs/gen/'
    log_dir = base_dir + data_name + str(reg_L2norm) + '/'
    log_dir2 = data_name + str(reg_L2norm) + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sigma2 = torch.tensor(np.mean(sigma2_arr))
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    # print(list(model.parameters()))
    # print('parameters before training')
    # print(list(features.parameters()))

    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
            data, labels = data.to(device), labels.to(device)
            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code)

            if transfer:
                # print(gen_samples.shape)
                # print(data.shape)
                gen_samples = torch.reshape(gen_samples, (batch_size, 1, 28, 28))
                # print(gen_samples.shape) # minibatch by 784
                _, data_feat = features(data.repeat(1,3,1,1))
                _, syn_data_feat = features(gen_samples.repeat(1,3,1,1))
                # print(data_feat.shape)
                # print(syn_data_feat.shape)
                L2norm = torch.mean(torch.sum(gen_samples**2,1).sqrt()) # squared sum across features, then average over datapoints
                # print(L2norm)
                loss = mmd_loss(data_feat, labels, syn_data_feat, gen_labels, n_classes, sigma2) + reg_L2norm*L2norm
                # print(loss)
            else:
                data = flatten_features(data)
                loss = mmd_loss(data, labels, gen_samples, gen_labels, n_classes, sigma2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(list(model.parameters()))
            # print('parameters after training')
            # print(list(features.parameters()))
            #
            # if batch_idx==0:
            #     break

        n_data = len(data_pkg.train_loader.dataset)
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))

        log_gen_data(model, device, epoch, data_pkg.n_labels, log_dir)
        scheduler.step()


    # evaluating synthetic data on a classifier
    # if data_name == 'mnist':
    #     data_name = 'digits'
    # elif data_name == 'fmnist':
    #     data_name = 'fashion'

    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=data_pkg.n_data,
                                                               n_labels=data_pkg.n_labels)

    dir_syn_data = log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
    # data_tuple = datasets_colletion_def(syn_data, syn_labels,
    #                                     data_pkg.train_data.data, data_pkg.train_data.targets,
    #                                     data_pkg.test_data.data, data_pkg.test_data.targets)
    #
    # # test_results(data_name, data_name, 'log_dir', data_tuple, data_pkg.eval_func, 'logistic_reg')
    final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=1.0, custom_keys='logistic_reg')
    print('on logistic regression, accuracy is', final_score)

if __name__ == '__main__':
    main()
