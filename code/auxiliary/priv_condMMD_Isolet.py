"""" test a simple generating training using MMD for relatively simple datasets """
""" for generating input features given random noise and the label for ISOLET data """
# Mijung wrote on Jan 09, 2020

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import socket

import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

# import autodp
from autodp import privacy_calibrator

import os

#Results_PATH = "/".join([os.getenv("HOME"), "condMMD/"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W).to(device)
    X = X.to(device)

    XWT = torch.mm(X, torch.t(W)).to(device)
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
    return Z

def Feature_labels(labels, weights):

    n_0 = torch.Tensor([weights[0]]).to(device)
    n_1 = torch.Tensor([weights[1]]).to(device)

    weighted_label_0 = 1/n_0*(labels[:,0]).to(device)
    weighted_label_1 = 1/n_1*(labels[:,1]).to(device)

    weighted_labels_feature = torch.cat((weighted_label_0[:,None], weighted_label_1[:,None]), 1).to(device)

    return weighted_labels_feature

class Generative_Model(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
            super(Generative_Model, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size
            # self.n_classes = n_classes

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)


        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)

            return output

#####################################################

# def main(n_features_arg, mini_batch_size_arg, how_many_epochs_arg):
def main():

    random.seed(0)

    """" parameters """
    #### parameters #####
    n_features_arg = 20000
    mini_batch_size_arg = 0.5
    how_many_epochs_arg = 1000
    is_private = True

    print("isolet dataset")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname():
        data_features_npy = np.load('../data/Isolet/isolet_data.npy')
        data_target_npy = np.load('../data/Isolet/isolet_labels.npy')
    else:
        # (1) load data
        data_features_npy = np.load('/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Isolet/isolet_data.npy')
        data_target_npy = np.load('/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR//data/Isolet/isolet_labels.npy')

    print(data_features_npy.shape)
    # dtype = [('Col1', 'int32'), ('Col2', 'float32'), ('Col3', 'float32')]
    values = data_features_npy
    index = ['Row' + str(i) for i in range(1, len(values) + 1)]

    values_l = data_target_npy
    index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

    data_features = pd.DataFrame(values, index=index)
    data_target = pd.DataFrame(values_l, index=index_l)

    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=0)

    # unpack data
    data_samps = X_train.values
    y_labels = y_train.values.ravel()

    ##########################################################

    # test logistic regression on the real data
    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(X_train, y_labels) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('ROC on real test data is', roc_auc_score(y_test, pred))
    print('PRC on real test data is', average_precision_score(y_test, pred))

    # ROC on real test data is 0.939003966752
    # PRC on real test data is 0.823399229853

    ################################################################

    n_classes = 2
    n, input_dim = data_samps.shape

    mini_batch_size = np.int(np.round(mini_batch_size_arg * n))
    print("total training datapoints: ", n)
    print("minibatch: ", mini_batch_size)

    how_many_iter = np.int(n / mini_batch_size)

    if is_private:
        # desired privacy level
        epsilon = 1.0
        delta = 1e-5
        privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta)
        print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

        sensitivity = 2/n
        noise_std_for_privacy = privacy_param['sigma']*sensitivity


    """ we use 10 datapoints to compute the median heuristic (then discard), and use the rest for training """
    idx_rp = np.random.permutation(n)
    num_data_pt_to_discard = 10
    idx_to_discard = idx_rp[0:num_data_pt_to_discard]
    idx_to_keep = idx_rp[num_data_pt_to_discard:]

    med = util.meddistance(data_samps[idx_to_discard, :])
    sigma2 = med**2
    print('length scale from median heuristic is', sigma2)

    data_samps = data_samps[idx_to_keep,:]
    n = idx_to_keep.shape[0]

    #######################################################

    true_labels = np.zeros((n, n_classes))
    idx_1 = y_labels[idx_to_keep] == 1
    idx_0 = y_labels[idx_to_keep] == 0
    true_labels[idx_1,1] = 1
    true_labels[idx_0,0] = 1


    # random Fourier features
    n_features = n_features_arg

    """ training a Generator via minimizing MMD """

    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 =  2* input_dim
    output_size = input_dim

    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    how_many_epochs = how_many_epochs_arg

    training_loss_per_epoch = np.zeros(how_many_epochs)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    # kernel for labels with weights
    unnormalized_weights = np.sum(true_labels,0)
    weights = unnormalized_weights/np.sum(unnormalized_weights)

    ######################################################

    """ computing mean embedding of true data """
    emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)
    emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
    outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
    mean_emb1 = torch.mean(outer_emb1, 0)

    if is_private:
        noise = noise_std_for_privacy*torch.randn(mean_emb1.size())
        noise = noise.to(device)
        mean_emb1 = mean_emb1 + noise

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):


            # zero the parameter gradients
            optimizer.zero_grad()

            label_input = (1 * (torch.rand((mini_batch_size)) < weights[1])).type(torch.FloatTensor)
            label_input = label_input.to(device)
            feature_input = torch.randn((mini_batch_size, input_size-1)).to(device)
            input_to_model = torch.cat((feature_input, label_input[:,None]), 1)
            outputs = model(input_to_model)


            """ computing mean embedding of generated samples """
            emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)

            label_input_t = torch.zeros((mini_batch_size, n_classes))
            idx_1 = (label_input == 1.).nonzero()[:,0]
            idx_0 = (label_input == 0.).nonzero()[:,0]
            label_input_t[idx_1, 1] = 1.
            label_input_t[idx_0, 0] = 1.

            emb2_labels = Feature_labels(label_input_t, weights)
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if epoch%100==0:
            print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss



    plt.figure(3)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')
    plt.yscale('log')

    plt.figure(4)
    plt.subplot(211)
    plt.plot(mean_emb1[:, 0].cpu(), 'b')
    plt.plot(mean_emb2[:, 0].cpu().detach().numpy(), 'b--')
    plt.subplot(212)
    plt.plot(mean_emb1[:, 1].cpu(), 'r')
    plt.plot(mean_emb2[:, 1].cpu().detach().numpy(), 'r--')

    """ now generate samples from the trained network """

    label_input = (1 * (torch.rand((n)) < weights[1])).type(torch.FloatTensor)
    label_input = label_input.to(device)
    feature_input = torch.randn((n, input_size - 1)).to(device)
    input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
    outputs = model(input_to_model)

    samp_input_features = outputs

    label_input_t = torch.zeros((n, n_classes))
    idx_1 = (label_input == 1.).nonzero()[:, 0]
    idx_0 = (label_input == 0.).nonzero()[:, 0]
    label_input_t[idx_1, 1] = 1.
    label_input_t[idx_0, 0] = 1.

    samp_labels = label_input_t

    generated_samples = samp_input_features.cpu().detach().numpy()
    generated_labels = samp_labels.cpu().detach().numpy()

    LR_model = LogisticRegression(solver='lbfgs', max_iter=5000)
    LR_model.fit(generated_samples, np.argmax(generated_labels, axis=1)) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('is private?', is_private)
    print('ROC is', roc_auc_score(y_test, pred))
    print('PRC is', average_precision_score(y_test, pred))
    print('n_features are ', n_features)

    # save results

    # n_0 = weights[0]
    # n_1 = weights[1]
    #
    # method = os.path.join(Results_PATH, 'Isolet_condMMD_mini_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s_sigma2=%s_n0=%s_n1=%s_nfeatures=%s' % (
    # mini_batch_size, input_size, hidden_size_1, hidden_size_2, sigma2, n_0, n_1, n_features))
    #
    # print('model specifics are', method)

    # np.save(method + '_loss.npy', training_loss_per_epoch)
    # np.save(method + '_input_feature_samps.npy', generated_samples)
    # np.save(method + '_output_label_samps.npy', generated_labels)


if __name__ == '__main__':
    main()
# if __name__ == '__main__':
#
#     n_features_arg=[100, 1000, 10000, 50000, 80000, 100000]
#     mini_batch_arg=[200, 500,1000]
#     how_many_epochs_arg=[1000, 2000]
#     grid=ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg, "how_many_epochs_arg": how_many_epochs_arg})
#     for elem in grid:
#         print (elem)
#         for i in range(5):
#             main(elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"])

