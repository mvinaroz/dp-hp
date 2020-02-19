"""" test a simple generating training using MMD for relatively simple datasets """
""" for generating input features given random noise and the label for ISOLET data """

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


def main(n_features_arg, mini_batch_size_arg, how_many_epochs_arg):


    # as a reference, with n_features= 500 and mini_batch_size=1000, after 1000 epochs
    # I get ROC= 0.54, PRC=0.21
    # with different seed, I get ROC=0.5 and PRC 0.2. Similar to before.

    # when I tested n_features = 500 and mini_batch_size=500,
    # I got ROC = 0.6163 and PRC = 0.2668

    random.seed(0)

    print("epileptic seizure recognition dataset")
    print(socket.gethostname())

    if 'g0' not in socket.gethostname():
        data = pd.read_csv("../data/Epileptic/data.csv")
    else:
        # (1) load data
        data = pd.read_csv('/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Epileptic/data.csv')

    feature_names = data.iloc[:, 1:-1].columns
    target = data.iloc[:, -1:].columns

    data_features = data[feature_names]
    data_target = data[target]

    for i, row in data_target.iterrows():
      if data_target.at[i,'y']!=1:
        data_target.at[i,'y'] = 0

    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=0)

    # unpack data
    data_samps = X_train.values
    y_labels = y_train.values.ravel()

    ##########################################################

    # test logistic regression on the real data
    LR_model = LogisticRegression(solver='lbfgs', max_iter=5000)
    LR_model.fit(X_train, y_labels) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('ROC on real test data is', roc_auc_score(y_test, pred))
    print('PRC on real test data is', average_precision_score(y_test, pred))

    # ROC on real test data is 0.5569809947080179
    # PRC on real test data is 0.2860654400815256

    ################################################################

    n_classes = 2
    n, input_dim = data_samps.shape

    """ we use 10 datapoints to compute the median heuristic (then discard), and use the rest for training """
    idx_rp = np.random.permutation(n)
    num_data_pt_to_discard = 10
    idx_to_discard = idx_rp[0:num_data_pt_to_discard]
    idx_to_keep = idx_rp[num_data_pt_to_discard:]

    # sigma_array = np.zeros(input_dim)
    # for i in np.arange(0,input_dim):
    #     med = util.meddistance(np.expand_dims(data_samps[idx_to_discard,i],1))
    #     sigma_array[i] = med
    # sigma2 = sigma_array**2

    med = util.meddistance(data_samps[idx_to_discard, :])
    sigma2 = med**2

    # print('length scale from median heuristic is', sigma2)

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

    mini_batch_size =  mini_batch_size_arg
    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 =  2* input_dim
    output_size = input_dim

    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    how_many_epochs = how_many_epochs_arg
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    # kernel for labels with weights
    unnormalized_weights = np.sum(true_labels,0)
    positive_label_ratio = unnormalized_weights[1]/unnormalized_weights[0]

    weights = unnormalized_weights/np.sum(unnormalized_weights)

    ######################################################

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            """ computing mean embedding of subsampled true data """
            sample_idx = random.choices(np.arange(n), k=mini_batch_size)
            sampled_data = data_samps[sample_idx,:]
            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(sampled_data), W_freq)
            sampled_labels = true_labels[sample_idx,:]
            emb1_labels = Feature_labels(torch.Tensor(sampled_labels), weights)
            outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
            mean_emb1 = torch.mean(outer_emb1, 0)


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

        if epoch%10==0:
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

    print('ROC is', roc_auc_score(y_test, pred))
    print('PRC is', average_precision_score(y_test, pred))
    print('n_features are ', n_features)

    # save results
    # n_0 = weights[0]
    # n_1 = weights[1]

    # method = os.path.join(Results_PATH, 'Epileptic_condMMD_mini_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s_sigma2=%s_n0=%s_n1=%s_nfeatures=%s' % (
    # mini_batch_size, input_size, hidden_size_1, hidden_size_2, sigma2, n_0, n_1, n_features))
    #
    # print('model specifics are', method)
    #
    # np.save(method + '_loss.npy', training_loss_per_epoch)
    # np.save(method + '_input_feature_samps.npy', generated_samples)
    # np.save(method + '_output_label_samps.npy', generated_labels)



#to test one setting put only one element array for each variable

if __name__ == '__main__':

    how_many_epochs_arg=[1000, 2000]
    n_features_arg=[50, 500, 5000, 50000, 80000, 100000]
    mini_batch_arg=[500,1000]

    grid = ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg,
                          "how_many_epochs_arg": how_many_epochs_arg})
    for elem in grid:
        print(elem)
        for i in range(3):
            print(i)
            main(elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"])
        print('*'*30)




