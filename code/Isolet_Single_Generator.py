""" test training seperate generators for each label """

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import argparse

import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import os

# user='kamil'

user='mijung'

if user=='mijung':
    Results_PATH = "/".join([os.getenv("HOME"), "separate_Isolet/"])
elif user =='kamil':
    Results_PATH = "/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/results/separate_isolet"

def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
    return Z


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

# def main(inp, h1, h2, mini_batch_size):
def main():

    random.seed(0)

    #############################33
    # get classification on real data

    # (1) load data
    data_features_npy = np.load('../data/Isolet/isolet_data.npy')
    data_target_npy = np.load('../data/Isolet/isolet_labels.npy')

    # dtype = [('Col1', 'int32'), ('Col2', 'float32'), ('Col3', 'float32')]
    values = data_features_npy
    index = ['Row' + str(i) for i in range(1, len(values) + 1)]

    values_l = data_target_npy
    index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

    data_features = pd.DataFrame(values, index=index)
    data_target = pd.DataFrame(values_l, index=index_l)

    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=0)

    # test logistic regression on the real data
    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(X_train, y_train.values.ravel()) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('ROC on real test data is', roc_auc_score(y_test, pred))
    print('PRC on real test data is', average_precision_score(y_test, pred))

    y_labels = y_train.values.ravel()  # X_train_pos
    X_train = X_train.values

    n_tot = X_train.shape[0]

    X_train_pos =  X_train[y_labels==1,:]
    y_train_pos = y_labels[y_labels==1]

    X_train_neg = X_train[y_labels==0,:]
    y_train_neg = y_labels[y_labels == 0]

    #############################################3
    # train generator

    n_classes = 2

    for which_class in range(n_classes):

        if which_class==1:
            """ First train for positive label """
            n, input_dim = X_train_pos.shape
            data_samps = X_train_pos
            del X_train_pos
        else:
            n, input_dim = X_train_neg.shape
            data_samps = X_train_neg
            del X_train_neg


        # test how to use RFF for computing the kernel matrix
        idx_rp = np.random.permutation(np.min([n, 10000]))
        med = util.meddistance(data_samps[idx_rp,:])
        del idx_rp
        sigma2 = med**2
        print('length scale from median heuristic is', sigma2)

        # random Fourier features

        # n_features = 10000

        """ training a Generator via minimizing MMD """
        # try more random features with a larger batch size

        # mini_batch_size = n
        #
        # input_size = np.int(inp * input_dim)
        # hidden_size_1 = h1 * input_dim
        # hidden_size_2 = h2 * input_dim
        # output_size = input_dim

        # doesn't work
        # if which_class == 1:
        #
        #     mini_batch_size = 200
        #     input_size = 40
        #     hidden_size_1 = 200
        #     hidden_size_2 = 100
        #     output_size = input_dim
        #     how_many_epochs = 1000
        #
        # else:  # for extremely imbalanced dataset
        #     mini_batch_size = 4000  # large minibatch size for speeding up the training process
        #     input_size = 100
        #     hidden_size_1 = 500
        #     hidden_size_2 = 200
        #     output_size = input_dim
        #     how_many_epochs = 100  # 400

        # if which_class==1:
        #
        #     mini_batch_size = 400
        #     input_size = 400
        #     hidden_size_1 = 100
        #     hidden_size_2 = 100
        #     output_size = input_dim
        #     how_many_epochs = 30
        #
        #
        # else: # for extremely imbalanced dataset
        #
        #     mini_batch_size = 400
        #     input_size = 400
        #     hidden_size_1 = 300
        #     hidden_size_2 = 100
        #     output_size = input_dim
        #     how_many_epochs = 30

            # mini_batch_size = 4000 # large minibatch size for speeding up the training process
            # input_size = 100
            # hidden_size_1 = 500
            # hidden_size_2 = 200
            # output_size = input_dim
            # how_many_epochs = 400


        # in this particular setup
        # n_features = 10000
        # mini_batch_size = n
        # input_size = 10 + 1
        # hidden_size_1 = 2*input_dim
        # hidden_size_2 = np.int(1.2* input_dim)
        # output_size = input_dim
        #
        # ROC is 0.8255916092106221
        # PRC is 0.44640266105356663

        # in this particular setup:
        # n_features = 10000
        # mini_batch_size = n
        # input_size = 5 + 1
        # hidden_size_1 = input_dim
        # hidden_size_2 = np.int(1.2* input_dim)
        # output_size = input_dim
        # how_many_epochs = 200
        #
        # ROC is 0.8116268530623435
        # PRC is 0.4288499474923227


        # n_features = 15000
        # mini_batch_size = n
        # input_size = 10 + 1
        # hidden_size_1 = 2*input_dim
        # hidden_size_2 = np.int(1.2* input_dim)
        # output_size = input_dim


        n_features = 10000
        mini_batch_size = n
        input_size = 10 + 1
        hidden_size_1 = 4*input_dim
        hidden_size_2 = 2* input_dim
        output_size = input_dim


        how_many_epochs = 1000

        model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                                 output_size=output_size)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        how_many_iter = np.int(n/mini_batch_size)

        training_loss_per_epoch = np.zeros(how_many_epochs)

        draws = n_features // 2
        W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)

        """ computing mean embedding of true data """
        emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)
        mean_emb1 = torch.mean(emb1_input_features, 0)

        del data_samps
        del emb1_input_features

        #plt.plot(mean_emb1, 'b')

        ######################################33
        # start training

        print('Starting Training')

        for epoch in range(how_many_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            # annealing_rate =

            for i in range(how_many_iter):

                # zero the parameter gradients
                optimizer.zero_grad()
                input_to_model = torch.randn((mini_batch_size, input_size))
                outputs = model(input_to_model)

                """ computing mean embedding of generated samples """
                emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                mean_emb2 = torch.mean(emb2_input_features, 0)

                loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            print('epoch # and running loss are ', [epoch, running_loss])
            training_loss_per_epoch[epoch] = running_loss


        #################################3
        # generate samples

        """ now generated samples using the trained generator """
        input_to_model = torch.randn((n, input_size))
        outputs = model(input_to_model)
        samp_input_features = outputs

        if which_class==1:
            generated_samples_pos = samp_input_features.detach().numpy()

            # save results
            method = os.path.join(Results_PATH, 'Isolet_pos_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s_nfeat=%s' %(mini_batch_size, input_size, hidden_size_1, hidden_size_2, n_features))
            np.save(method + '_loss.npy', training_loss_per_epoch)
            np.save(method + '_input_feature_samps.npy', generated_samples_pos)

        else:
            generated_samples_neg = samp_input_features.detach().numpy()

            # save results
            method = os.path.join(Results_PATH, 'Isolet_neg_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s_nfeat=%s' %(mini_batch_size, input_size, hidden_size_1, hidden_size_2, n_features))
            np.save(method + '_loss.npy', training_loss_per_epoch)
            np.save(method + '_input_feature_samps.npy', generated_samples_neg)

        # plt.figure(3)
        # plt.plot(training_loss_per_epoch)
        # plt.title('MMD as a function of epoch')
        # plt.yscale('log')
        #
        # plt.figure(4)
        # plt.plot(mean_emb1, 'b')
        # plt.plot(mean_emb2.detach().numpy(), 'r--')

    # mix data for positive and negative labels
    generated_input_features = np.concatenate((generated_samples_pos, generated_samples_neg), axis=0)
    corresponding_labels = np.concatenate((y_train_pos, y_train_neg))

    idx_shuffle = np.random.permutation(n_tot)
    shuffled_x_train = generated_input_features[idx_shuffle, :]
    shuffled_y_train = corresponding_labels[idx_shuffle]


    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(shuffled_x_train, shuffled_y_train) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data


    ROC = roc_auc_score(y_test, pred)
    PRC = average_precision_score(y_test, pred)

    print('ROC is', ROC)
    print('PRC is', PRC)

    method = os.path.join(Results_PATH, 'Isolet_separate_generators_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s_nfeat=%s'
                          %(mini_batch_size, input_size, hidden_size_1, hidden_size_2, n_features)) # save with the label 1 setup
    np.save(method + '_PRC.npy', ROC)
    np.save(method + '_ROC.npy', PRC)

    print('model specifics are', method)



if __name__ == '__main__':
    # main(0.2,4,2, 100)

    main()
    # not just looking at the numbers, let's also look at the statistic of each of the input features
    # inspection code from https://www.kaggle.com/renjithmadhavan/credit-card-fraud-detection-using-python

    # plt.figure(1, figsize=(15, 12))
    # df = data
    # plt.subplot(5, 6, 1); plt.plot(df.V1); plt.subplot(5, 6, 15); plt.plot(df.V15)
    # plt.subplot(5, 6, 2); plt.plot(df.V2); plt.subplot(5, 6, 16); plt.plot(df.V16)
    # plt.subplot(5, 6, 3); plt.plot(df.V3); plt.subplot(5, 6, 17); plt.plot(df.V17)
    # plt.subplot(5, 6, 4); plt.plot(df.V4); plt.subplot(5, 6, 18); plt.plot(df.V18)
    # plt.subplot(5, 6, 5); plt.plot(df.V5); plt.subplot(5, 6, 19); plt.plot(df.V19)
    # plt.subplot(5, 6, 6); plt.plot(df.V6); plt.subplot(5, 6, 20); plt.plot(df.V20)
    # plt.subplot(5, 6, 7); plt.plot(df.V7); plt.subplot(5, 6, 21); plt.plot(df.V21)
    # plt.subplot(5, 6, 8); plt.plot(df.V8); plt.subplot(5, 6, 22); plt.plot(df.V22)
    # plt.subplot(5, 6, 9); plt.plot(df.V9); plt.subplot(5, 6, 23); plt.plot(df.V23)
    # plt.subplot(5, 6, 10); plt.plot(df.V10); plt.subplot(5, 6, 24); plt.plot(df.V24)
    # plt.subplot(5, 6, 11); plt.plot(df.V11); plt.subplot(5, 6, 25); plt.plot(df.V25)
    # plt.subplot(5, 6, 12); plt.plot(df.V12); plt.subplot(5, 6, 26); plt.plot(df.V26)
    # plt.subplot(5, 6, 13); plt.plot(df.V13); plt.subplot(5, 6, 27); plt.plot(df.V27)
    # plt.subplot(5, 6, 14); plt.plot(df.V14); plt.subplot(5, 6, 28); plt.plot(df.V28)
    # plt.subplot(5, 6, 29); plt.plot(df.Amount)
    #
    # plt.figure(2, figsize=(15, 12))
    # plt.subplot(5, 6, 1); plt.plot(generated_samples[:,0]); plt.subplot(5, 6, 15); plt.plot(generated_samples[:,14])
    # plt.subplot(5, 6, 2); plt.plot(generated_samples[:,1]); plt.subplot(5, 6, 16); plt.plot(generated_samples[:,15])
    # plt.subplot(5, 6, 3); plt.plot(generated_samples[:,2]); plt.subplot(5, 6, 17); plt.plot(generated_samples[:,16])
    # plt.subplot(5, 6, 4); plt.plot(generated_samples[:,3]); plt.subplot(5, 6, 18); plt.plot(generated_samples[:,17])
    # plt.subplot(5, 6, 5); plt.plot(generated_samples[:,4]); plt.subplot(5, 6, 19); plt.plot(generated_samples[:,18])
    # plt.subplot(5, 6, 6); plt.plot(generated_samples[:,5]); plt.subplot(5, 6, 20); plt.plot(generated_samples[:,19])
    # plt.subplot(5, 6, 7); plt.plot(generated_samples[:,6]); plt.subplot(5, 6, 21); plt.plot(generated_samples[:, 20])
    # plt.subplot(5, 6, 8); plt.plot(generated_samples[:,7]); plt.subplot(5, 6, 22); plt.plot(generated_samples[:,21])
    # plt.subplot(5, 6, 9); plt.plot(generated_samples[:,8]); plt.subplot(5, 6, 23); plt.plot(generated_samples[:,22])
    # plt.subplot(5, 6, 10); plt.plot(generated_samples[:,9]); plt.subplot(5, 6, 24); plt.plot(generated_samples[:,23])
    # plt.subplot(5, 6, 11); plt.plot(generated_samples[:,10]); plt.subplot(5, 6, 25); plt.plot(generated_samples[:,24])
    # plt.subplot(5, 6, 12); plt.plot(generated_samples[:,11]); plt.subplot(5, 6, 26); plt.plot(generated_samples[:,25])
    # plt.subplot(5, 6, 13); plt.plot(generated_samples[:,12]); plt.subplot(5, 6, 27); plt.plot(generated_samples[:,26])
    # plt.subplot(5, 6, 14); plt.plot(generated_samples[:,13]); plt.subplot(5, 6, 28); plt.plot(generated_samples[:,27])
    # plt.subplot(5, 6, 29); plt.plot(generated_samples[:,28])

    # plt.show()
