"""" test a simple generating training using MMD for relatively simple datasets """
""" for generating input features given random noise and the label """
# Mijung wrote on Dec 20, 2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random

import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import os

Results_PATH = "/".join([os.getenv("HOME"), "condMMD/"])

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

def Feature_labels(labels, weights):

    n_0 = torch.Tensor([weights[0]])
    n_1 = torch.Tensor([weights[1]])

    weighted_label_0 = 1/n_0*labels[:,0]
    weighted_label_1 = 1/n_1*labels[:,1]

    weighted_labels_feature = torch.cat((weighted_label_0[:,None], weighted_label_1[:,None]), 1)

    return weighted_labels_feature


def main():

    random.seed(0)

    # (1) load data
    data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")

    feature_names = data.iloc[:, 1:30].columns
    target = data.iloc[:1, 30:].columns

    data_features = data[feature_names]
    data_target = data[target]

    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=0)

    # unpack data
    data_samps = X_train.values
    y_labels = y_train.values.ravel()

    n_classes = 2
    n, input_dim = data_samps.shape

    true_labels = np.zeros((n,n_classes))
    idx_1 = y_labels==1
    idx_0 = y_labels==0
    true_labels[idx_1,1] = 1
    true_labels[idx_0,0] = 1

    sigma2 = 44.21956765521877

    # random Fourier features
    n_features = 100


    """ training a Generator via minimizing MMD """
    # try more random features with a larger batch size
    mini_batch_size = 4000

    # input_size = 100 + 1
    # hidden_size_1 = 500
    # hidden_size_2 = 200

    # input_size = 50 + 1
    # hidden_size_1 = 300
    # hidden_size_2 = 200

    input_size = 10 + 1
    hidden_size_1 = 200
    hidden_size_2 = 100
    output_size = input_dim


    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    """ computing mean embedding of true data """
    emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)

    # kernel for labels with weights
    n_0 = 1
    n_1 = 0.002
    weights = [n_0, n_1]
    emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
    outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
    mean_emb1 = torch.mean(outer_emb1, 0)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(mean_emb1[:, 0], 'b')
    plt.subplot(212)
    plt.plot(mean_emb1[:, 1], 'r')

    # load results
    method = os.path.join(Results_PATH, 'condMMD_mini_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s_sigma2=%s' % (
    mini_batch_size, input_size, hidden_size_1, hidden_size_2, sigma2))



    training_loss_per_epoch = np.load(method + '_loss.npy')

    plt.figure(2)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')

    generated_samples = np.load(method+'_input_feature_samps.npy')
    generated_labels = np.load(method+'_output_label_samps.npy')

    emb2_input_features = RFF_Gauss(n_features, torch.Tensor(generated_samples), W_freq)
    emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights)
    outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
    mean_emb2 = torch.mean(outer_emb2, 0)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(mean_emb2[:, 0].detach().numpy(), 'b--')
    plt.subplot(212)
    plt.plot(mean_emb2[:, 1].detach().numpy(), 'r--')


    LR_model = LogisticRegression(solver='lbfgs', max_iter=5000)
    LR_model.fit(generated_samples, np.argmax(generated_labels, axis=1)) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('ROC is', roc_auc_score(y_test, pred))
    print('PRC is', average_precision_score(y_test, pred))


# if __name__ == '__main__':
#     main()

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

    plt.show()

if __name__ == '__main__':
    main()


# if this doesn't work, then try out the conditional generator, where label is also an input with the random noise