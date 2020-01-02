"""" test a simple generating training using MMD for relatively simple datasets """
""" with generating labels together with the input features """
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

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, n_classes):
            super(Generative_Model, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size
            self.n_classes = n_classes

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
            self.softmax = torch.nn.Softmax(dim=1)

            # self.fc1 = torch.nn.utils.weight_norm(torch.nn.Linear(self.input_size, self.hidden_size_1), name='weight')
            # self.relu = torch.nn.ReLU()
            # self.fc2 = torch.nn.utils.weight_norm(torch.nn.Linear(self.hidden_size_1, self.hidden_size_2), name='weight')

            # self.fc1 = torch.nn.utils.spectral_norm(torch.nn.Linear(self.input_size, self.hidden_size_1))
            # self.relu = torch.nn.ReLU()
            # self.fc2 = torch.nn.utils.spectral_norm(torch.nn.Linear(self.hidden_size_1, self.hidden_size_2))

            # self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
            # self.softmax = torch.nn.Softmax()



        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)

            # hidden = self.fc1(x)
            # relu = self.relu(hidden)
            # output = self.fc2(relu)
            # output = self.relu(output)
            # output = self.fc3(output)

            output_features = output[:, 0:-self.n_classes]
            output_labels = self.softmax(output[:, -self.n_classes:])
            output_total = torch.cat((output_features, output_labels), 1)
            return output_total

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

    # test how to use RFF for computing the kernel matrix
    idx_rp = np.random.permutation(5000)
    med = util.meddistance(data_samps[idx_rp,:])
    # sigma2 = med**2
    sigma2 = med # it seems to be more useful to use smaller length scale than median heuristic
    print('length scale from median heuristic is', sigma2)

    # random Fourier features
    n_features = 100


    """ training a Generator via minimizing MMD """
    mini_batch_size = 1000

    input_size = 100
    hidden_size_1 = 500
    hidden_size_2 = 200
    output_size = input_dim + n_classes

    # model = Generative_Model(input_dim=input_dim, how_many_Gaussians=num_Gaussians)
    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size, n_classes = n_classes)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    how_many_epochs = 2000
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    """ computing mean embedding of true data """
    emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)
    emb1_labels = torch.Tensor(true_labels)
    outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
    mean_emb1 = torch.mean(outer_emb1, 0)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(torch.randn((mini_batch_size, input_size)))

            samp_input_features = outputs[:,0:input_dim]
            samp_labels = outputs[:,-n_classes:]

            """ computing mean embedding of generated samples """
            emb2_input_features = RFF_Gauss(n_features, samp_input_features, W_freq)
            emb2_labels = samp_labels
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

            # loss = torch.norm(mean_emb1[:,0]-mean_emb2[:,0], p=2) + torch.norm(mean_emb1[:,1]-mean_emb2[:,1], p=2) + torch.norm(mean_emb1[:,2]-mean_emb2[:,2], p=2)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if running_loss<=1e-4:
            break
        print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss



    plt.figure(1)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')


    model.eval()
    generated_samples = samp_input_features.detach().numpy()
    generated_labels = samp_labels.detach().numpy()

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(generated_samples, np.argmax(generated_labels, axis=1)) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('ROC is', roc_auc_score(y_test, pred))
    print('PRC is', average_precision_score(y_test, pred))

    plt.show()

if __name__ == '__main__':
    main()

