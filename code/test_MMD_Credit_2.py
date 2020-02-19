"""" test a simple generating training using MMD for relatively simple datasets """
""" with generating labels together with the input features """


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

def Feature_labels(labels, weights):

    n_0 = torch.Tensor([weights[0]])
    n_1 = torch.Tensor([weights[1]])

    weighted_label_0 = 1/n_0*labels[:,0]
    weighted_label_1 = 1/n_1*labels[:,1]

    weighted_labels_feature = torch.cat((weighted_label_0[:,None], weighted_label_1[:,None]), 1)

    return weighted_labels_feature

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
            # self.softmax = torch.nn.Softmax(dim=1)



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
    format = 'npy'
    # (1) load data
    if format == 'csv':
        data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")

        print(data.shape)

        # data.info()

        class_names = {0: 'Not Fraud', 1: 'Fraud'}
        print(data.Class.value_counts().rename(index=class_names))
        # highly imbalanced data: 492(Fraud) and 284315 (Not Fraud)

        feature_names = data.iloc[:, 1:30].columns
        target = data.iloc[:1, 30:].columns
        # print(feature_names)
        # print(target)

        data_features = data[feature_names]
        data_target = data[target]

        n_0 = 1
        n_1 = 0.002

    elif format == 'npy':
        data_features_npy = np.load('../data/Isolet/isolet_data.npy')
        data_target_npy = np.load('../data/Isolet/isolet_labels.npy')

        # target_pos=np.where(data_target_npy==1)
        # target_neg=np.where(data_target_npy==0)
        #
        # data_features_npy_pos=data_features_npy[target_pos]
        # data_target_npy_pos=np.ones(len(data_features_npy_pos))
        #
        # data_features_npy_neg = data_features_npy[target_neg]
        # data_target_npy_neg=np.ones(len(data_features_npy_neg))
        #
        # data_features_npy=data_features_npy_pos
        # data_target_npy=data_target_npy_pos

        # dtype = [('Col1', 'int32'), ('Col2', 'float32'), ('Col3', 'float32')]
        values = data_features_npy
        index = ['Row' + str(i) for i in range(1, len(values) + 1)]

        values_l = data_target_npy
        index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

        data_features = pd.DataFrame(values, index=index)
        data_target = pd.DataFrame(values_l, index=index_l)

        n_0 = 10
        n_1 = 2


    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.80, test_size=0.20, random_state=0)

    # unpack data
    data_samps = X_train.values
    y_labels = y_train.values.ravel()

    n_classes=2

    if n_classes==2:
        n, input_dim = data_samps.shape

        true_labels = np.zeros((n,n_classes))
        idx_1 = y_labels==1
        idx_0 = y_labels==0
        true_labels[idx_1,1] = 1
        true_labels[idx_0,0] = 1
    elif n_classes==1:
        n, input_dim = data_samps.shape

        true_labels = np.ones((n, n_classes))
        #idx_1 = y_labels == 1
        #idx_0 = y_labels == 0
        #true_labels[idx_1, 1] = 1
        #true_labels[idx_0, 0] = 1

    # test how to use RFF for computing the kernel matrix
    idx_rp = np.random.permutation(800)#10000
    med = util.meddistance(data_samps[idx_rp,:])
    sigma2 = med**2
    # sigma2 = med # it seems to be more useful to use smaller length scale than median heuristic
    print('length scale from median heuristic is', sigma2)

    # random Fourier features
    n_features = 100


    """ training a Generator via minimizing MMD """
    # try more random features with a larger batch size
    mini_batch_size = 4000

    input_size = 100
    hidden_size_1 = 400
    hidden_size_2 = 200
    output_size = input_dim + n_classes

    # model = Generative_Model(input_dim=input_dim, how_many_Gaussians=num_Gaussians)
    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size, n_classes = n_classes)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    how_many_epochs = 100 #1000
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)


    ###########################################################
    """ computing mean embedding of true data """

    #we choose arbitrary n_features number and then we reduce the existing features from each training instance to that with n_features
    emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq) ## data_samps (199364, 29)

    # kernel for labels with weights
    # n_0, n_1 = np.sum(true_labels, 0)

    #these are manual numebers for the weights, e.g. in creadit dataset 0.2% are positive labels
    n_0 = 80
    n_1 = 20


    if n_classes==2:
        weights = [n_0, n_1]
        #true labels: for each instance we have a one-got vector
        # to balance we weigh each entry, e.g. 0.01 for false and 5 for true label because we have 500 more false labels
        # array([[1., 0.],
        #        [1., 0.],
        #        [0., 1.],
        #        ...,
        #        [0., 1.],
        #        [1., 0.],
        #        [0., 1.]])
        # tensor([[0.0100, 0.0000],
        #         [0.0100, 0.0000],
        #         [0.0000, 5.0000],
        #         ...,
        #         [0.0000, 5.0000],
        #         [0.0100, 0.0000],
        #         [0.0000, 5.0000]])

        emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
        # emb1_labels = torch.Tensor(true_labels)
        #for some reason combining features and labels
        outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
        mean_emb1 = torch.mean(outer_emb1, 0)
    elif n_classes==1:
        mean_emb1 = torch.mean(emb1_input_features, 0)

    #plt.plot(mean_emb1[:, 0].cpu().numpy(), 'b')
    #plt.plot(mean_emb1[:, 1].cpu().numpy(), 'r')

    #######################################################3
    # TRAINING
    # we train the generator with the MMD loss
    #
    # generator -> features -> emb2/RFF_Gaussian (inputs and labels)-> outer_emb2 -> mean mb2


    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        # annealing_rate =

        for i in range(how_many_iter):

            optimizer.zero_grad()
            outputs = model(torch.randn((mini_batch_size, input_size)))

            samp_input_features = outputs[:,0:input_dim]
            samp_labels = outputs[:,-n_classes:]

            """ computing mean embedding of generated samples """
            emb2_input_features = RFF_Gauss(n_features, samp_input_features, W_freq)
            # kernel for labels with weights
            # emb2_labels = samp_labels
            # ns_0 = 1
            # ns_1 = 1
            # weights = [ns_0, ns_1]
            emb2_labels = Feature_labels(samp_labels, weights)
            #e.g. [4000, 100, 2]=[4000, 100], [4000, 2]
            # k stays means we deal with each row (data/label pair separately)
            # each feature vector is multiplied by both frequencies getting [num_features, 2] matrix for each instance
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            #mean across all the examples [100,2]
            mean_emb2 = torch.mean(outer_emb2, 0)


            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

            # loss = torch.norm(mean_emb1[:,0]-mean_emb2[:,0], p=2) + torch.norm(mean_emb1[:,1]-mean_emb2[:,1], p=2) + torch.norm(mean_emb1[:,2]-mean_emb2[:,2], p=2)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # if running_loss<=1e-4:
        #     break
        print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss



    # plt.figure(3)
    # plt.plot(training_loss_per_epoch)
    # plt.title('MMD as a function of epoch')
    # plt.yscale('log')



    # plt.figure(4)
    # plt.plot(mean_emb1[:, 0].cpu().numpy(), 'b')
    # plt.plot(mean_emb1[:, 1].cpu().numpy(), 'r')
    # plt.plot(mean_emb2[:, 0].detach().numpy(), 'b--')
    # plt.plot(mean_emb2[:, 1].detach().numpy(), 'r--')


    # model.eval()

    ##################################################3
    # generate data

    # cannot produce the data for sparse label
    outputs = model(torch.randn((n, input_size)))
    samp_input_features = outputs[:, 0:input_dim]
    samp_labels = outputs[:, -n_classes:]

    generated_samples = samp_input_features.detach().numpy()
    generated_labels = samp_labels.detach().numpy()


    ##############################################################################
    # train on the generated data

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(generated_samples, np.argmax(generated_labels, axis=1)) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    print('ROC is', roc_auc_score(y_test, pred))
    print('PRC is', average_precision_score(y_test, pred))

    # not just looking at the numbers, let's also look at the statistic of each of the input features
    # inspection code from https://www.kaggle.com/renjithmadhavan/credit-card-fraud-detection-using-python
    plt.figure(1, figsize=(15, 12))
    df = data
    plt.subplot(5, 6, 1); plt.plot(df.V1); plt.subplot(5, 6, 15); plt.plot(df.V15)
    plt.subplot(5, 6, 2); plt.plot(df.V2); plt.subplot(5, 6, 16); plt.plot(df.V16)
    plt.subplot(5, 6, 3); plt.plot(df.V3); plt.subplot(5, 6, 17); plt.plot(df.V17)
    plt.subplot(5, 6, 4); plt.plot(df.V4); plt.subplot(5, 6, 18); plt.plot(df.V18)
    plt.subplot(5, 6, 5); plt.plot(df.V5); plt.subplot(5, 6, 19); plt.plot(df.V19)
    plt.subplot(5, 6, 6); plt.plot(df.V6); plt.subplot(5, 6, 20); plt.plot(df.V20)
    plt.subplot(5, 6, 7); plt.plot(df.V7); plt.subplot(5, 6, 21); plt.plot(df.V21)
    plt.subplot(5, 6, 8); plt.plot(df.V8); plt.subplot(5, 6, 22); plt.plot(df.V22)
    plt.subplot(5, 6, 9); plt.plot(df.V9); plt.subplot(5, 6, 23); plt.plot(df.V23)
    plt.subplot(5, 6, 10); plt.plot(df.V10); plt.subplot(5, 6, 24); plt.plot(df.V24)
    plt.subplot(5, 6, 11); plt.plot(df.V11); plt.subplot(5, 6, 25); plt.plot(df.V25)
    plt.subplot(5, 6, 12); plt.plot(df.V12); plt.subplot(5, 6, 26); plt.plot(df.V26)
    plt.subplot(5, 6, 13); plt.plot(df.V13); plt.subplot(5, 6, 27); plt.plot(df.V27)
    plt.subplot(5, 6, 14); plt.plot(df.V14); plt.subplot(5, 6, 28); plt.plot(df.V28)
    plt.subplot(5, 6, 29); plt.plot(df.Amount)

    plt.figure(2, figsize=(15, 12))
    plt.subplot(5, 6, 1); plt.plot(generated_samples[:,0]); plt.subplot(5, 6, 15); plt.plot(generated_samples[:,14])
    plt.subplot(5, 6, 2); plt.plot(generated_samples[:,1]); plt.subplot(5, 6, 16); plt.plot(generated_samples[:,15])
    plt.subplot(5, 6, 3); plt.plot(generated_samples[:,2]); plt.subplot(5, 6, 17); plt.plot(generated_samples[:,16])
    plt.subplot(5, 6, 4); plt.plot(generated_samples[:,3]); plt.subplot(5, 6, 18); plt.plot(generated_samples[:,17])
    plt.subplot(5, 6, 5); plt.plot(generated_samples[:,4]); plt.subplot(5, 6, 19); plt.plot(generated_samples[:,18])
    plt.subplot(5, 6, 6); plt.plot(generated_samples[:,5]); plt.subplot(5, 6, 20); plt.plot(generated_samples[:,19])
    plt.subplot(5, 6, 7); plt.plot(generated_samples[:,6]); plt.subplot(5, 6, 21); plt.plot(generated_samples[:, 20])
    plt.subplot(5, 6, 8); plt.plot(generated_samples[:,7]); plt.subplot(5, 6, 22); plt.plot(generated_samples[:,21])
    plt.subplot(5, 6, 9); plt.plot(generated_samples[:,8]); plt.subplot(5, 6, 23); plt.plot(generated_samples[:,22])
    plt.subplot(5, 6, 10); plt.plot(generated_samples[:,9]); plt.subplot(5, 6, 24); plt.plot(generated_samples[:,23])
    plt.subplot(5, 6, 11); plt.plot(generated_samples[:,10]); plt.subplot(5, 6, 25); plt.plot(generated_samples[:,24])
    plt.subplot(5, 6, 12); plt.plot(generated_samples[:,11]); plt.subplot(5, 6, 26); plt.plot(generated_samples[:,25])
    plt.subplot(5, 6, 13); plt.plot(generated_samples[:,12]); plt.subplot(5, 6, 27); plt.plot(generated_samples[:,26])
    plt.subplot(5, 6, 14); plt.plot(generated_samples[:,13]); plt.subplot(5, 6, 28); plt.plot(generated_samples[:,27])
    plt.subplot(5, 6, 29); plt.plot(generated_samples[:,28])

    plt.show()

if __name__ == '__main__':
    main()


# if this doesn't work, then try out the conditional generator, where label is also an input with the random noise