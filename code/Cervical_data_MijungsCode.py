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

from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree

import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')

import os

from sdgym import load_dataset

user='anon_m'
save_results = False

if user=='anon_m':
    Results_PATH = "/".join([os.getenv("HOME"), "separate_Cervical/"])
elif user =='anon_k':
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


# def Feature_categorical_data(labels, weights):
#
#     n_0 = torch.Tensor([weights[0]])
#     n_1 = torch.Tensor([weights[1]])
#
#     weighted_label_0 = 1/n_0*labels[:,0]
#     weighted_label_1 = 1/n_1*labels[:,1]
#
#     weighted_labels_feature = torch.cat((weighted_label_0[:,None], weighted_label_1[:,None]), 1)
#
#     return weighted_labels_feature


class Generative_Model(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, num_categorical_inputs, num_numerical_inputs):
            super(Generative_Model, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size
            # self.n_classes = n_classes
            self.num_numerical_inputs = num_numerical_inputs
            self.num_categorical_inputs = num_categorical_inputs

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)


        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)

            output_numerical = self.relu(output[:, 0:self.num_numerical_inputs]) # these numerical values are non-negative
            # output_numerical = torch.round(output_numerical)

            output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
            # output_categorical = torch.round(output_categorical)

            output_combined = torch.cat((output_numerical, output_categorical),1)

            return output_combined


def main():

    data, categorical_columns, ordinal_columns = load_dataset('adult')


    seed_number = 1
    random.seed(seed_number)

    # data pre-prosessing code from https://www.kaggle.com/kernels/scriptcontent/17172286/download
    print(socket.gethostname())
    if'g0' not in socket.gethostname():
        df=pd.read_csv("../data/Cervical/kag_risk_factors_cervical_cancer.csv")
    else:
        df=pd.read_csv("/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")

    df.head()

    df_nan = df.replace("?", np.nan)
    df_nan.head()

    df1 = df_nan.convert_objects(convert_numeric=True)

    df1.columns = df1.columns.str.replace(' ', '')  # deleting spaces for ease of use

    """ this is the key in this data-preprocessing """
    df = df1[df1.isnull().sum(axis=1) < 10]

    numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse', 'Numofpregnancies', 'Smokes(years)',
                    'Smokes(packs/year)', 'HormonalContraceptives(years)', 'IUD(years)', 'STDs(number)', 'STDs:Numberofdiagnosis',
                    'STDs:Timesincefirstdiagnosis', 'STDs:Timesincelastdiagnosis']
    categorical_df = ['Smokes', 'HormonalContraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
                      'STDs:vulvo-perinealcondylomatosis', 'STDs:syphilis', 'STDs:pelvicinflammatorydisease',
                      'STDs:genitalherpes', 'STDs:AIDS', 'STDs:cervicalcondylomatosis',
                      'STDs:molluscumcontagiosum', 'STDs:HIV', 'STDs:HepatitisB', 'STDs:HPV',
                      'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy']

    feature_names = numerical_df + categorical_df[:-1]
    num_numerical_inputs = len(numerical_df)
    num_categorical_inputs = len(categorical_df[:-1])

    for feature in numerical_df:
        # print(feature, '', df[feature].convert_objects(convert_numeric=True).mean())
        feature_mean = round(df[feature].convert_objects(convert_numeric=True).median(), 1)
        df[feature] = df[feature].fillna(feature_mean)

    for feature in categorical_df:
        df[feature] = df[feature].convert_objects(convert_numeric=True).fillna(0.0)


    target = df['Biopsy']
    # feature_names = df.iloc[:, :-1].columns
    inputs = df[feature_names]

    X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.80, test_size=0.20, random_state=seed_number)  # 60% training and 40% test

    # test logistic regression on the real data

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(X_train, y_train)  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data

    print('ROC on real test data from Logistic regression is', roc_auc_score(y_test, pred)) # 0.9444444444444444
    print('PRC on real test data from Logistic regression is', average_precision_score(y_test, pred)) # 0.8955114054451803

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('ROC on real test data from decision tree is', roc_auc_score(y_test, y_pred)) # 1.0
    print('PRC on real test data from decision tree is', average_precision_score(y_test, y_pred)) # 1.0

    clfAB = AdaBoostClassifier(n_estimators=100)
    clfAB.fit(X_train, y_train)
    y_predAB = clfAB.predict(X_test)

    print('ROC on real test data from AdaBoost is', roc_auc_score(y_test, y_predAB)) # 1.0
    print('PRC on real test data from AdaBoost is', average_precision_score(y_test, y_predAB)) # 1.0


    y_labels = y_train.values.ravel()  # X_train_pos
    X_train = X_train.values

    n_tot = X_train.shape[0]

    X_train_pos = X_train[y_labels == 1, :]
    y_train_pos = y_labels[y_labels == 1]

    X_train_neg = X_train[y_labels == 0, :]
    y_train_neg = y_labels[y_labels == 0]

    #############################################3
    # train generators

    n_classes = 2

    for which_class in range(n_classes):

        if which_class == 1:
            """ First train for positive label """
            n, input_dim = X_train_pos.shape
            data_samps = X_train_pos
            # del X_train_pos
            print('number of data samples for this class is', n)
        else:
            n, input_dim = X_train_neg.shape
            data_samps = X_train_neg
            # del X_train_neg
            print('number of data samples for this class is', n)

        # test how to use RFF for computing the kernel matrix
        idx_rp = np.random.permutation(np.min([n, 10000]))
        med = util.meddistance(data_samps[idx_rp, :])
        del idx_rp
        sigma2 = med ** 2
        print('length scale from median heuristic is', sigma2)

        """ specifics of generators are defined here, comment this later """
        features_num = 1000
        if which_class == 1:
            batch_cl1 = np.int(n/3)
            input_cl1 = 5
            hidden1_cl1 = 2*input_dim
            hidden2_cl1 = np.int(1.2*input_dim)
            epochs_num_cl1 = 500
        else:
            batch_cl0 = np.int(n/3)
            input_cl0 = 5
            hidden1_cl0 = 2*input_dim
            hidden2_cl0 = np.int(1.2*input_dim)
            epochs_num_cl0 = 500

        """ end of comment """

        # random Fourier features
        n_features = features_num  # 20000

        """ training a Generator via minimizing MMD """
        if which_class == 1:

            mini_batch_size = batch_cl1  # 400
            input_size = input_cl1  # 400
            hidden_size_1 = hidden1_cl1  # 100
            hidden_size_2 = hidden2_cl1  # 100
            output_size = input_dim
            how_many_epochs = epochs_num_cl1  # 30

        else:  # for extremely imbalanced dataset

            mini_batch_size = batch_cl0  # 400
            input_size = input_cl0  # 400
            hidden_size_1 = hidden1_cl0  # 300
            hidden_size_2 = hidden2_cl0  # 100

            output_size = input_dim
            how_many_epochs = epochs_num_cl0  # 30


        output_size = input_dim

        model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                                 output_size=output_size, num_categorical_inputs = num_categorical_inputs, num_numerical_inputs = num_numerical_inputs)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        how_many_iter = np.int(n / mini_batch_size)

        training_loss_per_epoch = np.zeros(how_many_epochs)

        draws = n_features // 2
        # W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)
        #
        # """ computing mean embedding of true data """
        # emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)

        W_freq = np.random.randn(draws, num_numerical_inputs) / np.sqrt(sigma2)

        """ computing mean embedding of true data """
        numerical_input_data = data_samps[:, 0:num_numerical_inputs]
        emb1_numerical = torch.mean(RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq),0)

        categorical_input_data = data_samps[:, num_numerical_inputs:]
        emb1_categorical = torch.Tensor(np.mean(categorical_input_data, 0)/np.sqrt(num_categorical_inputs))

        # emb1_numerical = RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)
        # emb1_categorical =

        mean_emb1 = torch.cat((emb1_numerical, emb1_categorical))

        # del data_samps
        # del emb1_input_features

        # plt.plot(mean_emb1, 'b')

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
                # emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                # mean_emb2 = torch.mean(emb2_input_features, 0)

                """ write down the embeddings for two types of features """

                numerical_samps = outputs[:, 0:num_numerical_inputs]
                emb2_numerical = torch.mean(RFF_Gauss(n_features, torch.Tensor(numerical_samps), W_freq), 0)

                categorical_samps = outputs[:, num_numerical_inputs:]
                emb2_categorical = torch.mean(categorical_samps, 0) * torch.sqrt(1.0/torch.Tensor([num_categorical_inputs]))

                mean_emb2 = torch.cat((emb2_numerical, emb2_categorical))

                loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

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

        if which_class == 1:
            generated_samples_pos = samp_input_features.detach().numpy()

            # save results
            method = os.path.join(Results_PATH, 'Cervical_pos_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s' % (
                mini_batch_size, input_size, hidden_size_1, hidden_size_2))
            np.save(method + '_loss.npy', training_loss_per_epoch)
            np.save(method + '_input_feature_samps.npy', generated_samples_pos)

        else:
            generated_samples_neg = samp_input_features.detach().numpy()

            # save results
            method = os.path.join(Results_PATH, 'Cervical_neg_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s' % (
                mini_batch_size, input_size, hidden_size_1, hidden_size_2))
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
    generated_input_features = np.round(np.concatenate((generated_samples_pos, generated_samples_neg), axis=0))
    corresponding_labels = np.concatenate((y_train_pos, y_train_neg))

    idx_shuffle = np.random.permutation(n_tot)
    shuffled_x_train = generated_input_features[idx_shuffle, :]
    shuffled_y_train = corresponding_labels[idx_shuffle]

    LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model_ours.fit(shuffled_x_train, shuffled_y_train)  # training on synthetic data
    pred_ours = LR_model_ours.predict(X_test)  # test on real data

    ROC_ours = roc_auc_score(y_test, pred_ours)
    PRC_ours = average_precision_score(y_test, pred_ours)

    print('ROC on generated samples using Logistic regression is', ROC_ours)
    print('PRC on generated samples using Logistic regression is', PRC_ours)


    clf = tree.DecisionTreeClassifier()
    clf.fit(shuffled_x_train, shuffled_y_train)
    y_pred = clf.predict(X_test)

    print('ROC by decision tree trained on generated data is', roc_auc_score(y_test, y_pred)) # 1.0
    print('PRC by decision tree trained on generated data', average_precision_score(y_test, y_pred)) # 1.0

    clfAB = AdaBoostClassifier(n_estimators=100)
    clfAB.fit(shuffled_x_train, shuffled_y_train)
    y_predAB = clfAB.predict(X_test)

    print('ROC by AdaBoost trained on generated data is', roc_auc_score(y_test, y_predAB)) # 1.0
    print('PRC by AdaBoost trained on generated data is', average_precision_score(y_test, y_predAB)) # 1.0


    if save_results:
        method = os.path.join(Results_PATH, 'Cervical_separate_generators_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s'
                              % (mini_batch_size, input_size, hidden_size_1, hidden_size_2))  # save with the label 1 setup
        np.save(method + '_PRC.npy', ROC_ours)
        np.save(method + '_ROC.npy', PRC_ours)

    # return ROC, PRC


    """ visualization of learned input features """

    # plt.figure(1)

    # visualization of features
    plt.figure(1)
    ax = plt.subplot(211)
    which_feature_to_plot = 0
    sns.distplot(X_train_neg[:,which_feature_to_plot], bins=50, label='Normal', kde=False)
    sns.distplot(X_train_pos[:,which_feature_to_plot], bins=50, label='Cancer', kde=False)
    ax.set_xlabel('')

    cn = feature_names[which_feature_to_plot]
    ax.set_title('Histogram of feature from real data: ' + str(cn))
    plt.legend()

    ax = plt.subplot(212)
    sns.distplot(generated_samples_neg[:,which_feature_to_plot], bins=50, label='Normal', kde=False)
    sns.distplot(generated_samples_pos[:,which_feature_to_plot], bins=50, label='Cancer', kde=False)
    ax.set_xlabel('')

    cn = feature_names[which_feature_to_plot]
    ax.set_title('Histogram of feature from generated data: ' + str(cn))
    plt.legend()
    plt.show()

    # Select the features.

    # plt.figure(figsize=(12, 120))
    # gs = gridspec.GridSpec(30, 1)
    #
    # for i, cn in enumerate(feature_names):
    #     ax = plt.subplot(gs[i])
    #     sns.distplot(dataset[cn][dataset.Class == 0], bins=50, label='Normal')
    #     sns.distplot(dataset[cn][dataset.Class == 1], bins=50, label='Fraud')
    #     ax.set_xlabel('')
    #     ax.set_title('Histogram of feature: ' + str(cn))
    #     plt.legend()
    # plt.show()


if __name__ == '__main__':

    main()


# if __name__ == '__main__':
#     ROCs = []
#     PRCs = []
#
#     from sklearn.model_selection import ParameterGrid
#     grid = ParameterGrid({
#                             "f":        [300,500],
#
#                             "b_0":      [100,200,300],
#                             "i_0":      [300,400,500],
#                             "lh1_0":    [300,400,500],
#                             "lh2_0":    [200,300,400],
#                             "e_0":      [300, 500,700,1000],
#
#                             "b_1":      [20,30,50],
#                             "i_1":      [10,15,20],
#                             "lh1_1":    [3,5,7],
#                             "lh2_1":    [3,5,7],
#                             "e_1":      [300, 500, 700, 1000]
#                           })
#
#     for params in grid:
#         print("*"*100)
#         print(params)
#         for i in range(3):
#             roc, prc = main(params["f"],
#                             params["b_0"], params["i_0"], params["lh1_0"], params["lh2_0"], params["e_0"],
#                             params['b_1'], params["i_1"], params["lh1_1"], params["lh2_1"], params['e_1'])
#                 #explanation
#                 #main(features_num,
#                 # batch_cl0, input_cl0, hidden1_cl0, hidden2_cl0, epochs_num_cl0,
#                 # batch_cl1, input_cl1, hidden1_cl1, hidden2_cl1,    epochs_num_cl1):


