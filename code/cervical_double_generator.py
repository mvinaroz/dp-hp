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



import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

import os

from sdgym import load_dataset

data, categorical_columns, ordinal_columns = load_dataset('adult')

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

user='anon_k'
save_results = False

if user=='anon_m':
    Results_PATH = "/".join([os.getenv("HOME"), "separate_Isolet/"])
elif user =='anon_k':
    Results_PATH = "/home/kamil/Desktop/Dropbox/Current_research/privacy/DPDR/results/separate_isolet"

        #n_features - random Fourier features
        #X - real/generated data
        #W - some random features (half of n_features)
def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W).to(device)
    X = X.to(device)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]).to(device))
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


def main(features_num, batch_cl0,  input_cl0, hidden1_cl0, hidden2_cl0, epochs_num_cl0, batch_cl1, input_cl1, hidden1_cl1, hidden2_cl1, epochs_num_cl1):

    #input_dim - number of input features in data
    #n  - number of samples

    ######################3


    data, categorical_columns, ordinal_columns = load_dataset('adult')



    ###############3

    random.seed(0)


    print(socket.gethostname())
    if'g0' not in socket.gethostname():
        data_nan=pd.read_csv("../data/Cervical/kag_risk_factors_cervical_cancer.csv")
    else:
        data_nan=pd.read_csv("/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")


    preprocessing='removdal'

    numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse', 'Numofpregnancies', 'Smokes(years)',
                        'Smokes(packs/year)', 'HormonalContraceptives(years)', 'IUD(years)', 'STDs(number)'
                        #]
                        ,'STDs:Timesincefirstdiagnosis', 'STDs:Timesincelastdiagnosis']
    categorical_df = ['Smokes', 'HormonalContraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
                          'STDs:vulvo-perinealcondylomatosis', 'STDs:syphilis', 'STDs:pelvicinflammatorydisease',
                          'STDs:genitalherpes', 'STDs:AIDS', 'STDs:cervicalcondylomatosis',
                          'STDs:molluscumcontagiosum', 'STDs:HIV', 'STDs:HepatitisB', 'STDs:HPV', 'STDs:Numberofdiagnosis',
                          'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']


    data = data_nan.replace("?", np.nan)
    feature_names = data.iloc[:, :-1].columns


    #data = data_con.convert_objects(convert_numeric=True)


    #######################################
    if preprocessing=='removal':

        df1 = data_nan.convert_objects(convert_numeric=True)
        #df1=pd.to_numeric(data_nan)

        df1.columns = df1.columns.str.replace(' ', '')  # deleting spaces for ease of use

        """ this is the key in this data-preprocessing """
        data = df1[df1.isnull().sum(axis=1) < 3]



        for feature in numerical_df:
            # print(feature, '', df[feature].convert_objects(convert_numeric=True).mean())
            # print(df[feature])
            feature_mean = round(data[feature].convert_objects(convert_numeric=True).mean(), 1)
            data[feature] = data[feature].fillna(feature_mean)
            # print(df[feature])

        for feature in categorical_df:
            data[feature] = data[feature].convert_objects(convert_numeric=True).fillna(0.0)

        data_target = data['Biopsy']
        data_features = data.iloc[:, :-1]

        data_numerical = data[numerical_df]
        data_categorical = data[categorical_df]


    ################################

    else:


        target = data.iloc[:, -1:].columns

        #print(feature_names)


        for i in feature_names:
            #print (i)

            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            data[[i]] = imputer.fit_transform(data[[i]])

        data_features = data[feature_names]
        data_target = data[target]

        numerical_df2 = feature_names[[0, 1, 2, 3, 5, 6, 8, 10, 12, 26, 27]]
        categorical_df2 = feature_names[
            [4, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34]]

        data_numerical = data[numerical_df2]
        data_categorical = data[categorical_df2]





    #print(data_features)

    #print(data_target)
    #print("asa")
    #print(np.sum(data_target))

    # for index, row in data_target.iterrows():
    #     print(row)

    # for i, row in data_target.iterrows():
    #   if data_target.at[i,'y']!=1:
    #     #print(data_target.at[i,'y'])
    #     data_target.at[i,'y'] = 0

    ###################################################################################################################3

    #optionsfor data_features:
    # data - all
    # data_categorical
    # data_numerical

    print("*" * 100)
    for data_used in ["all", "numerical", "categorical"]:

        if data_used=="all":
            data_features=data_features
        elif data_used=="numerical":
            data_features=data_numerical
        elif data_used=="categorical":
            data_features=data_categorical

        print("data used: " , data_used)


        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.60, test_size=0.40,
                                                            random_state=0)

        # test logistic regression on the real data
        LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        #LR_model = DecisionTreeClassifier()

        LR_model.fit(X_train, y_train.values.ravel())  # training on synthetic data
        pred = LR_model.predict(X_test)  # test on real data

        print('ROC on real test data is', roc_auc_score(y_test, pred))
        print('PRC on real test data is', average_precision_score(y_test, pred), '\n')

        y_labels = y_train.values.ravel()  # X_train_pos
        X_train = X_train.values

        n_tot = X_train.shape[0]

        X_train_pos = X_train[y_labels == 1, :]
        y_train_pos = y_labels[y_labels == 1]

        X_train_neg = X_train[y_labels == 0, :]
        y_train_neg = y_labels[y_labels == 0]

        ###############################################################3
        # train generator

        n_classes = 2

        for which_class in range(n_classes):

            print("Class: ", which_class)

            if which_class == 1:
                # First train for positive label
                n, input_dim = X_train_pos.shape
                data_samps = X_train_pos
                del X_train_pos
            else:
                n, input_dim = X_train_neg.shape
                data_samps = X_train_neg
                del X_train_neg


            ##################

            #training a Generator via minimizing MMD
            # try more random features with a larger batch size

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

            ############

            # test how to use RFF for computing the kernel matrix
            # same for the real and generated data
            idx_rp = np.random.permutation(np.min([n, 10000]))
            med = util.meddistance(data_samps[idx_rp, :])
            del idx_rp
            sigma2 = med ** 2
            print('length scale from median heuristic is', sigma2)

            # random Fourier features
            n_features = features_num  # 20000


            draws = n_features // 2
            W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)


            #####

            """ computing mean embedding of true data """
            #n_features - random Fourier features, e.g. 400
            #data_samps - real/generated data, e.g. 498x11
            #W_freq - some random features (half of n_features), e.g [400, 498]
            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)
            mean_emb1 = torch.mean(emb1_input_features, 0) #e.g. [400]

            del data_samps
            del emb1_input_features

            # plt.plot(mean_emb1, 'b')

            ############################################################33
            # start training

            model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                                     output_size=output_size).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            how_many_iter = np.int(n / mini_batch_size)

            training_loss_per_epoch = np.zeros(how_many_epochs)

            print('Starting Training')

            for epoch in range(how_many_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                # annealing_rate =

                for i in range(how_many_iter):
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    input_to_model = torch.randn((mini_batch_size, input_size))
                    input_to_model = input_to_model.to(device)
                    outputs = model(input_to_model)

                    """ computing mean embedding of generated samples """
                    emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                    mean_emb2 = torch.mean(emb2_input_features, 0)

                    loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()

                #print('epoch # and running loss are ', [epoch, running_loss])
                training_loss_per_epoch[epoch] = running_loss

            #################################3
            # generate samples

            """ now generated samples using the trained generator """
            input_to_model = torch.randn((n, input_size))
            input_to_model=input_to_model.to(device)
            outputs = model(input_to_model)
            samp_input_features = outputs

            if which_class == 1:
                generated_samples_pos = samp_input_features.cpu().detach().numpy()

                # save results
                method = os.path.join(Results_PATH, 'Isolet_pos_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s' % (
                    mini_batch_size, input_size, hidden_size_1, hidden_size_2))
                # np.save(method + '_loss.npy', training_loss_per_epoch)
                # np.save(method + '_input_feature_samps.npy', generated_samples_pos)

            else:
                generated_samples_neg = samp_input_features.cpu().detach().numpy()

                # save results
                method = os.path.join(Results_PATH, 'Isolet_neg_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s' % (
                    mini_batch_size, input_size, hidden_size_1, hidden_size_2))
                # np.save(method + '_loss.npy', training_loss_per_epoch)
                # np.save(method + '_input_feature_samps.npy', generated_samples_neg)

        # plt.figure(3)
        # plt.plot(training_loss_per_epoch)
        # plt.title('MMD as a function of epoch')
        # plt.yscale('log')
        #
        # plt.figure(4)
        # plt.plot(mean_emb1, 'b')
        # plt.plot(mean_emb2.detach().numpy(), 'r--')
        print("*" * 100)

    print("\nall generated data: ")
    # mix data for positive and negative labels
    generated_input_features = np.concatenate((generated_samples_pos, generated_samples_neg), axis=0)
    corresponding_labels = np.concatenate((y_train_pos, y_train_neg))

    idx_shuffle = np.random.permutation(n_tot)
    shuffled_x_train = generated_input_features[idx_shuffle, :]
    shuffled_y_train = corresponding_labels[idx_shuffle]

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(shuffled_x_train, shuffled_y_train)  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data

    ROC = roc_auc_score(y_test, pred)
    PRC = average_precision_score(y_test, pred)

    print('ROC is', ROC)
    print('PRC is', PRC)

    if save_results:
        method = os.path.join(Results_PATH, 'Isolet_separate_generators_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s'
                              % (mini_batch_size, input_size, hidden_size_1, hidden_size_2))  # save with the label 1 setup
        np.save(method + '_PRC.npy', ROC)
        np.save(method + '_ROC.npy', PRC)

    return ROC, PRC


if __name__ == '__main__':

    run='single'

    if run=='single':
        main(400, 100, 300, 300, 200, 300, 20, 10, 3, 3, 300)

    elif run == 'grid':



        ROCs = []
        PRCs = []

        from sklearn.model_selection import ParameterGrid
        grid = ParameterGrid({
                                "f":        [300,500],

                                "b_0":      [100,200,300],
                                "i_0":      [300,400,500],
                                "lh1_0":    [300,400,500],
                                "lh2_0":    [200,300,400],
                                "e_0":      [300, 500,700,1000],

                                "b_1":      [20,30,50],
                                "i_1":      [10,15,20],
                                "lh1_1":    [3,5,7],
                                "lh2_1":    [3,5,7],
                                "e_1":      [300, 500, 700, 1000]
                              })

        for params in grid:
            print("*"*100)
            print(params)
            for i in range(3):
                roc, prc = main(params["f"],
                                params["b_0"], params["i_0"], params["lh1_0"], params["lh2_0"], params["e_0"],
                                params['b_1'], params["i_1"], params["lh1_1"], params["lh2_1"], params['e_1'])
                    #explanation
                    #main(features_num,
                    # batch_cl0, input_cl0, hidden1_cl0, hidden2_cl0, epochs_num_cl0,
                    # batch_cl1, input_cl1, hidden1_cl1, hidden2_cl1,    epochs_num_cl1):


