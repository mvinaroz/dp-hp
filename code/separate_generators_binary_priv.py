
# """ for generating input features for each class separately"""
# Mijung wrote on Jan 24, 2020

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import socket
from sdgym import load_dataset
import argparse


import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from autodp import privacy_calibrator

import warnings
warnings.filterwarnings('ignore')

import os

#Results_PATH = "/".join([os.getenv("HOME"), "condMMD/"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

args=argparse.ArgumentParser()
args.add_argument("--dataset")
arguments=args.parse_args()
print("arg", arguments.dataset)

############################### kernels to use ###############################
""" we use the random fourier feature representation for Gaussian kernel """

def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

    W = torch.Tensor(W).to(device)
    X = X.to(device)

    XWT = torch.mm(X, torch.t(W)).to(device)
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
    return Z

############################### end of kernels ###############################

############################### generative models to use ###############################
""" two types of generative models depending on the type of features in a given dataset """

class Generative_Model_homogeneous_data(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
            super(Generative_Model_homogeneous_data, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size

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


class Generative_Model_heterogeneous_data(nn.Module):

            def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, num_categorical_inputs, num_numerical_inputs):
                super(Generative_Model_heterogeneous_data, self).__init__()

                self.input_size = input_size
                self.hidden_size_1 = hidden_size_1
                self.hidden_size_2 = hidden_size_2
                self.output_size = output_size
                self.num_numerical_inputs = num_numerical_inputs
                self.num_categorical_inputs = num_categorical_inputs

                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
                self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
                self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
                self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                hidden = self.fc1(x)
                relu = self.relu(self.bn1(hidden))
                output = self.fc2(relu)
                output = self.relu(self.bn2(output))
                output = self.fc3(output)

                output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
                output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
                output_combined = torch.cat((output_numerical, output_categorical), 1)

                return output_combined

############################### end of generative models ###############################




############################### beginning of main script ######################

def main(dataset, n_features_arg, mini_batch_size_arg, how_many_epochs_arg):
    seed_number=0
    random.seed(seed_number)

    is_private = True

    ############################### data loading ##################################

    if dataset=='epileptic':
        print("epileptic seizure recognition dataset") # this is homogeneous

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

    elif dataset=="credit":

        print("Creditcard fraud detection dataset") # this is homogeneous

        if 'g0' not in socket.gethostname():
            data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")
        else:
            # (1) load data
            data = pd.read_csv(
                '/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Kaggle_Credit/creditcard.csv')

        feature_names = data.iloc[:, 1:30].columns
        target = data.iloc[:1, 30:].columns

        data_features = data[feature_names]
        data_target = data[target]
        print(data_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        raw_input_features = data_features.values
        raw_labels = data_target.values.ravel()

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = 0.5
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.90,
                                                            test_size=0.10, random_state=0)

        data_samps = X_train
        y_labels = y_train

    elif dataset=='census':

        print("census dataset") # this is heterogenous

        print(socket.gethostname())
        if 'g0' not in socket.gethostname():
            data = np.load("../data/real/census/train.npy")
        else:
            data = np.load(
                "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/real/census/train.npy")

        numerical_columns = [0, 5, 16, 17, 18, 29, 38]
        ordinal_columns = []
        categorical_columns = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
        n_classes = 2

        data = data[:, numerical_columns + ordinal_columns + categorical_columns]

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        raw_input_features = data[:, :-1]
        raw_labels = data[:, -1]
        print('raw input features', raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = 1.0
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)

        data_samps = X_train
        y_labels = y_train

    elif dataset=='cervical':

        print("dataset is", dataset) # this is heterogenous
        print(socket.gethostname())
        if 'g0' not in socket.gethostname():
            df = pd.read_csv("../data/Cervical/kag_risk_factors_cervical_cancer.csv")
        else:
            df = pd.read_csv(
                "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")
            print("Loaded Cervical")

        # df.head()

        df_nan = df.replace("?", np.float64(np.nan))
        df_nan.head()

        #df1 = df_nan.convert_objects(convert_numeric=True)
        df1=df.apply(pd.to_numeric, errors="coerce")

        df1.columns = df1.columns.str.replace(' ', '')  # deleting spaces for ease of use

        """ this is the key in this data-preprocessing """
        df = df1[df1.isnull().sum(axis=1) < 10]

        numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse', 'Numofpregnancies', 'Smokes(years)',
                        'Smokes(packs/year)', 'HormonalContraceptives(years)', 'IUD(years)', 'STDs(number)',
                        'STDs:Numberofdiagnosis',
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
            feature_mean = round(df[feature].median(), 1)
            df[feature] = df[feature].fillna(feature_mean)

        for feature in categorical_df:
            #df[feature] = df[feature].convert_objects(convert_numeric=True).fillna(0.0)
            df[feature] = df[feature].fillna(0.0)


        target = df['Biopsy']
        # feature_names = df.iloc[:, :-1].columns
        inputs = df[feature_names]
        print('raw input features', inputs.shape)

        X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.80, test_size=0.20,
                                                            random_state=seed_number)  # 60% training and 40% test


        y_labels = y_train.values.ravel()  # X_train_pos
        data_samps = X_train.values

    elif dataset=='adult':

        print("dataset is", dataset) # this is heterogenous
        print(socket.gethostname())
        #if 'g0' not in socket.gethostname():
        data, categorical_columns, ordinal_columns = load_dataset('adult')
        # else:

        """ some specifics on this dataset """
        numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
        n_classes = 2

        data = data[:, numerical_columns + ordinal_columns + categorical_columns]

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        inputs = data[:, :-1]
        target = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
                                                            random_state=seed_number)

        y_labels = y_train
        data_samps = X_train

    elif dataset=='isolet':

        print("isolet dataset")
        print(socket.gethostname())
        if 'g0' not in socket.gethostname():
            data_features_npy = np.load('../data/Isolet/isolet_data.npy')
            data_target_npy = np.load('../data/Isolet/isolet_labels.npy')
        else:
            # (1) load data
            data_features_npy = np.load(
                '/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Isolet/isolet_data.npy')
            data_target_npy = np.load(
                '/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR//data/Isolet/isolet_labels.npy')

        print(data_features_npy.shape)
        # dtype = [('Col1', 'int32'), ('Col2', 'float32'), ('Col3', 'float32')]
        values = data_features_npy
        index = ['Row' + str(i) for i in range(1, len(values) + 1)]

        values_l = data_target_npy
        index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

        data_features = pd.DataFrame(values, index=index)
        data_target = pd.DataFrame(values_l, index=index_l)

        X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30,
                                                            random_state=0)

        # unpack data
        data_samps = X_train.values
        y_labels = y_train.values.ravel()

    ############################### end of data loading ##################################

    # specify heterogeneous dataset or not
    heterogeneous_datasets = ['cervical', 'adult', 'census']
    homogeneous_datasets = ['epileptic','credit','isolet']

    ###########################################################################

    # As a reference, we first test logistic regression on the real data
    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(X_train, y_labels)  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data

    print('ROC on real test data is', roc_auc_score(y_test, pred))
    print('PRC on real test data is', average_precision_score(y_test, pred))

    ###########################################################################

    n_classes = 2

    #################### split data into two classes for separate training of each generator ########################333

    X_train_pos =  data_samps[y_labels==1,:]
    y_train_pos = y_labels[y_labels==1]

    X_train_neg = data_samps[y_labels==0,:]
    y_train_neg = y_labels[y_labels == 0]


    # random Fourier features
    n_features = n_features_arg


    print('start training')



    for which_class in range(n_classes):

        if which_class==1:

            n, input_dim = X_train_pos.shape

            """ we use 10 datapoints to compute the median heuristic (then discard), and use the rest for training """
            idx_rp = np.random.permutation(n)
            num_data_pt_to_discard = 10
            idx_to_discard = idx_rp[0:num_data_pt_to_discard]
            idx_to_keep = idx_rp[num_data_pt_to_discard:]

            med = util.meddistance(X_train_pos[idx_to_discard, :])
            sigma2 = med ** 2

            X = X_train_pos[idx_to_keep, :]
            del X_train_pos
            y_train_pos = y_train_pos[idx_to_keep]
            n = idx_to_keep.shape[0]

            print('num of datapoints for class 1', n)


        else: # class==0

            print('training a generator for data corresponding to label 0')

            n, input_dim = X_train_neg.shape

            """ we use 10 datapoints to compute the median heuristic (then discard), and use the rest for training """
            idx_rp = np.random.permutation(n)
            num_data_pt_to_discard = 10
            idx_to_discard = idx_rp[0:num_data_pt_to_discard]
            idx_to_keep = idx_rp[num_data_pt_to_discard:]

            med = util.meddistance(X_train_neg[idx_to_discard, :])
            sigma2 = med ** 2

            X = X_train_neg[idx_to_keep, :]
            y_train_neg = y_train_neg[idx_to_keep]
            n = idx_to_keep.shape[0]

            print('num of datapoints for class 0', n)


        """ specify a model depending on the data type """
        # model specifics
        mini_batch_size = np.int(np.round(mini_batch_size_arg * n))
        print("minibatch: ", mini_batch_size)
        input_size = 5 + 1
        hidden_size_1 = 2 * input_dim
        hidden_size_2 = 1 * input_dim
        output_size = input_dim

        if dataset in homogeneous_datasets:

            model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                      hidden_size_2=hidden_size_2,
                                                      output_size=output_size).to(device)

        elif dataset in heterogeneous_datasets:

            model = Generative_Model_heterogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                        hidden_size_2=hidden_size_2,
                                                        output_size=output_size,
                                                        num_categorical_inputs=num_categorical_inputs,
                                                        num_numerical_inputs=num_numerical_inputs).to(device)
        else:
            print('sorry, please enter the name of your dataset either in homogeneous_dataset or heterogeneous_dataset list ')


        # define details for training
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        how_many_epochs = how_many_epochs_arg
        how_many_iter = np.int(n / mini_batch_size)
        training_loss_per_epoch = np.zeros(how_many_epochs)

        draws = n_features // 2
        W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

        ######################################################

        if is_private:
            # desired privacy level
            epsilon = 1.0
            delta = 1e-5
            privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=2) # split the privacy cost for training two generators
            print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

            sensitivity = 2 / n
            noise_std_for_privacy = privacy_param['sigma'] * sensitivity

        """ computing mean embedding of  true data """
        if dataset in homogeneous_datasets:

            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(X), W_freq)
            mean_emb1 = torch.mean(emb1_input_features, 0)

        elif dataset in heterogeneous_datasets:

            numerical_input_data = X[:, 0:num_numerical_inputs]
            emb1_numerical = (RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)).to(device)

            categorical_input_data = X[:, num_numerical_inputs:]
            emb1_categorical = (torch.Tensor(categorical_input_data) / np.sqrt(num_categorical_inputs)).to(device)

            emb1_input_features = torch.cat((emb1_numerical, emb1_categorical), 1)
            mean_emb1 = torch.mean(emb1_input_features, 0)


        if is_private:
            noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
            noise = noise.to(device)
            mean_emb1 = mean_emb1 + noise

        for epoch in range(how_many_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(how_many_iter):

                # zero the parameter gradients
                optimizer.zero_grad()
                input_to_model = torch.randn((mini_batch_size, input_size)).to(device)
                outputs = model(input_to_model)

                if dataset in homogeneous_datasets:

                    """ computing mean embedding of generated samples """
                    emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                    mean_emb2 = torch.mean(emb2_input_features, 0)

                elif dataset in heterogeneous_datasets:

                    numerical_samps = outputs[:, 0:num_numerical_inputs]
                    emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq)

                    categorical_samps = outputs[:, num_numerical_inputs:]
                    emb2_categorical = categorical_samps / (torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device)  # 8

                    emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)

                    mean_emb2 = torch.mean(emb2_input_features, 0)


                loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 100 == 0:
                print('epoch # and running loss are ', [epoch, running_loss])
            training_loss_per_epoch[epoch] = running_loss


        """ now generated samples using the trained generator """
        input_to_model = torch.randn((n, input_size)).to(device)
        outputs = model(input_to_model)
        samp_input_features = outputs

        if which_class==1:
            generated_samples_pos = samp_input_features.cpu().detach().numpy()

        else:
            generated_samples_neg = samp_input_features.cpu().detach().numpy()



    ##################################### When both training routines are over #####################################

    # mix data for positive and negative labels
    generated_input_features = np.concatenate((generated_samples_pos, generated_samples_neg), axis=0)
    corresponding_labels = np.concatenate((y_train_pos, y_train_neg))

    idx_shuffle = np.random.permutation(corresponding_labels.shape[0])
    shuffled_x_train = generated_input_features[idx_shuffle, :]
    shuffled_y_train = corresponding_labels[idx_shuffle]

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(shuffled_x_train, shuffled_y_train) # training on synthetic data
    pred = LR_model.predict(X_test) # test on real data

    roc = roc_auc_score(y_test, pred)
    prc = average_precision_score(y_test, pred)


    print('is private?', is_private)
    print('ROC ours is', roc)
    print('PRC ours is', prc)
    print('n_features are ', n_features)


    return roc, prc


if __name__ == '__main__':

    #epileptic, credit, census, cervical, adult, isolet

    #for dataset in ["epileptic", "credit", "census", "cervical", "adult", "isolet"]:
    # for dataset in [arguments.dataset]:
    for dataset in ["credit"]:
        print("\n\n")
        how_many_epochs_arg = [2000]
        n_features_arg = [1000, 5000, 10000, 50000, 80000, 100000]
        mini_batch_arg = [1.0]

        grid = ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg,
                              "how_many_epochs_arg": how_many_epochs_arg})
        for elem in grid:
            print(elem)
            prc_arr = []; roc_arr = []
            repetitions = 1
            for ii in range(repetitions):
                roc, prc = main(dataset, elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"])
                roc_arr.append(roc)
                prc_arr.append(prc)
            print("Average ROC: ", np.mean(roc_arr)); print("Avergae PRC: ", np.mean(prc_arr))




