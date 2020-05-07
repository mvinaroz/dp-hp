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
from sklearn.preprocessing import OneHotEncoder

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

def main(dataset, n_features_arg, mini_batch_size_arg, how_many_epochs_arg):
    seed_number=0
    random.seed(seed_number)

    is_private = True


    if dataset=='epileptic': #numeric
        print("epileptic seizure recognition dataset")

        # as a reference, with n_features= 500 and mini_batch_size=1000, after 1000 epochs
        # I get ROC= 0.54, PRC=0.21
        # with different seed, I get ROC=0.5 and PRC 0.2. Similar to before.

        # when I tested n_features = 500 and mini_batch_size=500,
        # I got ROC = 0.6163 and PRC = 0.2668

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

    elif dataset=="credit": #numeric

        print("Creditcard fraud detection dataset")

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
        under_sampling_rate = 0.025
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.90,
                                                            test_size=0.10, random_state=0)

        data_samps = X_train
        y_labels = y_train



    elif dataset=='cervical': #numeric

        print("dataset is", dataset)
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

    elif dataset=='isolet': #numeric

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


    elif dataset=='census': #mixed

        print("census dataset")
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
        under_sampling_rate = 0.2
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)

        data_samps = X_train
        y_labels = y_train

    elif dataset=='adult': #mixed

        print("dataset is", dataset)
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



    #########################################3

        # test logistic regression on the real data
    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(data_samps, y_labels)  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data

    print('ROC on real test data is', roc_auc_score(y_test, pred))
    print('PRC on real test data is', average_precision_score(y_test, pred))

    ###########################################################################3
    ################################################################
    ########################################################

    n_classes = 2
    n, input_dim = data_samps.shape
    # one-hot encoding of labels.
    #n, input_dim = X_train.shape
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = onehot_encoder.fit_transform(y_train)



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
    sigma2 = med ** 2

    # print('length scale from median heuristic is', sigma2)

    data_samps = data_samps[idx_to_keep, :]
    n = idx_to_keep.shape[0]

    #######################################################

    true_labels = np.zeros((n, n_classes))
    idx_1 = y_labels[idx_to_keep] == 1
    idx_0 = y_labels[idx_to_keep] == 0
    true_labels[idx_1, 1] = 1
    true_labels[idx_0, 0] = 1

    # random Fourier features
    n_features = n_features_arg


    ############  """ training a Generator via minimizing MMD """
    # network params


    #mini_batch_size = mini_batch_size_arg
    mini_batch_size = np.int(np.round(mini_batch_size_arg*n)); print("minibatch: ", mini_batch_size)
    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 = 2 * input_dim
    output_size = input_dim

    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    how_many_epochs = how_many_epochs_arg
    how_many_iter = np.int(n / mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    ########
    # MMD

    draws = n_features // 2
    W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    # kernel for labels with weights
    unnormalized_weights = np.sum(true_labels, 0)
    positive_label_ratio = unnormalized_weights[1] / unnormalized_weights[0]

    weights = unnormalized_weights / np.sum(unnormalized_weights)

    ######################################################

    # if is_private:
    #     # desired privacy level
    #     epsilon = 1.0
    #     delta = 1e-5
    #     privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta)
    #     print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])
    #
    #     sensitivity = 2 / n
    #     noise_std_for_privacy = privacy_param['sigma'] * sensitivity

    """ computing mean embedding of  true data """
    emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)
    emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
    outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
    mean_emb1 = torch.mean(outer_emb1, 0)

    # if is_private:
    #     noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
    #     noise = noise.to(device)
    #     mean_emb1 = mean_emb1 + noise

    ###############################################################################3
    #

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):
            # zero the parameter gradients
            optimizer.zero_grad()

            label_input = (1 * (torch.rand((mini_batch_size)) < weights[1])).type(torch.FloatTensor)
            label_input = label_input.to(device)
            feature_input = torch.randn((mini_batch_size, input_size - 1)).to(device)
            input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
            outputs = model(input_to_model)

            """ computing mean embedding of generated samples """
            emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)

            label_input_t = torch.zeros((mini_batch_size, n_classes))
            idx_1 = (label_input == 1.).nonzero()[:, 0]
            idx_0 = (label_input == 0.).nonzero()[:, 0]
            label_input_t[idx_1, 1] = 1.
            label_input_t[idx_0, 0] = 1.

            emb2_labels = Feature_labels(label_input_t, weights)
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if epoch % 100 == 0:
            print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss

    # plt.figure(3)
    # plt.plot(training_loss_per_epoch)
    # plt.title('MMD as a function of epoch')
    # plt.yscale('log')
    #
    # plt.figure(4)
    # plt.subplot(211)
    # plt.plot(mean_emb1[:, 0].cpu(), 'b')
    # plt.plot(mean_emb2[:, 0].cpu().detach().numpy(), 'b--')
    # plt.subplot(212)
    # plt.plot(mean_emb1[:, 1].cpu(), 'r')
    # plt.plot(mean_emb2[:, 1].cpu().detach().numpy(), 'r--')

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
    LR_model.fit(generated_samples, np.argmax(generated_labels, axis=1))  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data

    roc = roc_auc_score(y_test, pred)
    prc = average_precision_score(y_test, pred)


    print('is private?', is_private)
    print('ROC ours is', roc)
    print('PRC ours is', prc)
    print('n_features are ', n_features)


    return roc, prc

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


# to test one setting put only one element array for each variable

if __name__ == '__main__':

    #epileptic, credit, census, cervical, adult, isolet
    #for dataset in ["epileptic", "credit", "census", "cervical", "adult", "isolet"]:
    #for dataset in [arguments.dataset]:
    for dataset in ["cervical"]:
        print("\n\n")
        how_many_epochs_arg = [1000]
        #n_features_arg = #[20, 50, 100, 500, 1000, 5000, 10000, 50000, 80000, 100000]
        n_features_arg = [100000]
        mini_batch_arg = [0.5]

        grid = ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg,
                              "how_many_epochs_arg": how_many_epochs_arg})
        for elem in grid:
            print(elem)
            prc_arr = []; roc_arr = []
            repetitions = 5
            for ii in range(repetitions):
                roc, prc = main(dataset, elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"])
                roc_arr.append(roc)
                prc_arr.append(prc)
            print("Average ROC: ", np.mean(roc_arr)); print("Avergae PRC: ", np.mean(prc_arr))



