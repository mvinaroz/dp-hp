# for training a single generator for multi-labeled dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import socket

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid

import pandas as pd
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import Imputer
# from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree

import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')

import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

from autodp import privacy_calibrator

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#torch.cuda.empty_cache()
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'


###################################################
# n_features - random Fourier features
# X - real/generated data
# W - some random features (half of n_features)

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


def Feature_labels(labels, weights):

    weights = torch.Tensor(weights)
    weights = weights.to(device)

    labels = labels.to(device)

    weighted_labels_feature = labels/weights

    return weighted_labels_feature


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


# def main(features_num, batch_size, input_layer, hidden1, hidden2, epochs_num, input_dim):
def main(n_features_arg2, mini_batch_arg2, how_many_epochs_arg2):
#def main():

    ##################
    # parameters
    seed_number = 0
    is_private = True
    #n_features_arg2 = 500
    #mini_batch_arg2 = 0.5
    #how_many_epochs_arg2 = 1000

    print("census dataset")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname():
        data=np.load("../data/real/census/train.npy")
    else:
        data = np.load(
            "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/real/census/train.npy")

    numerical_columns= [0,5, 16, 17, 18, 29, 38]
    ordinal_columns = []
    categorical_columns = [1,2,3,4,6,7,8,9,10,11,12,13,14,15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
    n_classes = 2

    data = data[:, numerical_columns + ordinal_columns + categorical_columns]

    num_numerical_inputs = len(numerical_columns)
    num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

    raw_input_features = data[:, :-1]
    raw_labels = data[:, -1]
    print('raw input features', raw_input_features.shape)

    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels==0
    idx_positive_label = raw_labels==1

    pos_samps_input = raw_input_features[idx_positive_label,:]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label,:]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = 0.2
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label)*under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep,:]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))


    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80, test_size=0.20, random_state=seed_number)

    ############################################################

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(X_train, y_train)  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data

    print('ROC on real test data is', roc_auc_score(y_test, pred))
    print('PRC on real test data is', average_precision_score(y_test, pred))

    # one-hot encoding of labels.
    n, input_dim = X_train.shape
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = onehot_encoder.fit_transform(y_train)


    if is_private:
        # desired privacy level
        epsilon = 1.0
        delta = 1e-5
        privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta)
        print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

        sensitivity = 2 / n
        noise_std_for_privacy = privacy_param['sigma'] * sensitivity

    """ specifying random fourier features """

    idx_rp = np.random.permutation(n)
    num_data_pt_to_discard = 10
    idx_to_discard = idx_rp[0:num_data_pt_to_discard]
    idx_to_keep = idx_rp[num_data_pt_to_discard:]

    med = util.meddistance(X_train[idx_to_discard, 0:num_numerical_inputs])
    sigma2 = med ** 2

    X_train = X_train[idx_to_keep,:]
    true_labels = true_labels[idx_to_keep,:]
    n = X_train.shape[0]

    n_features = n_features_arg2
    draws = n_features // 2

    # random fourier features for numerical inputs only
    W_freq = np.random.randn(draws, num_numerical_inputs) / np.sqrt(sigma2)

    """ specifying ratios of data to generate depending on the class lables """
    unnormalized_weights = np.sum(true_labels,0)
    weights = unnormalized_weights/np.sum(unnormalized_weights)
    print('weights are', weights)

    """ specifying the model """
    mini_batch_size = np.int(np.round(mini_batch_arg2*n)); print("minibatch: ", mini_batch_size)
    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 = 2 * input_dim
    output_size = input_dim
    how_many_epochs = how_many_epochs_arg2


    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                                 output_size=output_size, num_categorical_inputs=num_categorical_inputs,
                                 num_numerical_inputs=num_numerical_inputs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    how_many_iter = np.int(n / mini_batch_size)
    training_loss_per_epoch = np.zeros(how_many_epochs)


    """ computing mean embedding of subsampled true data """
    numerical_input_data = X_train[:, 0:num_numerical_inputs]
    emb1_numerical = (RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)).to(device)

    categorical_input_data = X_train[:, num_numerical_inputs:]
    emb1_categorical = (torch.Tensor(categorical_input_data) / np.sqrt(num_categorical_inputs)).to(device)

    emb1_input_features = torch.cat((emb1_numerical, emb1_categorical), 1)

    emb1_labels = Feature_labels(torch.Tensor(true_labels), weights)
    outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
    mean_emb1 = torch.mean(outer_emb1, 0)

    if is_private:
        noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
        noise = noise.to(device)
        mean_emb1 = mean_emb1 + noise

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            """ computing mean embedding of generated data """
            # zero the parameter gradients
            optimizer.zero_grad()

            # (1) generate labels
            label_input = torch.multinomial(torch.Tensor([weights]), mini_batch_size, replacement=True).type(torch.FloatTensor)
            label_input = label_input.transpose_(0,1)
            label_input = label_input.to(device)

            # (2) generate corresponding features
            feature_input = torch.randn((mini_batch_size, input_size-1)).to(device)
            input_to_model = torch.cat((feature_input, label_input), 1)
            outputs = model(input_to_model)

            # (3) compute the embeddings of those
            numerical_samps = outputs[:, 0:num_numerical_inputs] #[4553,6]
            emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq) #W_freq [n_features/2,6], n_features=10000

            categorical_samps = outputs[:, num_numerical_inputs:] #[4553,8]
            emb2_categorical = categorical_samps /(torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device) # 8

            emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)

            generated_labels = onehot_encoder.fit_transform(label_input.cpu().detach().numpy()) #[1008]
            emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights)
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 100 == 0:
            print('epoch # and running loss are ', [epoch, running_loss])
            training_loss_per_epoch[epoch] = running_loss



    #######################################################################33

    """ draw final data samples """

    label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
    label_input = label_input.transpose_(0, 1)
    label_input = label_input.to(device)

    # (2) generate corresponding features
    feature_input = torch.randn((n, input_size - 1)).to(device)
    input_to_model = torch.cat((feature_input, label_input), 1)
    outputs = model(input_to_model)


    # (3) round the categorial features
    output_numerical = outputs[:, 0:num_numerical_inputs]
    output_categorical = outputs[:, num_numerical_inputs:]
    output_categorical = torch.round(output_categorical)

    output_combined = torch.cat((output_numerical, output_categorical), 1)

    generated_input_features_final = output_combined.cpu().detach().numpy()
    generated_labels_final = label_input.cpu().detach().numpy()


    LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model_ours.fit(generated_input_features_final, generated_labels_final)  # training on synthetic data
    pred_ours = LR_model_ours.predict(X_test)  # test on real data

    roc=roc_auc_score(y_test, pred_ours)
    prc=average_precision_score(y_test, pred_ours)

    print('is private?', is_private)
    print('ROC ours is', roc)
    print('PRC ours is', prc)

    return roc, prc


if __name__ == '__main__':
    print("census")
    how_many_epochs_arg=[1000, 2000]
    n_features_arg = [50, 100, 300, 500, 1000, 5000, 10000]
    mini_batch_arg = [0.01, 0.02, 0.05, 0.1, 0.5]

    grid = ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg, "how_many_epochs_arg": how_many_epochs_arg})
    for elem in grid:
        print(elem)
        prc_arr=[]; roc_arr=[]
        repetitions=3
        for ii in range(repetitions):
            roc, prc = main(elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"])
            roc_arr.append(roc); prc_arr.append(prc)
        print("Average ROC: ", np.mean(roc_arr)); print("Avergae PRC: ", np.mean(prc_arr))
