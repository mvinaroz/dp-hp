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

# import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')

import os

from sdgym import load_dataset

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
            # self.num_rest_columns = num_rest_columns

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

            output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])
            output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])

            output_combined = torch.cat((output_numerical, output_categorical),1)

            return output_combined


# def main(features_num, batch_size, input_layer, hidden1, hidden2, epochs_num, input_dim):
# def main(n_features_arg2, mini_batch_arg2, how_many_epochs_arg2):
def main():

    ##################
    # parameters
    seed_number = 0
    n_features_arg2 = 100
    mini_batch_arg2 = 0.5
    how_many_epochs_arg2 = 1000
    is_private = True

    dataset = "adult"

    print("dataset is", dataset)
    print(socket.gethostname())
    if 'g0' not in socket.gethostname():
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

    LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model.fit(X_train, y_train)  # training on synthetic data
    pred = LR_model.predict(X_test)  # test on real data
    print('ROC on real test data from Logistic regression is', roc_auc_score(y_test, pred))  # 0.64
    print('PRC on real test data from Logistic regression is', average_precision_score(y_test, pred)) # 0.41

    # one-hot encoding of labels.
    n, input_dim = X_train.shape
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = onehot_encoder.fit_transform(y_train)

    """ specifying random fourier features """

    idx_rp = np.random.permutation(n)
    num_data_pt_to_discard = 10
    idx_to_discard = idx_rp[0:num_data_pt_to_discard]
    idx_to_keep = idx_rp[num_data_pt_to_discard:]

    sigma_array = np.zeros(num_numerical_inputs)
    for i in np.arange(0,num_numerical_inputs):
        med = util.meddistance(np.expand_dims(X_train[idx_to_discard,i],1))
        sigma_array[i] = med
    if np.var(sigma_array)>100:
        print('we will use separate frequencies for each column of numerical features')
        sigma2 = sigma_array**2
        sigma2[sigma2 == 0] = 0.1
        # sigma2 = np.mean(sigma2)
    else:
        # median heuristic to choose the frequency range
        med = util.meddistance(X_train[idx_to_discard, 0:num_numerical_inputs])
        sigma2 = med ** 2

    # med = util.meddistance(X_train[idx_to_discard, 0:num_numerical_inputs])
    # # sigma2 = med ** 2

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
    input_size = 2 + 1
    hidden_size_1 = input_dim
    hidden_size_2 = np.int(np.round(1.2* input_dim))
    # input_size = 10 + 1
    # hidden_size_1 = 4 * input_dim
    # hidden_size_2 = 2 * input_dim
    output_size = input_dim
    how_many_epochs = how_many_epochs_arg2

    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                                 output_size=output_size, num_categorical_inputs=num_categorical_inputs,
                                 num_numerical_inputs=num_numerical_inputs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    how_many_iter = np.int(n / mini_batch_size)
    training_loss_per_epoch = np.zeros(how_many_epochs)

    #######################################################33

    if is_private:
        # desired privacy level
        epsilon = 1.0
        delta = 1e-5
        privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta)
        print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

        sensitivity = 2 / n
        noise_std_for_privacy = privacy_param['sigma'] * sensitivity

    """ computing mean embedding of true data """

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
            emb2_numerical = RFF_Gauss(n_features, numerical_samps, W_freq)

            categorical_samps = outputs[:, num_numerical_inputs:]
            emb2_categorical = categorical_samps /(torch.sqrt(torch.Tensor([num_categorical_inputs]))).to(device) # 8

            emb2_input_features = torch.cat((emb2_numerical, emb2_categorical), 1)

            generated_labels = onehot_encoder.fit_transform(label_input.cpu().detach().numpy()) #[1008]
            emb2_labels = Feature_labels(torch.Tensor(generated_labels), weights)
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)
            # print('emb2 is', mean_emb2)

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
    # output_categorical = torch.round(output_categorical)

    output_combined = torch.cat((output_numerical, output_categorical), 1)

    generated_input_features_final = output_combined.cpu().detach().numpy()
    generated_labels_final = label_input.cpu().detach().numpy()


    LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model_ours.fit(generated_input_features_final, generated_labels_final)  # training on synthetic data
    pred_ours = LR_model_ours.predict(X_test)  # test on real data

    print('is private?', is_private)
    print('ROC ours is', roc_auc_score(y_test, pred_ours))
    print('PRC ours is', average_precision_score(y_test, pred_ours))


if __name__ == '__main__':
    main()

    # is_private = True

# if __name__ == '__main__':
#     print("intrusion")
#     how_many_epochs_arg=[1000, 2000]
#     n_features_arg = [50, 100, 300, 500, 1000, 5000, 10000]
#     mini_batch_arg = [0.01, 0.02, 0.05, 0.1, 0.5]
#     grid = ParameterGrid({"n_features_arg": n_features_arg, "mini_batch_arg": mini_batch_arg, "how_many_epochs_arg": how_many_epochs_arg})
#     for elem in grid:
#         print(elem)
#         for ii in range(1):
#             main(elem["n_features_arg"], elem["mini_batch_arg"], elem["how_many_epochs_arg"])
