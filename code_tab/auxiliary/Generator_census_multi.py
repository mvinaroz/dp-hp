# real
# ROC on real test data from Logistic regression is 0.6205366216950137
# PRC on real test data from Logistic regression is 0.3791835921904054


# lossed for generated
##epoch # and running loss are  [9980, 11.479012489318848]
## negative 14.742877006530762]

#ROC on generated samples using Logistic regression is 0.5147465437788018
#PRC on generated samples using Logistic regression is 0.26281903482106483

#epoch # and running loss are  [9980, 10.444331169128418]
#epoch # and running loss are  [9980, 13.737614631652832]
#ROC on generated samples using Logistic regression is 0.5032258064516129
#PRC on generated samples using Logistic regression is 0.24531711257085265

#epoch # and running loss are  [9980, 10.4443359375]
#epoch # and running loss are  [9980, 13.737520217895508]
#ROC on generated samples using Logistic regression is 0.5509010910420554
#PRC on generated samples using Logistic regression is 0.26801810670635856

#loss doesn't realy reflect the roc

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#torch.cuda.empty_cache()
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'

##################
# parameters
seed_number=1000
dataset="covtype"


###################
# datasets



if dataset=="news":
    print("news dataset")
    data, categorical_columns, ordinal_columns = load_dataset('news')
    numerical_columns=list(set(np.arange(data[:,:-1].shape[1]))-set(categorical_columns + ordinal_columns))



elif dataset=="adult": #last column is binary label
    print("adult dataset")
    data, categorical_columns, ordinal_columns = load_dataset('adult')
    numerical_columns=list(set(np.arange(data[:,:-1].shape[1]))-set(categorical_columns + ordinal_columns))
    n_classes = 2

    #numerical_input_data = data[:, numerical_columns]
    #categorical_input_data = data[:,ordinal_columns+categorical_columns][:,:-1]

elif dataset=="census":
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

elif dataset=="intrusion":
    print("intrusion dataset")
    data, categorical_columns, ordinal_columns = load_dataset('intrusion')
    numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
    #df = pd.read_csv("data/raw/intrusion/kddcup.data_10_percent", dtype='str', header=-1)
    #df = df.apply(lambda x: x.str.strip(' \t.'))
    n_classes=5


elif dataset=="covtype":
    print("covtype dataset")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname():
        data = np.load("../data/real/covtype/train.npy")
    else:
        data = np.load(
            "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/real/covtype/train.npy")

    numerical_columns = [0,1,2,3,4,5,6,7,8,9]
    ordinal_columns = []
    categorical_columns = list(set(np.arange(data.shape[1])) - set(numerical_columns + ordinal_columns))
    #0 - 122979, 1- 164062, 2 - 20740, 3 - 1560, 4 - 5510, 5 - 10065, 6 - 11792
    n_classes=7

# print(socket.gethostname())
    # if 'g0' not in socket.gethostname():
    #     data=np.load("../data/real/census/train.npy")
    # else:
    #     data = np.load(
    #         "/home/kadamczewski/Dropbox_from/Current_research/privacy/DPDR/data/Cervical/kag_risk_factors_cervical_cancer.csv")
    #
    # numerical_columns= [0,5, 16, 17, 18, 29, 38]
    # ordinal_columns = []
    # categorical_columns = [1,2,3,4,6,7,8,9,10,11,12,13,14,15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40]

#data=data[:50000, :]
print(data.shape)
print(numerical_columns)
print(categorical_columns)
print(ordinal_columns)


# np.set_printoptions(threshold=1000)
# np.set_printoptions(threshold=np.inf)

data=data[:, numerical_columns+ordinal_columns+categorical_columns]

num_numerical_inputs = len(numerical_columns)
num_categorical_inputs = len(categorical_columns+ordinal_columns)-1


inputs = data[:, :-1]
target = data[:,-1]


X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.70, test_size=0.30,
                                                    random_state=seed_number)  # 60% training and 40% test


#y_labels = y_train.values.ravel()  # X_train_pos
y_labels=y_train
#X_train = X_train.values

n_tot = X_train.shape[0]

X_train_arr=[]
y_train_arr=[]

for i in range(n_classes):

    X_train_arr.append(X_train[y_labels == i, :])
    y_train_arr.append(y_labels[y_labels == i])

n, input_dim = X_train_arr[0].shape


#####################################

LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
LR_model.fit(X_train, y_train)  # training on synthetic data
pred = LR_model.predict(X_test)  # test on real data

if n_classes > 2:
    print('F1-score', f1_score(y_test, pred, average='weighted'))
elif n_classes == 2:
    print('F1-score', f1_score(y_test, pred))
    print('ROC on real test data from Logistic regression is', roc_auc_score(y_test, pred))  # 0.9444444444444444
    print('PRC on real test data from Logistic regression is',
          average_precision_score(y_test, pred))  # 0.8955114054451803


####################################################
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






#############################################3
# train generators

def main(features_num, batch_size, input_layer, hidden1, hidden2, epochs_num, input_dim):

    generated_input_features = a=np.zeros(shape=(0,input_dim))
    corresponding_labels = []

    for which_class in range(n_classes):

    ##############3
        n, input_dim = X_train_arr[which_class].shape
        data_samps = X_train_arr[which_class]
        print('number of data samples for this class is', n)

    #############

        # test how to use RFF for computing the kernel matrix
        idx_rp = np.random.permutation(np.min([n, 10000]))
        med = util.meddistance(data_samps[idx_rp, 1:10])
        del idx_rp
        sigma2 = med ** 2
        print('length scale from median heuristic is', sigma2)

        #########################################################3
        # generator

        """ end of comment """

        # random Fourier features
        n_features = features_num  # 20000

        """ training a Generator via minimizing MMD """
    ##############
        mini_batch_size = batch_size[which_class]  # 400
        input_size = input_layer[which_class]  # 400
        hidden_size_1 = hidden1[which_class]  # 100
        hidden_size_2 = hidden2[which_class]  # 100
        output_size = input_dim
        how_many_epochs = epochs_num[which_class]  # 30

        ##################3

        output_size = input_dim

        # model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
        #                          output_size=output_size, num_categorical_inputs=num_categorical_inputs,
        #                          num_numerical_inputs=num_numerical_inputs)
        #
        # optimizer = optim.Adam(model.parameters(), lr=1e-2)
        # how_many_iter = np.int(n / mini_batch_size)

        # training_loss_per_epoch = np.zeros(how_many_epochs)

        draws = n_features // 2

        datatype='mixed'

        if datatype=='numerical_only':
            W_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

            #computing mean embedding of true data """
            emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)

        elif datatype=='mixed':
            W_freq = np.random.randn(draws, num_numerical_inputs) / np.sqrt(sigma2)

            #computing mean embedding of true data """
            numerical_input_data = data_samps[:, 0:num_numerical_inputs]
            emb1_numerical = torch.mean(RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq), 0).to(device)

            categorical_input_data = data_samps[:, num_numerical_inputs:]
            emb1_categorical = torch.Tensor(np.mean(categorical_input_data, 0) / np.sqrt(num_categorical_inputs)).to(device)

        # emb1_numerical = RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)
        # emb1_categorical =

        mean_emb1 = torch.cat((emb1_numerical, emb1_categorical))

        # del data_samps
        # del emb1_input_features

        # plt.plot(mean_emb1, 'b')
######################################33
        # start training

        model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                                 output_size=output_size, num_categorical_inputs=num_categorical_inputs,
                                 num_numerical_inputs=num_numerical_inputs).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        how_many_iter = 1 #np.int(n / mini_batch_size)

        training_loss_per_epoch = np.zeros(how_many_epochs)


        print('Starting Training')




        for epoch in range(how_many_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            # annealing_rate =

            for i in range(how_many_iter):
                # zero the parameter gradients
                optimizer.zero_grad()
                input_to_model = torch.randn((mini_batch_size, input_size))
                input_to_model=input_to_model.to(device) #[13164, 100]
                outputs = model(input_to_model) #[13164, 14]

                """ computing mean embedding of generated samples """
                # emb2_input_features = RFF_Gauss(n_features, outputs, W_freq)
                # mean_emb2 = torch.mean(emb2_input_features, 0)

                ###########
                if kernel_datasampling:
                    #W_freq = np.random.randn(draws, num_numerical_inputs) / np.sqrt(sigma2)

                    # computing mean embedding of true data """
                    sample = random.choices(np.arange(data_samps.shape[0]), k=k_datasamples)
                    numerical_input_data = data_samps[sample, 0:num_numerical_inputs]
                    emb1_numerical = torch.mean(RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq), 0).to(
                        device)

                    categorical_input_data = data_samps[:, num_numerical_inputs:]
                    emb1_categorical = torch.Tensor(
                        np.mean(categorical_input_data, 0) / np.sqrt(num_categorical_inputs)).to(device)

                    # emb1_numerical = RFF_Gauss(n_features, torch.Tensor(numerical_input_data), W_freq)
                    # emb1_categorical =

                    mean_emb1 = torch.cat((emb1_numerical, emb1_categorical))
                ###############

                numerical_samps = outputs[:, 0:num_numerical_inputs] #[4553,6]
                emb2_numerical = torch.mean(RFF_Gauss(n_features, numerical_samps, W_freq), 0) #W_freq [n_features/2,6], n_features=10000

                categorical_samps = outputs[:, num_numerical_inputs:] #[4553,8]
                emb2_categorical = torch.mean(categorical_samps, 0) * torch.sqrt(1.0/torch.Tensor([num_categorical_inputs])).to(device) # 8

                mean_emb2 = torch.cat((emb2_numerical, emb2_categorical)) #[1008]

                loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 10 == 0:
                print('epoch # and running loss are ', [epoch, running_loss])
            training_loss_per_epoch[epoch] = running_loss

#######################################################################3
        # generate samples

        """ now generated samples using the trained generator """
        input_to_model = torch.randn((n, input_size))
        input_to_model=input_to_model.to(device)
        outputs = model(input_to_model)
        samp_input_features = outputs

        # if which_class == 1:
        #     generated_samples_pos = samp_input_features.cpu().detach().numpy()
        #
        #     # save results
        #     # method = os.path.join(Results_PATH, 'Cervical_pos_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s' % (
        #     #     mini_batch_size, input_size, hidden_size_1, hidden_size_2))
        #     # np.save(method + '_loss.npy', training_loss_per_epoch)
        #     # np.save(method + '_input_feature_samps.npy', generated_samples_pos)
        #
        # else:
        #     generated_samples_neg = samp_input_features.cpu().detach().numpy()
        #
        #     # save results
        #     # method = os.path.join(Results_PATH, 'Cervical_neg_samps_batch_size=%s_input_size=%s_hidden1=%s_hidden2=%s' % (
        #     #     mini_batch_size, input_size, hidden_size_1, hidden_size_2))
        #     # np.save(method + '_loss.npy', training_loss_per_epoch)
        #     # np.save(method + '_input_feature_samps.npy', generated_samples_neg)

        # plt.figure(3)
        # plt.plot(training_loss_per_epoch)
        # plt.title('MMD as a function of epoch')
        # plt.yscale('log')
        #
        # plt.figure(4)
        # plt.plot(mean_emb1, 'b')
        # plt.plot(mean_emb2.detach().numpy(), 'r--')

        # mix data for positive and negative labels
        generated_input_features=np.concatenate((generated_input_features, np.around(samp_input_features.cpu().detach().numpy())))

        #we generated the number of samples equal to the the number of real data instances so we just add y_train which is the same both for real and generated
        corresponding_labels = np.concatenate((corresponding_labels,y_train_arr[which_class]))

    idx_shuffle = np.random.permutation(n_tot)
    shuffled_x_train = generated_input_features[idx_shuffle, :]
    shuffled_y_train = corresponding_labels[idx_shuffle]

    LR_model_ours = LogisticRegression(solver='lbfgs', max_iter=1000)
    LR_model_ours.fit(shuffled_x_train, shuffled_y_train)  # training on synthetic data
    pred_ours = LR_model_ours.predict(X_test)  # test on real data


    if n_classes > 2:
        f1score = f1_score(y_test, pred_ours, average='weighted')
        print('F1-score', f1score)
    elif n_classes == 2:
        f1score = f1_score(y_test, pred_ours)
        print('F1-score', f1score)
        print('ROC on real test data from Logistic regression is', roc_auc_score(y_test, pred))  # 0.9444444444444444
        print('PRC on real test data from Logistic regression is',
              average_precision_score(y_test, pred))  # 0.8955114054451803



    # ROC_ours = roc_auc_score(y_test, pred_ours)
    # PRC_ours = average_precision_score(y_test, pred_ours)
    #
    # print('ROC on generated samples using Logistic regression is', ROC_ours)
    # print('PRC on generated samples using Logistic regression is', PRC_ours)

    return f1score

########################################################################################33
#######################################################################3

kernel_datasampling=False
k_datasamples=1000
runs_num=5

## number of (training) samples
#input_dim - dimension of the input/number of features of the real data input
batch_var=[n,]*n_classes
input_var=[100]*n_classes
hidden1_var=[20]*n_classes
hidden2_var=[20]*n_classes
epoch_var=[2000]*n_classes



main(20000, batch_var, input_var, hidden1_var, hidden2_var, epoch_var, input_dim)
# PRC_ours_arr=[]
# ROC_ours_arr=[]
# for i in range(runs_num):
#     print("\n i: {} \n".format(i))
#     if dataset=='census':
#         #main(2000, n, 100, 20* input_dim, 20 * input_dim, 200,     n, 10, 20* input_dim, 20 * input_dim, 200)
#         main(2000, n, 100, 20* input_dim, 20 * input_dim, 20,     n, 10, 20* input_dim, 20 * input_dim, 20)
#     else:
#         roc, prc=main(5000, n, 100, 20* input_dim, 20 * input_dim, 300,     n, 10, 20* input_dim, 20 * input_dim, 300)
#         ROC_ours_arr.append(roc); PRC_ours_arr.append(prc)
# print(ROC_ours_arr)
# print("roc: ", np.mean(ROC_ours_arr))
# print(PRC_ours_arr)
# print("prc: ", np.mean(PRC_ours_arr))


# """ specifics of generators are defined here, comment this later """
# features_num = 1000
# if which_class == 1:
#     batch_cl1 = n
#     input_cl1 = 100
#     hidden1_cl1 = 20 * input_dim
#     hidden2_cl1 = np.int(20 * input_dim)
#     epochs_num_cl1 = 20
# else:
#     batch_cl0 = n
#     input_cl0 = 100
#     hidden1_cl0 = 20 * input_dim
#     hidden2_cl0 = np.int(20 * input_dim)
#     epochs_num_cl0 = 20