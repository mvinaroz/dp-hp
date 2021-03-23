import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# import util
import random
import socket
from sdgym import load_dataset
import argparse
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost
from collections import defaultdict, namedtuple
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from autodp import privacy_calibrator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from autodp import privacy_calibrator
import pandas as pd
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

import os

def Feature_labels(labels, weights, device):

    weights = torch.Tensor(weights)
    weights = weights.to(device)

    labels = labels.to(device)

    weighted_labels_feature = labels/weights

    return weighted_labels_feature


############################### generative models to use ###############################
""" two types of generative models depending on the type of features in a given dataset """

class Generative_Model_homogeneous_data(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dataset):
            super(Generative_Model_homogeneous_data, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

            self.dataset = dataset


        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)
            output = self.sigmoid(output) # because we preprocess data such that each feature is [0,1]

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
                output_numerical = self.sigmoid(output_numerical) # because we preprocess data such that each feature is [0,1]
                output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
                output_combined = torch.cat((output_numerical, output_categorical), 1)

                return output_combined

############################### end of generative models ###############################



def undersample(raw_input_features, raw_labels, undersampled_rate):
    """ we take a pre-processing step such that the dataset is a bit more balanced """
    idx_negative_label = raw_labels == 0
    idx_positive_label = raw_labels == 1

    pos_samps_input = raw_input_features[idx_positive_label, :]
    pos_samps_label = raw_labels[idx_positive_label]
    neg_samps_input = raw_input_features[idx_negative_label, :]
    neg_samps_label = raw_labels[idx_negative_label]

    # take random 10 percent of the negative labelled data
    in_keep = np.random.permutation(np.sum(idx_negative_label))
    under_sampling_rate = undersampled_rate  # 0.4
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    return feature_selected, label_selected


def data_loading(dataset, undersampled_rate, seed_number):

   if dataset=='epileptic':

        print("epileptic seizure recognition dataset") # this is homogeneous

        data = pd.read_csv("../data/Epileptic/data.csv")

        feature_names = data.iloc[:, 1:-1].columns
        target = data.iloc[:, -1:].columns

        data_features = data[feature_names]
        data_target = data[target]

        for i, row in data_target.iterrows():
          if data_target.at[i,'y']!=1:
            data_target.at[i,'y'] = 0

        ###################

        raw_labels=np.array(data_target)
        raw_input_features=np.array(data_features)

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        idx_negative_label=idx_negative_label.squeeze()
        idx_positive_label=idx_positive_label.squeeze()

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        # take random 10 percent of the negative labelled data
        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        label_selected=label_selected.squeeze()

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30, random_state=seed_number)

        n_classes = 2

        num_numerical_inputs = []
        num_categorical_inputs = []

   elif dataset=="credit":

        print("Creditcard fraud detection dataset") # this is homogeneous

        data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")

        feature_names = data.iloc[:, 1:30].columns
        target = data.iloc[:, 30:].columns

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
        under_sampling_rate = undersampled_rate
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)
        n_classes = 2
        num_numerical_inputs = []
        num_categorical_inputs = []

   elif dataset=='census':

        print("census dataset") # this is heterogenous

        data = np.load("../data/real/census/train.npy")

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
        under_sampling_rate = undersampled_rate #0.4
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)


   elif dataset=='cervical':

        print("dataset is", dataset) # this is heterogenous

        df = pd.read_csv("../data/Cervical/kag_risk_factors_cervical_cancer.csv")

        # df.head()
        df_nan = df.replace("?", np.float64(np.nan))
        df_nan.head()

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
            feature_mean = round(df[feature].median(), 1)
            df[feature] = df[feature].fillna(feature_mean)

        for feature in categorical_df:
            df[feature] = df[feature].fillna(0.0)


        target = df['Biopsy']
        inputs = df[feature_names]
        print('raw input features', inputs.shape)

        n_classes = 2

        raw_input_features = inputs.values
        raw_labels = target.values.ravel()

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
        under_sampling_rate = undersampled_rate #0.5
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                            test_size=0.20, random_state=seed_number)

   elif dataset=='adult':

        print("dataset is", dataset) # this is heterogenous
        data, categorical_columns, ordinal_columns = load_dataset('adult')

        """ some specifics on this dataset """
        numerical_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns + ordinal_columns))
        n_classes = 2

        data = data[:, numerical_columns + ordinal_columns + categorical_columns]
        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        inputs = data[:, :-1]
        target = data[:, -1]

        inputs, target=undersample(inputs, target, undersampled_rate)

        X_train, X_test, y_train, y_test = train_test_split(inputs, target, train_size=0.90, test_size=0.10,
                                                            random_state=seed_number)

   elif dataset=='isolet':

        print("isolet dataset")

        data_features_npy = np.load('../data/Isolet/isolet_data.npy')
        data_target_npy = np.load('../data/Isolet/isolet_labels.npy')

        values = data_features_npy
        index = ['Row' + str(i) for i in range(1, len(values) + 1)]

        values_l = data_target_npy
        index_l = ['Row' + str(i) for i in range(1, len(values) + 1)]

        data_features = pd.DataFrame(values, index=index)
        data_target = pd.DataFrame(values_l, index=index_l)

        ####

        raw_labels = np.array(data_target)
        raw_input_features = np.array(data_features)

        idx_negative_label = raw_labels == 0
        idx_positive_label = raw_labels == 1

        idx_negative_label = idx_negative_label.squeeze()
        idx_positive_label = idx_positive_label.squeeze()

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.01
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        label_selected = label_selected.squeeze()

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                            random_state=seed_number)
        n_classes = 2
        num_numerical_inputs = []
        num_categorical_inputs = []


   elif dataset=='intrusion':

        print("dataset is", dataset)
        data, categorical_columns, ordinal_columns = load_dataset('intrusion')

        """ some specifics on this dataset """
        n_classes = 5

        """ some changes we make in the type of features for applying to our model """
        categorical_columns_binary = [6, 11, 13, 20]  # these are binary categorical columns
        the_rest_columns = list(set(np.arange(data[:, :-1].shape[1])) - set(categorical_columns_binary))

        num_numerical_inputs = len(the_rest_columns)  # 10. Separately from the numerical ones, we compute the length-scale for the rest columns
        num_categorical_inputs = len(categorical_columns_binary)

        raw_labels = data[:, -1]
        raw_input_features = data[:, the_rest_columns + categorical_columns_binary]
        print(raw_input_features.shape)

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 0  # this is a dominant one about 80%, which we want to undersample
        idx_positive_label = raw_labels != 0

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate#0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70,
                                                            test_size=0.30,
                                                            random_state=seed_number)

   elif dataset=='covtype':

        print("dataset is", dataset)

        train_data = np.load("../data/real/covtype/train.npy")
        test_data = np.load("../data/real/covtype/test.npy")
        # we put them together and make a new train/test split in the following
        data = np.concatenate((train_data, test_data))

        """ some specifics on this dataset """
        numerical_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ordinal_columns = []
        categorical_columns = list(set(np.arange(data.shape[1])) - set(numerical_columns + ordinal_columns))
        # Note: in this dataset, the categorical variables are all binary
        n_classes = 7

        print('data shape is', data.shape)
        print('indices for numerical columns are', numerical_columns)
        print('indices for categorical columns are', categorical_columns)
        print('indices for ordinal columns are', ordinal_columns)

        # sorting the data based on the type of features.
        data = data[:, numerical_columns + ordinal_columns + categorical_columns]

        num_numerical_inputs = len(numerical_columns)
        num_categorical_inputs = len(categorical_columns + ordinal_columns) - 1

        inputs = data[:20000, :-1]
        target = data[:20000, -1]

        ##################3

        raw_labels=target
        raw_input_features=inputs

        """ we take a pre-processing step such that the dataset is a bit more balanced """
        idx_negative_label = raw_labels == 1  # 1 and 0 are dominant but 1 has more labels
        idx_positive_label = raw_labels != 1

        pos_samps_input = raw_input_features[idx_positive_label, :]
        pos_samps_label = raw_labels[idx_positive_label]
        neg_samps_input = raw_input_features[idx_negative_label, :]
        neg_samps_label = raw_labels[idx_negative_label]

        in_keep = np.random.permutation(np.sum(idx_negative_label))
        under_sampling_rate = undersampled_rate  # 0.3
        in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

        neg_samps_input = neg_samps_input[in_keep, :]
        neg_samps_label = neg_samps_label[in_keep]

        feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
        label_selected = np.concatenate((pos_samps_label, neg_samps_label))


        ###############3

        X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.70, test_size=0.30,
                                                            random_state=seed_number)  # 60% training and 40% test

   return X_train, X_test, y_train, y_test, n_classes, num_numerical_inputs, num_categorical_inputs


def test_models(X_tr, y_tr, X_te, y_te, n_classes, datasettype, args):

    print("\n", datasettype, "data\n")

    roc_arr = []
    prc_arr = []
    f1_arr = []

    models = np.array(
        [LogisticRegression(solver='lbfgs', max_iter=50000), GaussianNB(), BernoulliNB(alpha=0.02),
         LinearSVC(max_iter=10000, tol=1e-8, loss='hinge'),
         DecisionTreeClassifier(class_weight='balanced', criterion='gini', splitter='best',
                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                    min_impurity_decrease=0.0),
         LinearDiscriminantAnalysis(solver='eigen', tol=1e-8, shrinkage=0.5),
         AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R'),
         BaggingClassifier(max_samples=0.1, n_estimators=20),
         RandomForestClassifier(n_estimators=100, class_weight='balanced'),
         GradientBoostingClassifier(subsample=0.1, n_estimators=50),
         MLPClassifier(),
         xgboost.XGBClassifier()])

    models_to_test = models[np.array(args)]

    for model in models_to_test:
        print('\n', type(model))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)  # test on real data

        if n_classes > 2:

            f1score = f1_score(y_te, pred, average='weighted')

            print("F1-score on test %s data is %.3f" % (datasettype, f1score))
            f1_arr.append(f1score)

        else:

            roc = roc_auc_score(y_te, pred)
            prc = average_precision_score(y_te, pred)

            print("ROC on test %s data is %.3f" % (datasettype, roc))
            print("PRC on test %s data is %.3f" % (datasettype, prc))

            roc_arr.append(roc)
            prc_arr.append(prc)

    if n_classes > 2:

        res1 = np.mean(f1_arr)
        res1_arr = f1_arr
        print("------\nf1 mean across methods is %.3f\n" % res1)
        res2_arr = 0  # dummy

    else:

        res1 = np.mean(roc_arr)
        res1_arr = roc_arr
        res2 = np.mean(prc_arr)
        res2_arr = prc_arr
        print("-" * 40)
        print("roc mean across methods is %.3f" % res1)
        print("prc mean across methods is %.3f\n" % res2)

    return res1_arr, res2_arr




def save_generated_samples(samples, args):
    path_gen_data = f"../data/generated/{args.dataset}"
    os.makedirs(path_gen_data, exist_ok=True)
    if args.is_private:
        np.save(os.path.join(path_gen_data, f"{args.dataset}_generated_privatized_{args.is_private}_eps_{args.epsilon}_epochs_{args.epochs}_features_{args.num_features}_samples_{samples.shape[0]}_features_{samples.shape[1]}"), samples.detach().cpu().numpy())
    else:
        np.save(os.path.join(path_gen_data, f"{args.dataset}_generated_privatized_{args.is_private}_epochs_{args.epochs}_features_{args.num_features}_samples_{samples.shape[0]}_features_{samples.shape[1]}"), samples.detach().cpu().numpy())
    print(f"Generated data saved to {path_gen_data}")


def heuristic_for_length_scale(dataset, X_train, num_numerical_inputs, input_dim, heterogeneous_datasets):

    if dataset == 'census':

        sigma_array = np.zeros(num_numerical_inputs)
        for i in np.arange(0, num_numerical_inputs):
            med = meddistance(np.expand_dims(X_train[:, i], 1), subsample=5000)
            sigma_array[i] = med

        print('we will use separate frequencies for each column of numerical features')
        sigma2 = sigma_array ** 2
        sigma2[sigma2 == 0] = 1.0

    elif dataset == 'credit':

        # large value at the last column
        med = meddistance(X_train[:, 0:-1], subsample=5000)
        med_last = meddistance(np.expand_dims(X_train[:, -1], 1), subsample=5000)
        sigma_array = np.concatenate((med * np.ones(input_dim - 1), [med_last]))

        sigma2 = sigma_array ** 2
        sigma2[sigma2 == 0] = 1.0

    else:

        if dataset in heterogeneous_datasets:
            med = meddistance(X_train[:, 0:num_numerical_inputs], subsample=5000)
        elif dataset == 'cervical':
            med = meddistance(X_train, subsample=500)
        else:
            med = meddistance(X_train, subsample=5000)

        sigma2 = med ** 2

    return sigma2


def meddistance(x, subsample=None, mean_on_fail=True):
  """
  Compute the median of pairwise distances (not distance squared) of points
  in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

  Parameters
  ----------
  x : n x d numpy array
  mean_on_fail: True/False. If True, use the mean when the median distance is 0.
      This can happen especially, when the data are discrete e.g., 0/1, and
      there are more slightly more 0 than 1. In this case, the m

  Return
  ------
  median distance
  """
  if subsample is None:
    d = dist_matrix(x, x)
    itri = np.tril_indices(d.shape[0], -1)
    tri = d[itri]
    med = np.median(tri)
    if med <= 0:
      # use the mean
      return np.mean(tri)
    return med

  else:
    assert subsample > 0
    rand_state = np.random.get_state()
    np.random.seed(9827)
    n = x.shape[0]
    ind = np.random.choice(n, min(subsample, n), replace=False)
    np.random.set_state(rand_state)
    # recursion just one
    return meddistance(x[ind, :], None, mean_on_fail)


def dist_matrix(x, y):
  """
  Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
  """
  sx = np.sum(x ** 2, 1)
  sy = np.sum(y ** 2, 1)
  d2 = sx[:, np.newaxis] - 2.0 * x.dot(y.T) + sy[np.newaxis, :]
  # to prevent numerical errors from taking sqrt of negative numbers
  d2[d2 < 0] = 0
  d = np.sqrt(d2)
  return d
