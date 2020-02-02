import socket
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


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
#import xgboost

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
#from autodp import privacy_calibrator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score


def load_isolet():

    seed_number=0

    print("isolet dataset")
    print(socket.gethostname())
    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
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
                                                        random_state=seed_number)

    # unpack data
    X_train = X_train.values
    y_train = y_train.values.ravel()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    n_classes = 2

    data_features=np.array(data_features)
    data_target=np.array(data_target).squeeze()

    return X_train, y_train, X_test, y_test.squeeze()

def load_credit():

    seed_number=0


    print("Creditcard fraud detection dataset") # this is homogeneous

    if 'g0' not in socket.gethostname() and 'p0' not in socket.gethostname():
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
    under_sampling_rate = 0.01# undersampled_rate #0.01
    # under_sampling_rate = 0.3
    in_keep = in_keep[0:np.int(np.sum(idx_negative_label) * under_sampling_rate)]

    neg_samps_input = neg_samps_input[in_keep, :]
    neg_samps_label = neg_samps_label[in_keep]

    feature_selected = np.concatenate((pos_samps_input, neg_samps_input))
    label_selected = np.concatenate((pos_samps_label, neg_samps_label))

    X_train, X_test, y_train, y_test = train_test_split(feature_selected, label_selected, train_size=0.80,
                                                        test_size=0.20, random_state=seed_number)
    n_classes = 2

    return X_train, y_train, X_test, y_test.squeeze()



n_classes=2

def test_models(X_tr, y_tr, X_te, y_te, datasettype, n_classes=2):

    roc_arr=[]
    prc_arr=[]
    f1_arr=[]

#    for model in [LogisticRegression(solver='lbfgs', max_iter=1000), GaussianNB(), BernoulliNB(alpha=0.02), LinearSVC(), DecisionTreeClassifier(), LinearDiscriminantAnalysis(), AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier(), xgboost.XGBClassifier()]:
    for model in [LogisticRegression(solver='lbfgs', max_iter=1000), GaussianNB(), BernoulliNB(alpha=0.02), LinearSVC(), DecisionTreeClassifier(), LinearDiscriminantAnalysis(), AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]:

    #for model in [LogisticRegression(solver='lbfgs', max_iter=1000), BernoulliNB(alpha=0.02)]:

        print('\n', type(model))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)  # test on real data

    #LR_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    # LR_model.fit(X_train, y_train)  # training on synthetic data
    # pred = LR_model.predict(X_test)  # test on real data

        if n_classes>2:

            f1score = f1_score(y_te, pred, average='weighted')

            print("F1-score on test %s data is %.3f" % (datasettype, f1score))
            # 0.6742486709433465 for covtype data, 0.9677751506935462 for intrusion data
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
        print("f1 mean across methods is %.3f\n" % res1)
        res2 = 0  # dummy
    else:

        res1=np.mean(roc_arr)
        res2=np.mean(prc_arr)
        print("roc mean across methods is %.3f" % res1)
        print("prc mean across methods is %.3f\n" % res2)


    return res1, res2
