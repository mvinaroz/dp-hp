""" download data from : https://www.kaggle.com/mlg-ulb/creditcardfraud """
""" first, set up the same setting as Pate-GAN paper for baseline comparison """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# df = pd.read_csv('../input/creditcard.csv')
# print(df.shape)
# df.head()


# (1) load data
data = pd.read_csv("../data/Kaggle_Credit/creditcard.csv")

print(data.shape)

# data.info()

class_names = {0:'Not Fraud', 1:'Fraud'}
print(data.Class.value_counts().rename(index = class_names))
# highly imbalanced data: 492(Fraud) and 284315 (Not Fraud)


feature_names = data.iloc[:, 1:30].columns
target = data.iloc[:1, 30: ].columns
# print(feature_names)
# print(target)


data_features = data[feature_names]
data_target = data[target]

# data split following from https://www.kaggle.com/renjithmadhavan/credit-card-fraud-detection-using-python
# 10-fold Cross validation
how_many_fold = 10
random_state = np.arange(0,how_many_fold)
AUROC = np.zeros(how_many_fold)
AUPRC = np.zeros(how_many_fold)

model = LogisticRegression(solver='lbfgs', max_iter=1000)

for i in random_state:
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=i)
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)

    AUROC[i] = roc_auc_score(y_test, pred)
    AUPRC[i] = average_precision_score(y_test, pred)
    print('ROC and PRC are', [AUROC[i], AUPRC[i]])


print('avg AUROC is', np.mean(AUROC))
print('avg AUPRC is', np.mean(AUPRC))