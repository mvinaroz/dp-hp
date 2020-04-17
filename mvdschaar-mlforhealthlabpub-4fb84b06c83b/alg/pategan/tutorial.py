import pandas as pd
from collections import Counter
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab

fn_csv = '{}/spambase.csv.gz'.format(utilmlab.get_data_dir())
df = pd.read_csv(fn_csv)  # get UCI spam dataset
target = 'label'

df.head()

import numpy as np

train_ratio = 0.8
fn_train = 'train.csv'
fn_test = 'test.csv'

idx = np.random.permutation(len(df))

train_idx = idx[:int(train_ratio * len(df))]
test_idx = idx[int(train_ratio * len(df)):]
        
df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

df_train.to_csv(fn_train, index=False)
df_test.to_csv(fn_test, index=False)

python_exe='python3'
niter=10000
fn_o_train = 'otrain.csv'
fn_o_test =  'otest.csv'
teachers = 50  # use a reduced number of teachers to limit the execution time
epsilon = 1
delta = 5

import subprocess

cmd_arg = '--iter {} --target {} --itrain {} --itest {} --otrain {} --otest {} --teachers {} --epsilon {} --delta {}'.format(niter, target, fn_train, fn_test, fn_o_train, fn_o_test, teachers, epsilon, delta)
print(cmd_arg)

#!{python_exe} pategan.py {cmd_arg}

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter


def train_evaluate(df_trn, df_tst):
    model = LogisticRegression(solver='lbfgs', max_iter=4000)

    features = list(df_trn.columns)
    features.remove(target)

    model.fit(df_trn[features], df_trn[target])
    pred_proba = model.predict_proba(df_tst[features])
    return metrics.roc_auc_score(df_tst[target], pred_proba[:,1])

auc = dict()
auc['org'] = train_evaluate(df_train, df_test)

df_pategan_train = pd.read_csv(fn_o_train)

auc['pategan'] = train_evaluate(df_pategan_train, df_test)
    
print('aucroc orignal data {:0.4f} auc synthetic data {:0.4f}'.format(auc['org'], auc['pategan']))
