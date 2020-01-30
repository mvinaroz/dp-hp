import os
from collections import defaultdict
import numpy as np
from torchvision import datasets
import argparse
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score
import xgboost


def test_model(model, x_trn, y_trn, x_tst, y_tst):
  model.fit(x_trn, y_trn)
  y_pred = model.predict(x_tst)
  acc = accuracy_score(y_pred, y_tst)
  f1 = f1_score(y_true=y_tst, y_pred=y_pred, average='macro')
  return acc, f1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-base-dir', type=str, default='logs/gen/')
  parser.add_argument('--data-log-name', type=str, default='tb12_16_2')
  ar = parser.parse_args()

  train_data = datasets.MNIST('../../data', train=True)
  x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
  x_real_train = np.reshape(x_real_train, (-1, 784)) / 255
  test_data = datasets.MNIST('../../data', train=False)
  x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
  x_real_test = np.reshape(x_real_test, (-1, 784)) / 255
  gen_data = np.load(os.path.join(ar.data_base_dir, ar.data_log_name, 'synthetic_mnist.npz'))
  x_gen, y_gen = gen_data['data'], gen_data['labels']

  models = {'logistic_reg': linear_model.LogisticRegression,
            'random_forest': ensemble.RandomForestClassifier,
            'gaussian_nb': naive_bayes.GaussianNB,
            'bernoulli_nb': naive_bayes.BernoulliNB,
            'linear_svc': svm.LinearSVC,
            'decision_tree': tree.DecisionTreeClassifier,
            'lda': discriminant_analysis.LinearDiscriminantAnalysis,
            'adaboost': ensemble.AdaBoostClassifier,
            'bagging': ensemble.BaggingClassifier,
            'gbm': ensemble.GradientBoostingClassifier,
            'mlp': neural_network.MLPClassifier,
            'xgboost': xgboost.XGBClassifier
            }

  model_specs = defaultdict(dict)
  model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 1000, 'multi_class': 'auto'}

  for key in models.keys():
    print(f'Model: {key}', end='')
    model = models[key](**model_specs[key])
    baseline_acc, baseline_f1 = test_model(model, x_real_train, y_real_train, x_real_test, y_real_test)
    print('.', end='')

    model = models[key](**model_specs[key])
    gen_to_real_acc, gen_to_real_f1 = test_model(model, x_gen, y_gen, x_real_test, y_real_test)
    print('.', end='')

    model = models[key](**model_specs[key])
    real_to_gen_acc, real_to_gen_f1 = test_model(model, x_real_train, y_real_train, x_gen[:10000], y_gen[:10000])
    print('.')

    print(f'acc: real {baseline_acc}, gen to real {gen_to_real_acc}, real to gen {real_to_gen_acc}')
    print(f'f1:  real {baseline_f1}, gen to real {gen_to_real_f1}, real to gen {real_to_gen_f1}')


if __name__ == '__main__':
  main()
