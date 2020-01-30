import os
from collections import defaultdict
import numpy as np
from torchvision import datasets
import argparse
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost


def test_model(model, x_trn, y_trn, x_tst, y_tst):
  model.fit(x_trn, y_trn)
  y_pred = model.predict(x_tst)
  acc = accuracy_score(y_pred, y_tst)
  f1 = f1_score(y_true=y_tst, y_pred=y_pred, average='macro')
  conf = confusion_matrix(y_true=y_tst, y_pred=y_pred)
  return acc, f1, conf


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-base-dir', type=str, default='logs/gen/')
  parser.add_argument('--data-log-name', type=str, default='tb12_16_2')
  parser.add_argument('--log-results', action='store_true', default=False)
  parser.add_argument('--skip-slow-models', action='store_true', default=False)
  ar = parser.parse_args()

  gen_data_dir = os.path.join(ar.data_base_dir, ar.data_log_name)
  log_save_dir = os.path.join(gen_data_dir, 'synth_eval/')
  if ar.log_results and not os.path.exists(log_save_dir):
    os.makedirs(log_save_dir)

  train_data = datasets.MNIST('../../data', train=True)
  x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
  x_real_train = np.reshape(x_real_train, (-1, 784)) / 255

  test_data = datasets.MNIST('../../data', train=False)
  x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
  x_real_test = np.reshape(x_real_test, (-1, 784)) / 255

  gen_data = np.load(os.path.join(gen_data_dir, 'synthetic_mnist.npz'))
  x_gen, y_gen = gen_data['data'], gen_data['labels'].ravel()

  print(f'data ranges: [{np.min(x_real_test)}, {np.max(x_real_test)}], [{np.min(x_real_train)}, '
        f'{np.max(x_real_train)}], [{np.min(x_gen)}, {np.max(x_gen)}]')
  print(f'label ranges: [{np.min(y_real_test)}, {np.max(y_real_test)}], [{np.min(y_real_train)}, '
        f'{np.max(y_real_train)}], [{np.min(y_gen)}, {np.max(y_gen)}]')

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
            'xgboost': xgboost.XGBClassifier}

  slow_models = {'logistic_reg', 'bagging', 'gbm', 'xgboost'}

  model_specs = defaultdict(dict)
  model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 5000, 'multi_class': 'auto'}
  model_specs['random_forest'] = {'n_estimators': 100}
  model_specs['linear_svc'] = {'max_iter': 5000}  # still not enough??
  model_specs['bernoulli_nb'] = {'binarize': 0.5}

  for key in models.keys():
    if ar.skip_slow_models and key in slow_models:
      continue

    print(f'Model: {key}', end='')
    model = models[key](**model_specs[key])
    base_acc, base_f1, base_conf = test_model(model, x_real_train, y_real_train, x_real_test, y_real_test)
    print('.', end='')

    model = models[key](**model_specs[key])
    g_to_r_acc, g_to_r_f1, g_to_r_conf = test_model(model, x_gen, y_gen, x_real_test, y_real_test)
    print('.', end='')

    model = models[key](**model_specs[key])
    r_to_g_acc, r_to_g_f1, r_to_g_conv = test_model(model, x_real_train, y_real_train, x_gen[:10000], y_gen[:10000])
    print('.')

    print(f'acc: real {base_acc}, gen to real {g_to_r_acc}, real to gen {r_to_g_acc}')
    print(f'f1:  real {base_f1}, gen to real {g_to_r_f1}, real to gen {r_to_g_f1}')
    print('gen to real confusion matrix:')
    print(g_to_r_conf)

    if ar.log_results:
      accs = np.asarray([base_acc, g_to_r_acc, r_to_g_acc])
      f1_scores = np.asarray([base_f1, g_to_r_f1, r_to_g_f1])
      conf_mats = np.stack([base_conf, g_to_r_conf, r_to_g_conv])
      np.savez(os.path.join(log_save_dir, f'{key}_log'), accuracies=accs, f1_scores=f1_scores, conf_mats=conf_mats)


if __name__ == '__main__':
  main()
