import os
from collections import defaultdict
import numpy as np
from torchvision import datasets
import argparse
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost
import time


def test_model(model, x_trn, y_trn, x_tst, y_tst):
  model.fit(x_trn, y_trn)
  y_pred = model.predict(x_tst)
  acc = accuracy_score(y_pred, y_tst)
  f1 = f1_score(y_true=y_tst, y_pred=y_pred, average='macro')
  conf = confusion_matrix(y_true=y_tst, y_pred=y_pred)
  return acc, f1, conf


def subsample_data(x, y, frac, balance_classes=True):
  n_data = y.shape[0]
  n_classes = np.max(y) + 1
  new_n_data = int(n_data * frac)
  if not balance_classes:
    x, y = x[:new_n_data], y[:new_n_data]
  else:
    n_data_per_class = new_n_data // n_classes
    assert n_data_per_class * n_classes == new_n_data
    print(f'starting label count {[sum(y == k) for k in range(n_classes)]}')
    print('DEBUG: NCLASSES', n_classes, 'NDATA', n_data)
    rand_perm = np.random.permutation(n_data)
    x = x[rand_perm]
    y = y[rand_perm]
    # y_scalar = np.argmax(y, axis=1)

    data_ids = [[], [], [], [], [], [], [], [], [], []]
    n_full = 0
    for idx in range(n_data):
      l = y[idx]
      if len(data_ids[l]) < n_data_per_class:
        data_ids[l].append(idx)
        # print(l)
        if len(data_ids[l]) == n_data_per_class:
          n_full += 1
          if n_full == n_classes:
            break

    data_ids = np.asarray(data_ids)
    data_ids = np.reshape(data_ids, (new_n_data,))
    rand_perm = np.random.permutation(new_n_data)
    data_ids = data_ids[rand_perm]  # otherwise sorted by class
    x = x[data_ids]
    y = y[data_ids]

    print(f'subsampled label count {[sum(y == k) for k in range(n_classes)]}')
  return x, y


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-path', type=str, default=None, help='this is computed. only set to override')
  parser.add_argument('--data-base-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--data-log-name', type=str, default=None, help='subdirectory for this run')

  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--shuffle-data', action='store_true', default=False, help='shuffle data before testing')

  parser.add_argument('--log-results', action='store_true', default=False, help='if true, save results')
  parser.add_argument('--print-conf-mat', action='store_true', default=False, help='print confusion matrix')

  parser.add_argument('--skip-slow-models', action='store_true', default=False, help='skip models that take longer')
  parser.add_argument('--only-slow-models', action='store_true', default=False, help='only do slower the models')
  parser.add_argument('--custom-keys', type=str, default=None, help='enter model keys to run as key1,key2,key3...')

  parser.add_argument('--skip-gen-to-real', action='store_true', default=False, help='skip train:gen,test:real setting')
  parser.add_argument('--compute-real-to-real', action='store_true', default=False, help='add train:real,test:real')
  parser.add_argument('--compute-real-to-gen', action='store_true', default=False, help='add train:real,test:gen')

  parser.add_argument('--subsample', type=float, default=1., help='fraction on data to use in training')
  parser.add_argument('--sub-balanced-labels', action='store_true', default=False, help='add train:real,test:gen')

  ar = parser.parse_args()

  if ar.data_log_name is not None:
    print(f'processing {ar.data_log_name}')

  gen_data_dir = os.path.join(ar.data_base_dir, ar.data_log_name)
  log_save_dir = os.path.join(gen_data_dir, 'synth_eval/')
  print('attempting to make dir')
  if ar.log_results and not os.path.exists(log_save_dir):
    os.makedirs(log_save_dir)
  print('made dir')
  if ar.data_path is None:
    ar.data_path = os.path.join(gen_data_dir, 'synthetic_mnist.npz')
  if ar.data == 'digits':
    train_data = datasets.MNIST('data', train=True)
    test_data = datasets.MNIST('data', train=False)
  elif ar.data == 'fashion':
    train_data = datasets.FashionMNIST('data', train=True)
    test_data = datasets.FashionMNIST('data', train=False)
  else:
    raise ValueError

  print('got dataset')
  x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
  x_real_train = np.reshape(x_real_train, (-1, 784)) / 255
  print('reshaped train set')
  x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
  x_real_test = np.reshape(x_real_test, (-1, 784)) / 255
  print('reshaped test set')
  print('loading gen data')
  time.sleep(5)
  gen_data = np.load(ar.data_path)
  print('loaded gen data')
  x_gen, y_gen = gen_data['data'], gen_data['labels']
  if len(y_gen.shape) == 2:  # remove onehot
    if y_gen.shape[1] == 1:
      y_gen = y_gen.ravel()
    elif y_gen.shape[1] == 10:
      y_gen = np.argmax(y_gen, axis=1)
    else:
      raise ValueError

  print('shuffling data')
  if ar.shuffle_data:
    rand_perm = np.random.permutation(y_gen.shape[0])
    x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]

  print('checking subsample')
  if ar.subsample < 1.:
    x_gen, y_gen = subsample_data(x_gen, y_gen, ar.subsample, ar.sub_balanced_labels)
    x_real_train, y_real_train = subsample_data(x_real_train, y_real_train, ar.subsample, ar.sub_balanced_labels)

    print(f'training on {ar.subsample * 100.}% of the original syntetic dataset')

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
            'mlp': neural_network.MLPClassifier,
            'bagging': ensemble.BaggingClassifier,
            'gbm': ensemble.GradientBoostingClassifier,
            'xgboost': xgboost.XGBClassifier}

  slow_models = {'bagging', 'gbm', 'xgboost'}

  model_specs = defaultdict(dict)
  model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 5000, 'multi_class': 'auto'}
  model_specs['random_forest'] = {'n_estimators': 100, 'class_weight': 'balanced'}
  model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
  model_specs['bernoulli_nb'] = {'binarize': 0.5}
  model_specs['lda'] = {'solver': 'eigen', 'n_components': 9, 'tol': 1e-8, 'shrinkage': 0.5}
  model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini', 'splitter': 'best',
                                  'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                                  'min_impurity_decrease': 0.0}
  model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}
  model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
  model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
  model_specs['xgboost'] = {'colsample_bytree': 0.1, 'objective': 'multi:softprob', 'n_estimators': 50}

  print('got models, setting keys')
  if ar.custom_keys is not None:
    run_keys = ar.custom_keys.split(',')
  elif ar.skip_slow_models:
    run_keys = [k for k in models.keys() if k not in slow_models]
  elif ar.only_slow_models:
    run_keys = [k for k in models.keys() if k in slow_models]
  else:
    run_keys = models.keys()

  for key in run_keys:
    print(f'Model: {key}')

    acc_str = 'acc:'
    f1_str = 'f1:'

    if not ar.skip_gen_to_real:
      model = models[key](**model_specs[key])
      g_to_r_acc, g_to_r_f1, g_to_r_conf = test_model(model, x_gen, y_gen, x_real_test, y_real_test)
      acc_str = acc_str + f' gen to real {g_to_r_acc}'
      f1_str = f1_str + f' gen to real {g_to_r_f1}'
    else:
      g_to_r_acc, g_to_r_f1, g_to_r_conf = -1, -1, -np.ones((10, 10))

    if ar.compute_real_to_real:
      model = models[key](**model_specs[key])
      base_acc, base_f1, base_conf = test_model(model, x_real_train, y_real_train, x_real_test, y_real_test)
      acc_str = acc_str + f' real to real {base_acc}'
      f1_str = f1_str + f' real to real {base_f1}'
    else:
      base_acc, base_f1, base_conf = -1, -1, -np.ones((10, 10))

    if ar.compute_real_to_gen:
      model = models[key](**model_specs[key])
      r_to_g_acc, r_to_g_f1, r_to_g_conv = test_model(model, x_real_train, y_real_train, x_gen[:10000], y_gen[:10000])
      acc_str = acc_str + f' real to gen {r_to_g_acc}'
      f1_str = f1_str + f' real to gen {r_to_g_f1}'
    else:
      r_to_g_acc, r_to_g_f1, r_to_g_conv = -1, -1, -np.ones((10, 10))

    # print(f'acc: real {base_acc}, gen to real {g_to_r_acc}, real to gen {r_to_g_acc}')
    # print(f'f1:  real {base_f1}, gen to real {g_to_r_f1}, real to gen {r_to_g_f1}')
    print(acc_str)
    print(f1_str)
    if ar.print_conf_mat:
      print('gen to real confusion matrix:')
      print(g_to_r_conf)

    if ar.log_results:
      accs = np.asarray([base_acc, g_to_r_acc, r_to_g_acc])
      f1_scores = np.asarray([base_f1, g_to_r_f1, r_to_g_f1])
      conf_mats = np.stack([base_conf, g_to_r_conf, r_to_g_conv])
      file_name = f'{key}_log' if ar.subsample == 1. else f'sub{ar.subsample}_{key}_log'
      np.savez(os.path.join(log_save_dir, file_name), accuracies=accs, f1_scores=f1_scores, conf_mats=conf_mats)


if __name__ == '__main__':
  main()
