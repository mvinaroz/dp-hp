""" this contains all the relevant scripts, copied from the code_balanced folder"""

import torch as pt
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, namedtuple
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost

datasets_colletion_def = namedtuple('datasets_collection', ['x_gen', 'y_gen',
                                                            'x_real_train', 'y_real_train',
                                                            'x_real_test', 'y_real_test'])

def hermite_polynomial_induction(h_n, h_n_minus_1, degree, x_in, probabilists=False):
  fac = 1. if probabilists else 2.
  if degree == 0:
    return pt.tensor(1., dtype=pt.float32, device=x_in.device)
  elif degree == 1:
    return fac * x_in
  else:
    n = degree - 1
    h_n_plus_one = fac*x_in*h_n - fac*n*h_n_minus_1
    return h_n_plus_one

def eval_hermite_pytorch(x_in, n_degrees, device, return_only_last_term=True):
  n_samples = x_in.shape[0]
  n_features = x_in.shape[1]
  batch_embedding = pt.empty(n_samples, n_degrees, n_features, dtype=pt.float32, device=device)
  h_i_minus_one, h_i_minus_two = None, None
  for degree in range(n_degrees):
    h_i = hermite_polynomial_induction(h_i_minus_one, h_i_minus_two, degree, x_in, probabilists=False)

    h_i_minus_two = h_i_minus_one
    h_i_minus_one = h_i
    batch_embedding[:, degree, :] = h_i

  if return_only_last_term:
    return batch_embedding[:, -1, :]
  else:
    return batch_embedding

def synthesize_data_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  if n_data % gen_batch_size != 0:
    assert n_data % 100 == 0
    gen_batch_size = n_data // 100
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()

def plot_data(data, labels, save_str, class_centers=None, subsample=None, center_frame=False, title=''):
  n_classes = int(np.max(labels)) + 1
  colors = ['r', 'b', 'g', 'y', 'orange', 'black', 'grey', 'cyan', 'magenta', 'brown']
  plt.figure()
  plt.title(title)
  if center_frame:
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)

  for c_idx in range(n_classes):
    c_data = data[labels == c_idx]

    if subsample is not None:
      n_sub = int(np.floor(len(c_data) * subsample))
      c_data = c_data[np.random.permutation(len(c_data))][:n_sub]

    plt.scatter(c_data[:, 1], c_data[:, 0], label=c_idx, c=colors[c_idx], s=.1)

    if class_centers is not None:
      print(class_centers[c_idx, 0, :])
      plt.scatter(class_centers[c_idx, :, 1], class_centers[c_idx, :, 0], marker='x', c=colors[c_idx], s=50.)

  plt.xlabel('x')
  plt.ylabel('y')
  # plt.legend()
  plt.savefig(f'{save_str}.png')


def subsample_data(x, y, frac, balance_classes=True):
  n_data = y.shape[0]
  n_classes = np.max(y) + 1
  new_n_data = int(n_data * frac)
  if not balance_classes:
    x, y = x[:new_n_data], y[:new_n_data]
  else:
    n_data_per_class = new_n_data // n_classes
    assert n_data_per_class * n_classes == new_n_data
    # print(f'starting label count {[sum(y == k) for k in range(n_classes)]}')
    # print('DEBUG: NCLASSES', n_classes, 'NDATA', n_data)
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


def load_mnist_data(data_key, data_from_torch, base_dir='data/'):
  if not data_from_torch:
    if data_key == 'digits':
      d = np.load(
        os.path.join(base_dir, 'MNIST/numpy_dmnist.npz'))  # x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst

      return d['x_train'].reshape(60000, 784), d['y_train'], d['x_test'].reshape(10000, 784), d['y_test']
    elif data_key == 'fashion':
      d = np.load(os.path.join(base_dir, 'FashionMNIST/numpy_fmnist.npz'))
      return d['x_train'], d['y_train'], d['x_test'], d['y_test']
    else:
      raise ValueError
  else:
    from torchvision import datasets
    if data_key == 'digits':
      train_data = datasets.MNIST('data', train=True)
      test_data = datasets.MNIST('data', train=False)
    elif data_key == 'fashion':
      train_data = datasets.FashionMNIST('data', train=True)
      test_data = datasets.FashionMNIST('data', train=False)
    else:
      raise ValueError

    x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
    x_real_train = np.reshape(x_real_train, (-1, 784)) / 255

    x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
    x_real_test = np.reshape(x_real_test, (-1, 784)) / 255
    return x_real_train, y_real_train, x_real_test, y_real_test

def prep_data(data_key, data_from_torch, data_path, shuffle_data, subsample, sub_balanced_labels):
  x_real_train, y_real_train, x_real_test, y_real_test = load_mnist_data(data_key, data_from_torch)
  gen_data = np.load(data_path)
  x_gen, y_gen = gen_data['data'], gen_data['labels']
  if len(y_gen.shape) == 2:  # remove onehot
    if y_gen.shape[1] == 1:
      y_gen = y_gen.ravel()
    elif y_gen.shape[1] == 10:
      y_gen = np.argmax(y_gen, axis=1)
    else:
      raise ValueError

  if shuffle_data:
    rand_perm = np.random.permutation(y_gen.shape[0])
    x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]

  if subsample < 1.:
    x_gen, y_gen = subsample_data(x_gen, y_gen, subsample, sub_balanced_labels)
    x_real_train, y_real_train = subsample_data(x_real_train, y_real_train, subsample, sub_balanced_labels)

    print(f'training on {subsample * 100.}% of the original syntetic dataset')

  print(f'data ranges: [{np.min(x_real_test)}, {np.max(x_real_test)}], [{np.min(x_real_train)}, '
        f'{np.max(x_real_train)}], [{np.min(x_gen)}, {np.max(x_gen)}]')
  print(f'label ranges: [{np.min(y_real_test)}, {np.max(y_real_test)}], [{np.min(y_real_train)}, '
        f'{np.max(y_real_train)}], [{np.min(y_gen)}, {np.max(y_gen)}]')

  return datasets_colletion_def(x_gen, y_gen, x_real_train, y_real_train, x_real_test, y_real_test)


def test_gen_data(data_log_name, data_key, data_base_dir='logs/gen/', log_results=False, data_path=None,
                  data_from_torch=False, shuffle_data=False, subsample=1., sub_balanced_labels=True,
                  custom_keys=None, skip_slow_models=False, only_slow_models=False,
                  skip_gen_to_real=False, compute_real_to_real=False, compute_real_to_gen=False,
                  print_conf_mat=False, norm_data=False):

  gen_data_dir = os.path.join(data_base_dir, data_log_name)
  log_save_dir = os.path.join(gen_data_dir, 'synth_eval/')
  if data_path is None:
    data_path = os.path.join(gen_data_dir, 'synthetic_mnist.npz')
  datasets_colletion = prep_data(data_key, data_from_torch, data_path, shuffle_data, subsample, sub_balanced_labels)
  mean_acc = test_passed_gen_data(data_log_name, datasets_colletion, log_save_dir, log_results,
                                  subsample, custom_keys, skip_slow_models, only_slow_models,
                                  skip_gen_to_real, compute_real_to_real, compute_real_to_gen,
                                  print_conf_mat, norm_data)
  return mean_acc

def prep_models(custom_keys, skip_slow_models, only_slow_models):
  assert not (skip_slow_models and only_slow_models)

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
  model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}  # setting used in neurips2020 submission
  # model_specs['adaboost'] = {'n_estimators': 100, 'learning_rate': 0.1, 'algorithm': 'SAMME.R'}  best so far
  #  (not used for consistency with old results. change too small to warrant redoing everything)
  model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
  model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
  model_specs['xgboost'] = {'colsample_bytree': 0.1, 'objective': 'multi:softprob', 'n_estimators': 50}

  if custom_keys is not None:
    run_keys = custom_keys.split(',')
  elif skip_slow_models:
    run_keys = [k for k in models.keys() if k not in slow_models]
  elif only_slow_models:
    run_keys = [k for k in models.keys() if k in slow_models]
  else:
    run_keys = models.keys()

  return models, model_specs, run_keys


def normalize_data(x_train, x_test):
  mean = np.mean(x_train)
  sdev = np.std(x_train)
  x_train_normed = (x_train - mean) / sdev
  x_test_normed = (x_test - mean) / sdev
  assert not np.any(np.isnan(x_train_normed)) and not np.any(np.isnan(x_test_normed))

  return x_train_normed, x_test_normed


def model_test_run(model, x_tr, y_tr, x_ts, y_ts, norm_data, acc_str, f1_str):
  x_tr, x_ts = normalize_data(x_tr, x_ts) if norm_data else (x_tr, x_ts)
  model.fit(x_tr, y_tr)

  y_pred = model.predict(x_ts)
  acc = accuracy_score(y_pred, y_ts)
  f1 = f1_score(y_true=y_ts, y_pred=y_pred, average='macro')
  conf = confusion_matrix(y_true=y_ts, y_pred=y_pred)
  acc_str = acc_str + f' {acc}'
  f1_str = f1_str + f' {f1}'
  return acc, f1, conf, acc_str, f1_str

def test_passed_gen_data(data_log_name, datasets_colletion, log_save_dir, log_results=False,
                         subsample=1., custom_keys=None, skip_slow_models=False, only_slow_models=False,
                         skip_gen_to_real=False, compute_real_to_real=False, compute_real_to_gen=False,
                         print_conf_mat=False, norm_data=False):
  if data_log_name is not None:
    print(f'processing {data_log_name}')

  if log_results:
    os.makedirs(log_save_dir, exist_ok=True)

  models, model_specs, run_keys = prep_models(custom_keys, skip_slow_models, only_slow_models)

  g_to_r_acc_summary = []
  dc = datasets_colletion
  for key in run_keys:
    print(f'Model: {key}')
    a_str, f_str = 'acc:', 'f1:'

    if not skip_gen_to_real:
      model = models[key](**model_specs[key])
      g_to_r_acc, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model, dc.x_gen, dc.y_gen,
                                                                        dc.x_real_test, dc.y_real_test,
                                                                        norm_data, a_str + 'g2r', f_str + 'g2r')
      g_to_r_acc_summary.append(g_to_r_acc)
    else:
      g_to_r_acc, g_to_r_f1, g_to_r_conf = -1, -1, -np.ones((10, 10))

    if compute_real_to_real:
      model = models[key](**model_specs[key])
      base_acc, base_f1, base_conf, a_str, f_str = model_test_run(model,
                                                                  dc.x_real_train, dc.y_real_train,
                                                                  dc.x_real_test, dc.y_real_test,
                                                                  norm_data, a_str + 'r2r', f_str + 'r2r')
    else:
      base_acc, base_f1, base_conf = -1, -1, -np.ones((10, 10))

    if compute_real_to_gen:
      model = models[key](**model_specs[key])
      r_to_g_acc, r_to_g_f1, r_to_g_conv, a_str, f_str = model_test_run(model,
                                                                        dc.x_real_train, dc.y_real_train,
                                                                        dc.x_gen[:10000], dc.y_gen[:10000],
                                                                        norm_data, a_str + 'r2g', f_str + 'r2g')
    else:
      r_to_g_acc, r_to_g_f1, r_to_g_conv = -1, -1, -np.ones((10, 10))

    print(a_str)
    print(f_str)
    if print_conf_mat:
      print('gen to real confusion matrix:')
      print(g_to_r_conf)

    if log_results:
      accs = np.asarray([base_acc, g_to_r_acc, r_to_g_acc])
      f1_scores = np.asarray([base_f1, g_to_r_f1, r_to_g_f1])
      conf_mats = np.stack([base_conf, g_to_r_conf, r_to_g_conv])
      file_name = f'sub{subsample}_{key}_log'
      np.savez(os.path.join(log_save_dir, file_name), accuracies=accs, f1_scores=f1_scores, conf_mats=conf_mats)

  print('acc summary:')
  for acc in g_to_r_acc_summary:
    print(acc)
  mean_acc = np.mean(g_to_r_acc_summary)
  print(f'mean: {mean_acc}')
  return mean_acc


def test_results(data_key, log_name, log_dir, data_tuple, eval_func, skip_downstream_model):
  if data_key in {'digits', 'fashion'}:
    if not skip_downstream_model:
      final_score = test_gen_data(log_name, data_key, subsample=0.1, custom_keys='logistic_reg')
      log_final_score(log_dir, final_score)
  # elif data_key == '2d':
  #   if not skip_downstream_model:
  #     final_score = test_passed_gen_data(log_name, data_tuple, log_save_dir=None, log_results=False,
  #                                        subsample=.1, custom_keys='mlp', compute_real_to_real=True)
  #     log_final_score(log_dir, final_score)
  #   eval_score = eval_func(data_tuple.x_gen, data_tuple.y_gen.flatten())
  #   print(f'Score of evaluation function: {eval_score}')
  #   with open(os.path.join(log_dir, 'eval_score'), 'w') as f:
  #     f.writelines([f'{eval_score}'])
  #
  #   plot_data(data_tuple.x_real_train, data_tuple.y_real_train.flatten(), os.path.join(log_dir, 'plot_train'),
  #             center_frame=True)
  #   plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
  #   plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_sub0.2'), subsample=0.2)
  #   plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_centered'),
  #             center_frame=True)
  #
  #   plot_data_1d(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_norms_hist'))
  # elif data_key == '1d':
  #   plot_data_1d(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
  #   plot_data_1d(data_tuple.x_real_test, data_tuple.y_real_test.flatten(), os.path.join(log_dir, 'plot_data'))

def log_args(log_dir, args):
  """ print and save all args """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, 'args_log'), 'w') as f:
    lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    f.writelines(lines)
    for line in lines:
      print(line.rstrip())
  print('-------------------------------------------')

def log_final_score(log_dir, final_acc):
  """ print and save all args """
  os.makedirs(log_dir, exist_ok=True)
  with open(os.path.join(log_dir, 'final_score'), 'w') as f:
      lines = [f'acc: {final_acc}\n']
      f.writelines(lines)


def denormalize(mnist_mat):
  mnist_mean = 0.1307
  mnist_sdev = 0.3081
  return np.clip(mnist_mat * mnist_sdev + mnist_mean, a_min=0., a_max=1.)

def save_img(save_file, img):
  plt.imsave(save_file, img, cmap=cm.gray, vmin=0., vmax=1.)

def plot_mnist_batch(mnist_mat, n_rows, n_cols, save_path, denorm=True, save_raw=True):
  bs = mnist_mat.shape[0]
  n_to_fill = n_rows * n_cols - bs
  mnist_mat = np.reshape(mnist_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows*i:n_rows*(i+1)], n_rows) for i in range(n_cols)]
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)

  if denorm:
     mnist_mat_flat = denormalize(mnist_mat_flat)
  save_img(save_path + '.png', mnist_mat_flat)
  if save_raw:
    np.save(save_path + '_raw.npy', mnist_mat_flat)

def log_gen_data(gen, device, epoch, n_labels, log_dir):
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), n_labels)[:, None].to(device)
  gen_code, _ = gen.get_code(100, device, labels=ordered_labels)
  gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, n_labels, log_dir + f'samples_ep{epoch}', denorm=False)

class FCCondGen(nn.Module):
  def __init__(self, d_code, d_hid, d_out, n_labels, use_sigmoid=True, batch_norm=True):
    super(FCCondGen, self).__init__()
    d_hid = [int(k) for k in d_hid.split(',')]
    assert len(d_hid) < 5

    self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
    self.fc2 = nn.Linear(d_hid[0], d_hid[1])

    self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
    self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
    if len(d_hid) == 2:
      self.fc3 = nn.Linear(d_hid[1], d_out)
    elif len(d_hid) == 3:
      self.fc3 = nn.Linear(d_hid[1], d_hid[2])
      self.fc4 = nn.Linear(d_hid[2], d_out)
      self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
    elif len(d_hid) == 4:
      self.fc3 = nn.Linear(d_hid[1], d_hid[2])
      self.fc4 = nn.Linear(d_hid[2], d_hid[3])
      self.fc5 = nn.Linear(d_hid[3], d_out)
      self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
      self.bn4 = nn.BatchNorm1d(d_hid[3]) if batch_norm else None

    self.use_bn = batch_norm
    self.n_layers = len(d_hid)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.use_bn else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.use_bn else x
    x = self.fc3(self.relu(x))
    if self.n_layers > 2:
      x = self.bn3(x) if self.use_bn else x
      x = self.fc4(self.relu(x))
      if self.n_layers > 3:
        x = self.bn4(x) if self.use_bn else x
        x = self.fc5(self.relu(x))

    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


class ConvCondGen(nn.Module):
  def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, use_sigmoid=True, batch_norm=True):
    super(ConvCondGen, self).__init__()
    self.nc = [int(k) for k in nc_str.split(',')] + [1]  # number of channels
    self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
    d_hid = [int(k) for k in d_hid.split(',')]
    assert len(self.nc) == 3 and len(self.ks) == 2
    self.hw = 7  # image height and width before upsampling
    self.reshape_size = self.nc[0]*self.hw**2
    self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
    self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
    self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
    self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
    self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0]-1)//2)
    self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1]-1)//2)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.bn1 is not None else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.bn2 is not None else x
    # print(x.shape)
    x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
    x = self.upsamp(x)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.conv2(x)
    x = x.reshape(x.shape[0], -1)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code



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


def flatten_features(data):
  if len(data.shape) == 2:
    return data
  else:
    return pt.reshape(data, (data.shape[0], -1))



def flip_mnist_data(dataset):
  data = dataset.data
  flipped_data = 255 - data
  selections = np.zeros(data.shape[0], dtype=np.int)
  selections[:data.shape[0]//2] = 1
  selections = pt.tensor(np.random.permutation(selections), dtype=pt.uint8)
  print(selections.shape, data.shape, flipped_data.shape)
  dataset.data = pt.where(selections[:, None, None], data, flipped_data)



train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])


def get_dataloaders(dataset_key, batch_size, test_batch_size, use_cuda, normalize, synth_spec_string, test_split):
  if dataset_key in {'digits', 'fashion'}:
    train_loader, test_loader, trn_data, tst_data = get_mnist_dataloaders(batch_size, test_batch_size, use_cuda,
                                                                          dataset=dataset_key, normalize=normalize,
                                                                          return_datasets=True)
    n_features = 784
    n_data = 60_000
    n_labels = 10
    eval_func = None
  else:
    raise ValueError

  return train_data_tuple_def(train_loader, test_loader, trn_data, tst_data, n_features, n_data, n_labels, eval_func)


def get_mnist_dataloaders(batch_size, test_batch_size, use_cuda, normalize=False,
                          dataset='digits', data_dir='data', flip=False, return_datasets=False):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transforms_list = [transforms.ToTensor()]
  if dataset == 'digits':
    if normalize:
      mnist_mean = 0.1307
      mnist_sdev = 0.3081
      transforms_list.append(transforms.Normalize((mnist_mean,), (mnist_sdev,)))
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.MNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.MNIST(data_dir, train=False, transform=prep_transforms)
    if flip:
      assert not normalize
      print(pt.max(trn_data.data))
      flip_mnist_data(trn_data)
      flip_mnist_data(tst_data)

    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  elif dataset == 'fashion':
    assert not normalize
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.FashionMNIST(data_dir, train=False, transform=prep_transforms)
    if flip:
      print(pt.max(trn_data.data))
      flip_mnist_data(trn_data)
      flip_mnist_data(tst_data)
    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  else:
    raise ValueError

  if return_datasets:
    return train_loader, test_loader, trn_data, tst_data
  else:
    return train_loader, test_loader

