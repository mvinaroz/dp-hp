import os
import matplotlib

matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from aux import plot_mnist_batch, NamedArray
from aggregate_results import collect_results


def dpcgan_plot():
  # loads = np.load('dp_cgan_synth_mnist_eps9.6.npz')
  loads = np.load('reference_dpcgan1_9.6.npz')
  # loads = np.load('ref_dpcgan_fashion5-eps9.6.npz')
  data, labels = loads['data'], loads['labels']

  print(np.sum(labels, axis=0))
  print(np.max(data), np.min(data))

  rand_perm = np.random.permutation(data.shape[0])
  data = data[rand_perm]
  labels = np.argmax(labels[rand_perm], axis=1)

  data_ids = [[], [], [], [], [], [], [], [], [], []]
  n_full = 0
  for idx in range(data.shape[0]):
    l = labels[idx]
    if len(data_ids[l]) < 10:
      data_ids[l].append(idx)
      # print(l)
      if len(data_ids[l]) == 10:
        n_full += 1
        if n_full == 10:
          break

  data_ids = np.asarray(data_ids)
  data_ids = np.reshape(data_ids, (100,))
  plot_mat = data[data_ids]
  plot_mnist_batch(plot_mat, 10, 10, 'dp_cgan_digit_plot', denorm=False, save_raw=False)


def dpgan_plot():
  data = np.load('dpgan_data.npy')

  rand_perm = np.random.permutation(data.shape[0])
  data = data[rand_perm] / 255.

  data = data[:100]
  print(np.max(data), np.min(data))
  plot_mnist_batch(data, 10, 10, 'dpgan_digit_plot', denorm=False, save_raw=False)


def plot_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps',
            'dpmerf-nonprivate']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  # models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
  #           'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results_mean_subsampled.npy')
  print(mean_mat.shape)

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      # plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])
      plt.xticks(sub_ratios[1:][::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        if s == 'real_data':
          continue
        # plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios
        plt.plot(sub_ratios[1:], mean_mat[d_idx, s_idx, 1:, e_idx], label=s)  # don_t show 1.0

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.hlines([0.4, 0.5, 0.6, 0.7, 0.8], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.legend()
      plt.savefig(f'plot_subsampling_{d}_{e}.png')


def mar19_plot_nonprivate_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['()', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'merf-AE-nonprivate', 'merf-DE-nonprivate']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  # models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
  #           'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results_mean_mar19_nonp.npy')

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        if s == '()':
          continue
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.hlines([0.4, 0.5, 0.6, 0.7, 0.8], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.legend()
      plt.savefig(f'mar19_nonp_{d}_{e}.png')


def plot_subsampling_logreg_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps']
  sub_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  model_idx = 1
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  all_mat = np.load('results_full_subsampled.npy')
  mean_mat = np.mean(all_mat, axis=4)[:, :, :, model_idx, :]  # mean over runs, select logreg model

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'plot_{models[model_idx]}_{d}_{e}.png')


def plot_renorm_performance():
  data_ids = ['d']
  setups = ['real', 'base', 'renorm', 'clip']
  sub_ratios = [0.1, 0.01, 0.001]
  eval_metrics = ['accuracies']
  mean_mat = np.load('results_mean_renorm.npy')

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'renorm_plot_{d}_{e}.png')


def mar20_plot_sr_performance():
  data_ids = ['d', 'f']
  setups = ['()',
            'mar19_sr_rff1k_sig50', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff100k_sig50',
            'mar19_sr_rff1k_sig5', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff100k_sig5',
            'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
            'mar19_sr_rff1k_sig0', 'mar19_sr_rff10k_sig0', 'mar19_sr_rff100k_sig0']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']

  sub_ratios = [0.1, 0.01]
  # models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
  #           'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results_mean_mar20_sr.npy')

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        if s == '()':
          continue
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s, color=colors[s_idx])  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'mar20_sr_{d}_{e}.png')


def apr4_plot_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'DP-CGAN eps=9.6', 'DP-MERF eps=1.3', 'DP-MERF eps=2.9', 'DP-MERF eps=9.6',
            'DP-MERF non-DP']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  # models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
  #           'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('results/results_mean_subsampled.npy')
  print(mean_mat.shape)

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}, metric: {e}')
      plt.xscale('log')
      # plt.xticks(sub_ratios[1:][::-1], [str(k * 100) for k in sub_ratios[::-1]])
      plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        if s == 'real_data':
          continue
        print(d_idx, e_idx, s_idx)
        # plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios
        # plt.plot(sub_ratios[1:], mean_mat[d_idx, s_idx, 1:, e_idx], label=s)  # don_t show 1.0
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # do show 1.0

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.hlines([0.45, 0.5, 0.55], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.ylim((0.4, 0.6))
      plt.legend()
      plt.savefig(f'apr4_plot_subsampling_{d}_{e}.png')


def extract_numpy_data_mats():
  def prep_data(dataset):
    x, y = dataset.data.numpy(), dataset.targets.numpy()
    x = np.reshape(x, (-1, 784)) / 255
    return x, y

  x_trn, y_trn = prep_data(datasets.MNIST('data', train=True))
  x_tst, y_tst = prep_data(datasets.MNIST('data', train=False))
  np.savez('data/MNIST/numpy_dmnist.npz', x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst)

  x_trn, y_trn = prep_data(datasets.FashionMNIST('data', train=True))
  x_tst, y_tst = prep_data(datasets.FashionMNIST('data', train=False))
  np.savez('data/FashionMNIST/numpy_fmnist.npz', x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst)


def spot_synth_mnist_mar19():
  rff = [1, 10, 100]
  sig = ['50', '5', '2.5', '0']
  dat = ['d', 'f']
  run = [0, 1, 2, 3, 4]
  for f in rff:
    for s in sig:
      for d in dat:
        for r in run:
          path = f'logs/gen/mar19_sr_rff{f}k_sig{s}_{d}{r}/synthetic_mnist.npz'
          if not os.path.isfile(path):
            print(f'{path} not found')


def plot_with_variance(x, y, color, label, alpha=0.1):
  """
  assume y is of shape (x_settings, runs to average)
  """
  means_y = np.mean(y, axis=1)
  sdevs_y = np.std(y, axis=1)
  plt.plot(x, means_y, 'o-', label=label, color=color)
  plt.fill_between(x, means_y-sdevs_y, means_y+sdevs_y, alpha=alpha, color=color)
  # plt.errorbar(x, means_y, yerr=sdevs_y,
  #              fmt='.',
  #              # ls='dotted',
  #              color=color,
  #              ecolor=color,
  #              uplims=True,
  #              lolims=True,
  #              elinewidth=3, capsize=0,
  #              alpha=0.3)


def mar24_plot_selected_sr():
  data_ids = ['d', 'f']
  setups = ['mar19_sr_rff1k_sig50', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff100k_sig50',
            'mar19_sr_rff1k_sig5', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff100k_sig5',
            'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
            'mar19_sr_rff1k_sig0', 'mar19_sr_rff10k_sig0', 'mar19_sr_rff100k_sig0']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  metric = 'accuracies'
  sub_ratios = [0.1, 0.01]

  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  _, _, sr_array, _, _, sr_d_array, sr_f_array = collect_results()

  for d_idx, d in enumerate(data_ids):

    plt.figure()
    plt.title(f'data: {d}, metric: {metric}')
    plt.xscale('log')
    plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])

    for s_idx, s in enumerate(setups):
      # print(d, s, models)
      sub_mat = sr_array.get({'data_ids': [d], 'setups': [s], 'models': models, 'eval_metrics': [metric]})
      print(sub_mat.shape)
      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=s)
      # plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s, color=colors[s_idx])  # plot over ratios

    plt.xlabel('% of data')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(f'mar24_sr_var_{d}_{metric}.png')


def mar25_plot_selected_sr():
  sr_d_setups = ['mar19_sr_rff10k_sig50', 'mar19_sr_rff1k_sig5', 'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff1k_sig0']
  sr_f_setups = ['mar19_sr_rff100k_sig50', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff10k_sig0']

  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  metric = 'accuracies'
  # metric = 'f1_scores'
  sub_ratios = [1.0, 0.1, 0.01]

  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  _, _, _, _, _, sr_d_array, sr_f_array = collect_results()

  # digit plot
  plt.figure(), plt.title(f'data: d, metric: {metric}'), plt.xscale('log')
  plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])
  for s_idx, s in enumerate(sr_d_setups):
    sub_mat = sr_d_array.get({'data_ids': ['d'], 'setups': [s], 'models': models, 'eval_metrics': [metric]})
    print('sr_d:', s)
    by_model = np.mean(sub_mat, axis=2)[0, :]  # average over runs, select 1.0 subsampling
    for v in by_model:
      print(v)
    sub_mat = np.mean(sub_mat, axis=1)  # average over models
    plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=s)
  plt.xlabel('% of data'), plt.ylabel(metric), plt.legend()
  plt.savefig(f'plots/mar25_sr_var_d_{metric}.png')

  # digit plot
  plt.figure(), plt.title(f'data: f, metric: {metric}'), plt.xscale('log')
  plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])
  for s_idx, s in enumerate(sr_f_setups):
    sub_mat = sr_f_array.get({'data_ids': ['f'], 'setups': [s], 'models': models, 'eval_metrics': [metric]})
    print('sr_f:', s)
    by_model = np.mean(sub_mat, axis=2)[0, :]  # average over runs, select 1.0 subsampling
    for v in by_model:
      print(v)
    sub_mat = np.mean(sub_mat, axis=1)  # average over models
    plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=s)

  plt.xlabel('% of data'), plt.ylabel(metric), plt.legend()
  plt.savefig(f'plots/mar25_sr_var_f_{metric}.png')


def mar25_plot_nonprivate():
  queried_setups = ['real_data',
                    # 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps',
                    # 'dpmerf-high-eps', 'dpmerf-nonprivate_sb'
                    # '(np)',
                    'dpmerf-nonprivate', 'dpcgan-nonprivate', 'mar19_nonp_ae', 'mar19_nonp_de']

  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  metric = 'accuracies'
  # metric = 'f1_scores'
  data_ids = ['d', 'f']
  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  _, _, _, sb_np_array, _, _, _ = collect_results()

  # digit plot
  for d_id in data_ids:
    plt.figure(), plt.title(f'data: {d_id}, metric: {metric}'), plt.xscale('log')
    plt.xticks(sub_ratios[::-1], [str(k * 100) for k in sub_ratios[::-1]])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = sb_np_array.get({'data_ids': [d_id], 'setups': [s], 'models': models, 'eval_metrics': [metric]})
      print(f'sr_{d_id}:', s)
      by_model = np.mean(sub_mat, axis=2)[6, :]  # average over runs, select 1.0 subsampling
      for v in by_model:
        print(v)
      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=s)
    plt.xlabel('% of data'), plt.ylabel(metric), plt.legend()
    plt.savefig(f'plots/mar25_nonp_var_{d_id}_{metric}.png')


def apr4_plot_subsampling_performance_variance(plot_var=True):
  data_ids = ['MNIST', 'FashionMNIST']
  setups = ['real_data', 'DP-CGAN eps=9.6', 'DP-MERF eps=1.3', 'DP-MERF eps=2.9', 'DP-MERF eps=9.6',
            'DP-MERF non-DP']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  data_used = ['60k', '30k', '12k', '6k', '3k', '1.2k', '600', '300', '120', '60']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  # models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
  #           'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracy', 'f1_score']
  mean_mat = np.load('results/results_full_mar25_subsampled.npy')
  print(mean_mat.shape)
  mean_mat = np.mean(mean_mat, axis=3)

  for d_idx, d in enumerate(data_ids):
    for e_idx, e in enumerate(eval_metrics):
      plt.figure()
      plt.title(f'data: {d}')
      plt.xscale('log')
      plt.xticks(sub_ratios[::-1], data_used[::-1])

      for s_idx, s in enumerate(setups):
        if s == 'real_data':
          continue
        if plot_var:
          print('yes')
          sub_mat = mean_mat[d_idx, s_idx, :, :, e_idx]
          plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=s)
        else:
          sub_mat = np.mean(mean_mat[d_idx, s_idx, :, :, e_idx], axis=1)
          plt.plot(sub_ratios, sub_mat, label=s, color=colors[s_idx])  # do show 1.0

      plt.xlabel('# samples generated')
      plt.ylabel(e)
      if d == 'MNIST':
        plt.yticks([0.4, 0.45, 0.5, 0.55, 0.6])
        plt.hlines([0.45, 0.5, 0.55], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
        plt.ylim((0.4, 0.6))
      else:
        plt.hlines([0.3, 0.4, 0.5], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
        plt.ylim((0.25, 0.6))
      plt.legend(loc='lower right')

      plt.savefig(f'apr4_{"var_" if plot_var else ""}plot_subsampling_{d}_{e}.png')


def apr6_replot_nonprivate(plot_var=False):
  queried_setups = ['real_data', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'mar19_nonp_ae']
  setup_names = ['real data', 'DP-MERF non-DP', 'DP-CGAN non-DP', 'DP-MERF-AE non-DP']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  data_used = ['60k', '30k', '12k', '6k', '3k', '1.2k', '600', '300', '120', '60']
  metric = 'accuracies'
  # metric = 'f1_scores'
  data_ids = ['d', 'f']
  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  sb_np_array = collect_results()['sb_np']

  # digit plot
  for d_id in data_ids:
    plt.figure(), plt.title(f'data: {"MNIST" if d_id == "d" else "FashionMNIST"}'), plt.xscale('log')
    # plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])
    plt.xticks(sub_ratios[::-1], data_used[::-1])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = sb_np_array.get({'data_ids': [d_id], 'setups': [s], 'models': models, 'eval_metrics': [metric]})

      sub_mat = np.mean(sub_mat, axis=1)  # average over models

      if plot_var:
        plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.mean(sub_mat, axis=1)  # average over runs
        plt.plot(sub_ratios, sub_mat, label=s, color=colors[s_idx])  # do show 1.0

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
      # plt.hlines([0.5, 0.6, 0.7, 0.8], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.ylim((0.4, 0.9))
    else:
      plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8])
      # plt.hlines([0.5, 0.6, 0.7], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.ylim((0.4, 0.8))
    plt.legend(loc='upper left')
    plt.savefig(f'plots/apr4_nonp_{"var_" if plot_var else ""}{d_id}_{metric}.png')


def apr6_plot_overfit_conv(plot_var=False):
  queried_setups = ['apr4_sr_conv_sig_0', 'apr4_sr_conv_sig_2.5', 'apr4_sr_conv_sig_5',
                    'apr4_sr_conv_sig_10', 'apr4_sr_conv_sig_25', 'apr4_sr_conv_sig_50']
  setup_names = ['non-DP', 'eps=2', 'eps=1', 'eps=0.5', 'eps=0.2', 'eps=0.1']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  sub_ratios = [1.0, 0.1, 0.01, 0.001]
  data_used = ['60k', '6k', '600', '60']
  metric = 'accuracies'
  data_ids = ['d', 'f']
  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  sb_np_array = collect_results()['sr_conv_apr4']

  # digit plot
  for d_id in data_ids:
    plt.figure()
    plt.title(f'DP-MERF single release + convolutional generator: {"MNIST" if d_id == "d" else "FashionMNIST"}')
    plt.xscale('log')
    # plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])
    plt.xticks(sub_ratios[::-1], data_used[::-1])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = sb_np_array.get({'data_ids': [d_id], 'setups': [s], 'models': models, 'eval_metrics': [metric]})

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      if plot_var:
        plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(sub_ratios, sub_mat, label=s, color=colors[s_idx])  # do show 1.0

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      pass
      # plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
      # plt.hlines([0.5, 0.6, 0.7, 0.8], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      # plt.ylim((0.4, 0.9))
    else:
      pass
      # plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8])
      # plt.hlines([0.5, 0.6, 0.7], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      # plt.ylim((0.4, 0.8))
    plt.legend(loc='upper left')
    plt.savefig(f'plots/apr4_sr_conv_{"var_" if plot_var else ""}{d_id}_{metric}.png')


def apr6_plot_better_conv(plot_var=False):
  queried_setups = [  # 'real_data',
                    'dpcgan', 'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5', 'apr6_sr_conv_sig_25']
  setup_names = [  # 'real data',
                 'DP-CGAN $\epsilon=9.6$', 'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                 'DP-MERF $\epsilon=0.2$']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  data_used = ['60k', '30k', '12k', '6k', '3k', '1.2k', '600', '300', '120', '60']
  metric = 'accuracies'
  data_ids = ['d', 'f']
  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  ar_dict = collect_results()
  sr_conv_array = ar_dict['sr_conv_apr6']
  sb_array = ar_dict['sb']
  merged_array = sr_conv_array.merge(sb_array, merge_dim='setups')

  # digit plot
  for d_id in data_ids:
    plt.figure()
    plt.title(f'DP-MERF single release + convolutional generator: {"MNIST" if d_id == "d" else "FashionMNIST"}')
    plt.xscale('log')
    # plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])
    plt.xticks(sub_ratios[::-1], data_used[::-1])
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [d_id], 'setups': [s], 'models': models, 'eval_metrics': [metric]})

      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      if plot_var:
        plot_with_variance(sub_ratios, sub_mat, color=colors[s_idx], label=setup_names[s_idx])
      else:
        sub_mat = np.median(sub_mat, axis=1)
        plt.plot(sub_ratios, sub_mat, label=s, color=colors[s_idx])  # do show 1.0

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')
    if d_id == 'd':
      pass
      plt.yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
      plt.hlines([0.45, 0.5, 0.55, 0.6, 0.65], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.ylim((0.40, 0.7))
    else:
      pass
      plt.yticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
      plt.hlines([0.4, 0.45, 0.5, 0.55, 0.6], xmin=sub_ratios[-1], xmax=sub_ratios[0], linestyles='dotted')
      plt.ylim((0.35, 0.65))
    plt.legend(loc='upper left')
    plt.savefig(f'plots/apr6_sr_conv_{"var_" if plot_var else ""}{d_id}_{metric}_with_dpcgan.png')



if __name__ == '__main__':
  # collect_synth_benchmark_results()
  # aggregate_subsample_tests_paper_setups()
  # aggregate_subsample_tests_renorm_test()
  # aggregate_subsample_tests_paper_setups_redo()
  # plot_subsampling_performance()
  # plot_subsampling_logreg_performance()
  # plot_renorm_performance()
  # extract_numpy_data_mats()
  # aggregate_mar12_setups()
  # aggregate_mar19_nonp()
  # spot_synth_mnist_mar19()
  # mar19_plot_nonprivate_subsampling_performance()
  # aggregate_mar20_sr()
  # mar20_plot_sr_performance()
  # mar24_plot_selected_sr()
  # aggregate_mar25_sr()
  # mar25_plot_selected_sr()
  # mar25_plot_nonprivate()
  # aggregate_apr4_sr_conv()
  # apr4_plot_subsampling_performance_variance()

  # apr6_replot_nonprivate(plot_var=True)
  # apr6_replot_nonprivate(plot_var=False)

  # apr6_plot_overfit_conv(plot_var=False)
  # apr6_plot_overfit_conv(plot_var=True)
  apr6_plot_better_conv(plot_var=True)