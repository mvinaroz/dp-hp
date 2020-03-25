import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from aux import plot_mnist_batch, NamedArray

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


def real_nosub_results():  # copied from doc
  digit_acc = [[0.925, 0.925, 0.925, 0.925, 0.925], [0.968, 0.968, 0.970, 0.968, 0.970],
               [0.555, 0.555, 0.555, 0.555, 0.555], [0.842, 0.842, 0.842, 0.842, 0.842],
               [0.918, 0.918, 0.918, 0.918, 0.918], [0.876, 0.879, 0.877, 0.876, 0.879],
               [0.879, 0.879, 0.879, 0.879, 0.879], [0.729, 0.729, 0.729, 0.729, 0.729],
               [0.978, 0.978, 0.978, 0.979, 0.978], [0.927, 0.929, 0.932, 0.926, 0.927],
               [0.908, 0.909, 0.909, 0.909, 0.910], [0.912, 0.912, 0.912, 0.912, 0.912]]

  digit_f1 = [[0.924, 0.924, 0.924, 0.924, 0.924], [0.968, 0.968, 0.969, 0.968, 0.970],
              [0.508, 0.508, 0.508, 0.508, 0.508], [0.840, 0.840, 0.840, 0.840, 0.840],
              [0.917, 0.917, 0.917, 0.917, 0.917], [0.875, 0.878, 0.876, 0.875, 0.878],
              [0.878, 0.878, 0.878, 0.878, 0.878], [0.723, 0.723, 0.723, 0.723, 0.723],
              [0.978, 0.978, 0.978, 0.978, 0.978], [0.926, 0.928, 0.931, 0.924, 0.926],
              [0.907, 0.908, 0.908, 0.907, 0.908], [0.912, 0.912, 0.912, 0.912, 0.912]]

  fashion_acc = [[0.844, 0.844, 0.844, 0.844, 0.844], [0.875, 0.874, 0.875, 0.875, 0.876],
                 [0.585, 0.585, 0.585, 0.585, 0.585], [0.648, 0.648, 0.648, 0.648, 0.648],
                 [0.839, 0.839, 0.839, 0.839, 0.839], [0.791, 0.787, 0.791, 0.790, 0.791],
                 [0.799, 0.799, 0.799, 0.799, 0.799], [0.561, 0.561, 0.561, 0.561, 0.561],
                 [0.878, 0.876, 0.883, 0.884, 0.877], [0.842, 0.839, 0.841, 0.839, 0.842],
                 [0.834, 0.832, 0.833, 0.837, 0.833], [0.826, 0.826, 0.826, 0.826, 0.826]]

  fashion_f1 = [[0.843, 0.843, 0.843, 0.843, 0.843], [0.873, 0.873, 0.874, 0.874, 0.875],
                [0.556, 0.556, 0.556, 0.556, 0.556], [0.639, 0.639, 0.639, 0.639, 0.639],
                [0.836, 0.837, 0.836, 0.836, 0.837], [0.792, 0.788, 0.791, 0.790, 0.792],
                [0.800, 0.800, 0.800, 0.800, 0.800], [0.546, 0.546, 0.546, 0.546, 0.546],
                [0.878, 0.876, 0.883, 0.883, 0.877], [0.840, 0.838, 0.839, 0.837, 0.840],
                [0.833, 0.832, 0.833, 0.836, 0.833], [0.824, 0.824, 0.824, 0.824, 0.824]]

  return np.asarray(digit_acc), np.asarray(digit_f1), np.asarray(fashion_acc), np.asarray(fashion_f1)


def aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str,
                              load_real_nosub=False):
  all_the_results = np.zeros((len(data_ids),
                              len(setups) + 1,  # 0 goes to real data
                              len(sub_ratios),
                              len(models),
                              len(runs),
                              len(eval_metrics)))
  for d_idx, d in enumerate(data_ids):
    for s_idx, s in enumerate(setups):
      for r_idx, r in enumerate(sub_ratios):
        for m_idx, m in enumerate(models):
          for run_idx, run in enumerate(runs):
            load_file = f'logs/gen/{s}{d}{run}/synth_eval/sub{r}_{m}_log.npz'
            alternate_file = f'logs/gen/{s}{d}{run}/synth_eval/{m}_log.npz'
            if os.path.isfile(load_file):
              mat = np.load(load_file)
            elif r == '1.0' and os.path.isfile(alternate_file):
              mat = np.load(alternate_file)
              # failed to load logs/gen/dpcgan-d0/synth_eval/sub1.0_logistic_reg_log.npz
            else:
              print('failed to load', load_file)
              continue
            for e_idx, e in enumerate(eval_metrics):
              score = mat[e][1]
              all_the_results[d_idx, s_idx + 1, r_idx, m_idx, run_idx, e_idx] = score

              if s == setup_with_real_data:
                all_the_results[d_idx, 0, r_idx, m_idx, run_idx, e_idx] = mat[e][0]

  if load_real_nosub:
    digit_acc, digit_f1, fashion_acc, fashion_f1 = real_nosub_results()
    all_the_results[0, 0, 0, :, :, 0] = digit_acc
    all_the_results[0, 0, 0, :, :, 1] = digit_f1
    all_the_results[1, 0, 0, :, :, 0] = fashion_acc
    all_the_results[1, 0, 0, :, :, 1] = fashion_f1

  np.save(f'results_full_{save_str}', all_the_results)

  for e_idx, e in enumerate(eval_metrics):
    print('metric:', e)
    for d_idx, d in enumerate(data_ids):
      print('data:', d)
      for s_idx, s in enumerate(['real_data'] + setups):
        print('setup:', s)
        for r_idx, r in enumerate(sub_ratios):
          print('sub_ratio:', r, 'mean:', np.mean(all_the_results[d_idx, s_idx, r_idx, :, :, e_idx]))
          # print(all_the_results[d_idx, s_idx, r_idx, :, :, e_idx])

  np.save(f'results_mean_{save_str}', np.mean(all_the_results, axis=(3, 4)))


def aggregate_subsample_tests_paper_setups():
  data_ids = ['d', 'f']
  setups = ['dpcgan-', 'dpmerf-ae-', 'dpmerf-low-eps-', 'dpmerf-med-eps-', 'dpmerf-high-eps-',
            'dpmerf-nonprivate-']
  sub_ratios = ['1.0', '0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = 'dpmerf-high-eps'
  save_str = 'subsampled'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)


def aggregate_subsample_tests_paper_setups_redo():
  data_ids = ['d', 'f']
  setups = ['dpcgan-', 'dpmerf-low-eps-', 'dpmerf-med-eps-', 'dpmerf-high-eps-',
            'dpmerf-nonprivate-']
  sub_ratios = ['1.0', '0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = 'dpmerf-high-eps-'
  save_str = 'mar25_subsampled'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str, load_real_nosub=True)

# def aggregate_subsample_tests_renorm_test():
#   data_ids = ['']
#   setups = ['dmnist_rescale_release_off_', 'dmnist_rescale_release_on_', 'dmnist_rescale_release_clip_']
#   sub_ratios = ['0.1', '0.01', '0.001']
#   models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
#             'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
#   runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#   eval_metrics = ['accuracies']
#   setup_with_real_data = 'dmnist_rescale_release_off_'
#   save_str = 'renorm'
#   aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)

def aggregate_mar19_nonp():
  data_ids = ['d', 'f']
  setups = ['dpmerf-nonprivate-', 'dpcgan-nonprivate-', 'mar19_nonp_ae_', 'mar19_nonp_de_']
  sub_ratios = ['1.0', '0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = 'dpmerf-high-eps'
  save_str = 'mar19_nonp'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)


def aggregate_mar20_sr():
  data_ids = ['d', 'f']
  setups = ['mar19_sr_rff1k_sig50_',  'mar19_sr_rff10k_sig50_',  'mar19_sr_rff100k_sig50_',
            'mar19_sr_rff1k_sig5_',   'mar19_sr_rff10k_sig5_',   'mar19_sr_rff100k_sig5_',
            'mar19_sr_rff1k_sig2.5_', 'mar19_sr_rff10k_sig2.5_', 'mar19_sr_rff100k_sig2.5_',
            'mar19_sr_rff1k_sig0_',   'mar19_sr_rff10k_sig0_',   'mar19_sr_rff100k_sig0_']
  sub_ratios = ['0.1', '0.01']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = ''
  save_str = 'mar20_sr'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)


def aggregate_mar25_sr():
  sub_ratios = ['1.0', '0.1', '0.01']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = ''

  data_ids = ['d']
  setups = ['mar19_sr_rff10k_sig50_', 'mar19_sr_rff1k_sig5_', 'mar19_sr_rff1k_sig2.5_', 'mar19_sr_rff1k_sig0_']
  save_str = 'mar25_sr_digit'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)

  data_ids = ['f']
  setups = ['mar19_sr_rff100k_sig50_', 'mar19_sr_rff10k_sig5_', 'mar19_sr_rff10k_sig2.5_', 'mar19_sr_rff10k_sig0_']
  save_str = 'mar25_sr_fashion'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)


def aggregate_mar12_setups():
  data_ids = ['d', 'f']
  setups = ['mar12_dpmerf_de_']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  # setup_with_real_data = 'dpmerf-high-eps'
  save_str = 'mar12'
  # aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)
  all_the_results = np.zeros((len(data_ids),
                              len(setups) + 1,  # 0 goes to real data
                              len(models),
                              len(runs),
                              len(eval_metrics)))
  for d_idx, d in enumerate(data_ids):
    for s_idx, s in enumerate(setups):
      for m_idx, m in enumerate(models):
        for run_idx, run in enumerate(runs):
          load_file = f'logs/gen/{s}{d}{run}/synth_eval/{m}_log.npz'
          mat = np.load(load_file)
          for e_idx, e in enumerate(eval_metrics):
            score = mat[e][1]
            all_the_results[d_idx, s_idx + 1, m_idx, run_idx, e_idx] = score

  np.save(f'full_results_{save_str}', all_the_results)

  print('models:')
  for model in models:
    print(model)
  print('acc')
  for d_idx, d in enumerate(data_ids):
    print('dataset', d)
    for run_idx, run in enumerate(runs):
      print('run', run)
      for m_idx, m in enumerate(models):
        print(all_the_results[d_idx, 1, m_idx, run_idx, 0])

  print('f1')
  for d_idx, d in enumerate(data_ids):
    print('dataset', d)
    for run_idx, run in enumerate(runs):
      print('run', run)
      for m_idx, m in enumerate(models):
        print(all_the_results[d_idx, 1, m_idx, run_idx, 1])

  for e_idx, e in enumerate(eval_metrics):
    print('metric:', e)
    for d_idx, d in enumerate(data_ids):
      print('data:', d)
      for s_idx, s in enumerate(['real_data'] + setups):
        print('setup:', s)
        print('mean:', np.mean(all_the_results[d_idx, s_idx, :, :, e_idx]))

  np.save(f'mean_results_{save_str}', np.mean(all_the_results, axis=(3, 4)))


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
      plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])

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
      plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])

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
      plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s)  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'renorm_plot_{d}_{e}.png')


def mar20_plot_sr_performance():
  data_ids = ['d', 'f']
  setups = ['()',
            'mar19_sr_rff1k_sig50',  'mar19_sr_rff10k_sig50',  'mar19_sr_rff100k_sig50',
            'mar19_sr_rff1k_sig5',   'mar19_sr_rff10k_sig5',   'mar19_sr_rff100k_sig5',
            'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
            'mar19_sr_rff1k_sig0',   'mar19_sr_rff10k_sig0',   'mar19_sr_rff100k_sig0']
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
      plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])

      for s_idx, s in enumerate(setups):
        if s == '()':
          continue
        plt.plot(sub_ratios, mean_mat[d_idx, s_idx, :, e_idx], label=s, color=colors[s_idx])  # plot over ratios

      plt.xlabel('% of data')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(f'mar20_sr_{d}_{e}.png')


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


def collect_results():
  # read out subsampled, mar19_nonp and mar20_sr and combine them
  data_ids = ['d', 'f']
  eval_metrics = ['accuracies', 'f1_scores']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]

  sb_setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps',
               'dpmerf-nonprivate_1']
  sb_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  sb_mat = np.load('results_full_subsampled.npy')

  np_setups = ['(np)', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'mar19_nonp_ae', 'mar19_nonp_de']
  np_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  np_mat = np.load('results_full_mar19_nonp.npy')

  sr_setups = ['(sr)',
               'mar19_sr_rff1k_sig50', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff100k_sig50',
               'mar19_sr_rff1k_sig5', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff100k_sig5',
               'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
               'mar19_sr_rff1k_sig0', 'mar19_sr_rff10k_sig0', 'mar19_sr_rff100k_sig0']
  sr_ratios = [0.1, 0.01]
  sr_mat = np.load('results_full_mar20_sr.npy')

  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']  # order matters!
  base_idx_names = {'data_ids': data_ids, 'models': models, 'runs': runs, 'eval_metrics': eval_metrics}

  sb_idx_names = {'setups': sb_setups, 'sub_ratios': sb_ratios}
  sb_idx_names.update(base_idx_names)
  sb_array = NamedArray(sb_mat, dim_names, sb_idx_names)

  np_idx_names = {'setups': np_setups, 'sub_ratios': np_ratios}
  np_idx_names.update(base_idx_names)
  np_array = NamedArray(np_mat, dim_names, np_idx_names)

  sr_idx_names = {'setups': sr_setups, 'sub_ratios': sr_ratios}
  sr_idx_names.update(base_idx_names)
  sr_array = NamedArray(sr_mat, dim_names, sr_idx_names)

  sb_np_array = sb_array.merge(np_array, merge_dim='setups')

  all_array = sb_np_array.merge(sr_array, merge_dim='setups')

  return sb_array, np_array, sr_array, sb_np_array, all_array


def plot_with_variance(x, y, color, label):
  """
  assume y is of shape (x_settings, runs to average)
  """
  means_y = np.mean(y, axis=1)
  sdevs_y = np.std(y, axis=1)
  plt.plot(x, means_y, label=label, color=color)
  plt.fill_between(x, means_y-sdevs_y, means_y+sdevs_y, alpha=0.2, color=color)


def mar24_plot_selected_sr():
  data_ids = ['d', 'f']
  # setups = ['()',
  #           'mar19_sr_rff1k_sig50',  'mar19_sr_rff10k_sig50',  'mar19_sr_rff100k_sig50',
  #           'mar19_sr_rff1k_sig5',   'mar19_sr_rff10k_sig5',   'mar19_sr_rff100k_sig5',
  #           'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
  #           'mar19_sr_rff1k_sig0',   'mar19_sr_rff10k_sig0',   'mar19_sr_rff100k_sig0']
  setups = ['mar19_sr_rff1k_sig50',  'mar19_sr_rff10k_sig50',  'mar19_sr_rff100k_sig50',
            'mar19_sr_rff1k_sig5',   'mar19_sr_rff10k_sig5',   'mar19_sr_rff100k_sig5',
            'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
            'mar19_sr_rff1k_sig0',   'mar19_sr_rff10k_sig0',   'mar19_sr_rff100k_sig0']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray', 'tab:pink', 'limegreen', 'yellow']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  metric = 'accuracies'
  sub_ratios = [0.1, 0.01]

  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']

  _, _, sr_array, _, _ = collect_results()

  for d_idx, d in enumerate(data_ids):

    plt.figure()
    plt.title(f'data: {d}, metric: {metric}')
    plt.xscale('log')
    plt.xticks(sub_ratios[::-1], [str(k*100) for k in sub_ratios[::-1]])

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


if __name__ == '__main__':
  # dpcgan_plot()
  # dpgan_plot()
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
  aggregate_mar25_sr()