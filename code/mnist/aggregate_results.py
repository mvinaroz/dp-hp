import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import numpy as np
from aux import NamedArray


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


def aggregate_apr4_sr_conv():
  data_ids = ['d', 'f']

  setups = ['apr4_sr_conv_bigmodel_sig_0_', 'apr4_sr_conv_bigmodel_sig_2.5_', 'apr4_sr_conv_bigmodel_sig_5_',
            'apr4_sr_conv_bigmodel_sig_10_', 'apr4_sr_conv_bigmodel_sig_25_', 'apr4_sr_conv_bigmodel_sig_50_',
            'apr4_sr_conv_sig_0_', 'apr4_sr_conv_sig_2.5_', 'apr4_sr_conv_sig_5_',
            'apr4_sr_conv_sig_10_', 'apr4_sr_conv_sig_25_', 'apr4_sr_conv_sig_50_']
  sub_ratios = ['1.0', '0.1', '0.01', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = ''
  save_str = 'apr4_sr_conv'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)


def aggregate_apr6_sr_conv():
  data_ids = ['d', 'f']

  setups = ['apr6_sr_conv_sig0_', 'apr6_sr_conv_sig5_', 'apr6_sr_conv_sig25_']
  sub_ratios = ['1.0', '0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = ''
  save_str = 'apr6_sr_conv_partial'
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


def collect_results():
  # read out subsampled, mar19_nonp and mar20_sr and combine them
  data_ids = ['d', 'f']
  eval_metrics = ['accuracies', 'f1_scores']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']  # order matters!
  base_idx_names = {'data_ids': data_ids, 'models': models, 'runs': runs, 'eval_metrics': eval_metrics}

  sb_setups = ['real_data', 'dpcgan',
               # 'dpmerf-ae',
               'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps', 'dpmerf-nonprivate_sb']
  sb_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  sb_mat = np.load('results/results_full_mar25_subsampled.npy')
  sb_idx_names = {'setups': sb_setups, 'sub_ratios': sb_ratios}
  sb_idx_names.update(base_idx_names)
  sb_array = NamedArray(sb_mat, dim_names, sb_idx_names)

  np_setups = ['(np)', 'dpmerf-nonprivate', 'dpcgan-nonprivate', 'mar19_nonp_ae', 'mar19_nonp_de']
  np_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  np_mat = np.load('results/results_full_mar19_nonp.npy')
  np_idx_names = {'setups': np_setups, 'sub_ratios': np_ratios}
  np_idx_names.update(base_idx_names)
  np_array = NamedArray(np_mat, dim_names, np_idx_names)

  sr_setups = ['(sr)',
               'mar19_sr_rff1k_sig50', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff100k_sig50',
               'mar19_sr_rff1k_sig5', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff100k_sig5',
               'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff100k_sig2.5',
               'mar19_sr_rff1k_sig0', 'mar19_sr_rff10k_sig0', 'mar19_sr_rff100k_sig0']
  sr_ratios = [0.1, 0.01]
  sr_mat = np.load('results/results_full_mar20_sr.npy')
  sr_idx_names = {'setups': sr_setups, 'sub_ratios': sr_ratios}
  sr_idx_names.update(base_idx_names)
  sr_array = NamedArray(sr_mat, dim_names, sr_idx_names)

  sr_d_data_ids = ['d']
  sr_d_setups = ['(srd)', 'mar19_sr_rff10k_sig50', 'mar19_sr_rff1k_sig5', 'mar19_sr_rff1k_sig2.5', 'mar19_sr_rff1k_sig0']
  sr_d_ratios = [1.0, 0.1, 0.01]
  sr_d_mat = np.load('results/results_full_mar25_sr_digit.npy')
  sr_d_idx_names = {'setups': sr_d_setups, 'sub_ratios': sr_d_ratios}
  sr_d_idx_names.update(base_idx_names)
  sr_d_idx_names['data_ids'] = sr_d_data_ids
  sr_d_array = NamedArray(sr_d_mat, dim_names, sr_d_idx_names)

  sr_f_data_ids = ['f']
  sr_f_setups = ['(srf)', 'mar19_sr_rff100k_sig50', 'mar19_sr_rff10k_sig5', 'mar19_sr_rff10k_sig2.5', 'mar19_sr_rff10k_sig0']
  sr_f_ratios = [1.0, 0.1, 0.01]
  sr_f_mat = np.load('results/results_full_mar25_sr_fashion.npy')
  sr_f_idx_names = {'setups': sr_f_setups, 'sub_ratios': sr_f_ratios}
  sr_f_idx_names.update(base_idx_names)
  sr_f_idx_names['data_ids'] = sr_f_data_ids
  sr_f_array = NamedArray(sr_f_mat, dim_names, sr_f_idx_names)

  sr_conv_apr4_data_ids = ['d', 'f']
  sr_conv_apr4_setups = ['(apr4)',
                         'apr4_sr_conv_bigmodel_sig_0', 'apr4_sr_conv_bigmodel_sig_2.5', 'apr4_sr_conv_bigmodel_sig_5',
                         'apr4_sr_conv_bigmodel_sig_10', 'apr4_sr_conv_bigmodel_sig_25', 'apr4_sr_conv_bigmodel_sig_50',
                         'apr4_sr_conv_sig_0', 'apr4_sr_conv_sig_2.5', 'apr4_sr_conv_sig_5',
                         'apr4_sr_conv_sig_10', 'apr4_sr_conv_sig_25', 'apr4_sr_conv_sig_50'
                         ]
  sr_conv_apr4_ratios = [1.0, 0.1, 0.01, 0.001]
  sr_conv_apr4_mat = np.load('results/results_full_apr4_sr_conv.npy')
  sr_conv_apr4_idx_names = {'setups': sr_conv_apr4_setups, 'sub_ratios': sr_conv_apr4_ratios}
  sr_conv_apr4_idx_names.update(base_idx_names)
  sr_conv_apr4_idx_names['data_ids'] = sr_conv_apr4_data_ids
  sr_conv_apr4_array = NamedArray(sr_conv_apr4_mat, dim_names, sr_conv_apr4_idx_names)

  sr_conv_apr6_data_ids = ['d', 'f']
  sr_conv_apr6_setups = ['(apr6)',
                         'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5', 'apr6_sr_conv_sig_25']
  sr_conv_apr6_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  sr_conv_apr6_mat = np.load('results/results_full_apr6_sr_conv_partial.npy')
  sr_conv_apr6_idx_names = {'setups': sr_conv_apr6_setups, 'sub_ratios': sr_conv_apr6_ratios}
  sr_conv_apr6_idx_names.update(base_idx_names)
  sr_conv_apr6_idx_names['data_ids'] = sr_conv_apr6_data_ids
  sr_conv_apr6_array = NamedArray(sr_conv_apr6_mat, dim_names, sr_conv_apr6_idx_names)

  sb_np_array = sb_array.merge(np_array, merge_dim='setups')
  all_array = sb_np_array.merge(sr_array, merge_dim='setups')

  array_dict = {'sb': sb_array,
                'np': np_array,
                'sr': sr_array,
                'sb_np': sb_np_array,
                'all': all_array,
                'sr_d': sr_d_array,
                'sr_f': sr_f_array,
                'sr_conv_apr4': sr_conv_apr4_array,
                'sr_conv_apr6': sr_conv_apr6_array}

  return array_dict


def aggregate_sep18_realmmd():
  data_suffix = {'digits': '', 'fashion': '_fashion'}

  # setups = ['sep18_real_mmd_baseline_s']
  # sub_ratios = ['1.0']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [1, 2, 3, 4, 5]
  eval_metrics = ['accuracies', 'f1_scores']
  # save_str = 'sep18_real_mmd_baseline'

  for data in data_suffix:
    for m in models:
      scores = {'accuracies': [], 'f1_scores': []}
      for run in runs:
        load_file = f'logs/gen/sep18_real_mmd_baseline_s{run}{data_suffix[data]}/synth_eval/sub1.0_{m}_log.npz'
        if os.path.isfile(load_file):
          mat = np.load(load_file)
        else:
          print('failed to load', load_file)
          continue
        for e_idx, e in enumerate(eval_metrics):
          score = mat[e][1]
          scores[e].append(score)
      accs = np.asarray(scores["accuracies"])
      print(f'model: {m}')
      print(f'acc avg: {np.mean(accs)}')
      print(f'accs: {accs}')


def aggregate_oct13_mnist_redo(verbose):
  data_suffix = {'digits': 'd', 'fashion': 'f'}

  epsilons = [0, 5, 25]
  # sub_ratios = ['1.0']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies']
  # save_str = 'sep18_real_mmd_baseline'

  for data in data_suffix:
    print(data)
    for eps in epsilons:
      print(f'eps={eps}')
      for m in models:
        scores = {'accuracies': []}
        for run in runs:
          load_file = f'logs/gen/oct12_eps_{data_suffix[data]}{eps}_s{run}/synth_eval/sub1.0_{m}_log.npz'
          if os.path.isfile(load_file):
            mat = np.load(load_file)
          else:
            print('failed to load', load_file)
            continue
          for e_idx, e in enumerate(eval_metrics):
            score = mat[e][1]
            scores[e].append(score)
        accs = np.asarray(scores["accuracies"])
        if verbose:
          print(f'model: {m}')
          print(f'acc avg: {np.mean(accs)}')
          print(f'accs: {accs}')
        else:
          print(np.mean(accs))



if __name__ == '__main__':
  # aggregate_sep18_realmmd()
  aggregate_oct13_mnist_redo(True)
  aggregate_oct13_mnist_redo(False)
