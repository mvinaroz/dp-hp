import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from all_aux_files import NamedArray


DEFAULT_MODELS = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
                  'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
DEFAULT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:brown', 'tab:orange', 'tab:gray',
                  'tab:pink', 'limegreen', 'yellow']
DEFAULT_RATIOS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
DEFAULT_RATIO_KEYS = ['60k', '30k', '12k', '6k', '3k', '1.2k', '600', '300', '120', '60']


def collect_results():
  # read out subsampled, mar19_nonp and mar20_sr and combine them
  data_ids = ['d', 'f']
  eval_metrics = ['accuracies', 'f1_scores']

  runs = [0, 1, 2, 3, 4]
  dim_names = ['data_ids', 'setups', 'sub_ratios', 'models', 'runs', 'eval_metrics']  # order matters!
  sub_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  base_idx_names = {'data_ids': data_ids, 'models': DEFAULT_MODELS, 'runs': runs, 'eval_metrics': eval_metrics,
                    'sub_ratios': sub_ratios}

  sb_setups = ['real_data', 'dpcgan', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps', 'dpmerf-nonprivate_sb']
  sb_mat = np.load('mnist_results/results_full_mar25_subsampled.npy')
  sb_idx_names = {'setups': sb_setups}
  sb_idx_names.update(base_idx_names)
  sb_array = NamedArray(sb_mat, dim_names, sb_idx_names)

  sr_conv_apr6_setups = ['(apr6)', 'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5', 'apr6_sr_conv_sig_25']
  sr_conv_apr6_mat = np.load('mnist_results/results_full_apr6_sr_conv_partial.npy')
  sr_conv_apr6_idx_names = {'setups': sr_conv_apr6_setups}
  sr_conv_apr6_idx_names.update(base_idx_names)
  sr_conv_apr6_array = NamedArray(sr_conv_apr6_mat, dim_names, sr_conv_apr6_idx_names)

  full_mmd_jan7_idx_names = {'setups': ['full_mmd']}
  full_mmd_jan7_idx_names.update(base_idx_names)
  full_mmd_jan7_mat = np.load('mnist_results/results_full_sep18_real_mmd.npy')
  # full_mmd_jan7_mat = np.expand_dims(full_mmd_jan7_mat, 1)
  full_mmd_jan7_array = NamedArray(full_mmd_jan7_mat, dim_names, full_mmd_jan7_idx_names)

  mehp_fmnist_apr23_idx_names = {'setups': ['mehp_nonDP', 'mehp_eps=1']}
  mehp_fmnist_apr23_idx_names.update(base_idx_names)
  mehp_fmnist_apr23_idx_names['data_ids'] = ['f']
  mehp_fmnist_apr23_idx_names['eval_metrics'] = ['accuracies']
  mehp_fmnist_apr23_mat = np.load('mnist_results/results_full_apr23_fmnist_mehp.npy')
  mehp_fmnist_apr23_array = NamedArray(mehp_fmnist_apr23_mat, dim_names, mehp_fmnist_apr23_idx_names)


  merf_adaboost_may10_setups = ['adaboost merf non-DP', 'adaboost merf_eps=1', 'adaboost merf_eps=0.2']
  merf_adaboost_may10_idx_names = {'setups': merf_adaboost_may10_setups}
  merf_adaboost_may10_idx_names.update(base_idx_names)
  merf_adaboost_may10_idx_names['eval_metrics'] = ['accuracies']
  merf_adaboost_may10_idx_names['models'] = ['adaboost']
  merf_adaboost_may10_mat = np.load('mnist_results/results_full_may10_merf_adaboost.npy')
  merf_adaboost_may10_array = NamedArray(merf_adaboost_may10_mat, dim_names, merf_adaboost_may10_idx_names)

  real_adaboost_may10_idx_names = {'setups': ['real data adaboost']}
  real_adaboost_may10_idx_names.update(base_idx_names)
  real_adaboost_may10_idx_names['eval_metrics'] = ['accuracies']
  real_adaboost_may10_idx_names['models'] = ['adaboost']
  real_adaboost_may10_mat = np.load('mnist_results/results_full_may10_real_adaboost.npy')
  real_adaboost_may10_array = NamedArray(real_adaboost_may10_mat, dim_names, real_adaboost_may10_idx_names)

  gswgan_may14_idx_names = {'setups': ['gswgan']}
  gswgan_may14_idx_names.update(base_idx_names)
  gswgan_may14_idx_names['eval_metrics'] = ['accuracies']
  gswgan_may14_mat = np.load('mnist_results/results_full_may14_gswgan.npy')
  gswgan_may14_array = NamedArray(gswgan_may14_mat, dim_names, gswgan_may14_idx_names)

  # may20_mehp_dmnist_fc
  mehp_dmnist_may20_fc_setups = ['fc mehp non-DP order20', 'fc mehp non-DP order50', 'fc mehp non-DP order100',
                                 'fc mehp non-DP order200', 'fc mehp non-DP order500', 'fc mehp non-DP order1000',
                                 'fc mehp eps=1 order20', 'fc mehp eps=1 order50', 'fc mehp eps=1 order100',
                                 'fc mehp eps=1 order200', 'fc mehp eps=1 order500', 'fc mehp eps=1 order1000']
  mehp_dmnist_may20_fc_idx_names = {'setups': mehp_dmnist_may20_fc_setups}
  mehp_dmnist_may20_fc_idx_names.update(base_idx_names)
  mehp_dmnist_may20_fc_idx_names['data_ids'] = ['d']
  mehp_dmnist_may20_fc_idx_names['eval_metrics'] = ['accuracies']
  mehp_dmnist_may20_fc_mat = np.load('mnist_results/results_full_may20_mehp_dmnist_fc.npy')
  mehp_dmnist_may20_fc_array = NamedArray(mehp_dmnist_may20_fc_mat, dim_names, mehp_dmnist_may20_fc_idx_names)

  dpgan_may27_idx_names = {'setups': ['may27_dpgan']}
  dpgan_may27_idx_names.update(base_idx_names)
  dpgan_may27_idx_names['eval_metrics'] = ['accuracies']
  dpgan_may27_mat = np.load('mnist_results/results_full_may27_dpgan.npy')
  dpgan_may27_array = NamedArray(dpgan_may27_mat, dim_names, dpgan_may27_idx_names)

  array_dict = {'sb': sb_array,
                'sr_conv_apr6': sr_conv_apr6_array,
                'full_mmd_jan7': full_mmd_jan7_array,
                'mehp_fmnist_apr23': mehp_fmnist_apr23_array,
                'may10_merf_adaboost': merf_adaboost_may10_array,
                'may10_real_adaboost': real_adaboost_may10_array,
                'may14_gswgan': gswgan_may14_array,
                'mehp_dmnist_may20_fc': mehp_dmnist_may20_fc_array,
                'may27_dpgan': dpgan_may27_array}

  return array_dict


def downstream_model_comparison():
  data_ids = ['d', 'f']

  for data_id in data_ids:
    if data_id == 'd':
      save_str = 'digit'
      data_str = 'mehp_dmnist_may20_fc'
      setups = ['fc mehp non-DP order100', 'fc mehp eps=1 order100']
      y_lims = (0.2, 1.0)
      y_step = 0.1

    else:
      save_str = 'fashion'
      data_str = 'mehp_fmnist_apr23'
      setups = ['mehp_nonDP', 'mehp_eps=1']
      y_lims = (0.2, 0.9)
      y_step = 0.1

    ar_dict = collect_results()
    sr_conv_array = ar_dict['sr_conv_apr6']
    merf_adaboost_array = ar_dict['may10_merf_adaboost']
    sr_conv_array.array[:, 1:, :, 7, :, 0] = merf_adaboost_array.array[:, :, :, 0, :, 0]

    sb_array = ar_dict['sb']
    real_adaboost_array = ar_dict['may10_real_adaboost']
    sb_array.array[:, 0, :, 7, :, 0] = real_adaboost_array.array[:, 0, :, 0, :, 0]

    merged_array = sr_conv_array.merge([sb_array, ar_dict['full_mmd_jan7'],
                                        ar_dict[data_str],
                                        ar_dict['may14_gswgan']],
                                       merge_dim='setups')

    queried_setups = ['real_data', 'full_mmd',
                      setups[0], setups[1],
                      'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5',
                      'gswgan', 'dpcgan']
    setup_names = ['real data', 'full MMD $\epsilon=\infty$',
                   'DP-HP (ours) $\epsilon=\infty$', 'DP-HP (ours) $\epsilon=1$',
                   'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                   'GS-WGAN $\epsilon=10$', 'DP-CGAN $\epsilon=9.6$']
    plt.figure()
    ax = plt.subplot(111)
    plt.xticks(list(range(12)) + [13], DEFAULT_MODELS + ['MEAN'])
    plt.xticks(rotation=45, ha='right')
    print('data', data_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [data_id], 'setups': [s],
                                  'models': DEFAULT_MODELS, 'eval_metrics': ['accuracies']})
      print(f'setup: {s}')
      print(sub_mat.shape)

      sub_mat = sub_mat[0, :, :]  # select full data exp
      print(sub_mat.shape)

      means_y = np.mean(sub_mat, axis=1)
      means_y_plus_mean = np.concatenate([means_y, np.mean(means_y, keepdims=True)])
      plt.plot(list(range(12)) + [13], means_y_plus_mean, 'o', label=setup_names[s_idx], color=DEFAULT_COLORS[s_idx])

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=0, xmax=13, linestyles='dotted')
    plt.vlines(12, ymin=0.2, ymax=1.)
    plt.ylim(y_lims)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.75, box.height * 0.95])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{save_str.capitalize()} MNIST downstream accuracy by model', loc='center', pad=10.)
    plt.savefig(f'plot_downstream_model_comparison_{save_str}.png')


def plot_with_variance(x, y, color, label, alpha=0.1):
  """
  assume y is of shape (x_settings, runs to average)
  """
  means_y = np.mean(y, axis=1)
  sdevs_y = np.std(y, axis=1)
  plt.plot(x, means_y, 'o-', label=label, color=color)
  plt.fill_between(x, means_y-sdevs_y, means_y+sdevs_y, alpha=alpha, color=color)


def subsampled_accuracies():
  metric = 'accuracies'
  data_ids = ['d', 'f']

  for data_id in data_ids:
    if data_id == 'd':
      save_str = 'digit'
      data_str = 'mehp_dmnist_may20_fc'
      setups = ['fc mehp non-DP order100', 'fc mehp eps=1 order100']
      y_lims = (0.4, 0.9)
      y_step = 0.05

    else:
      save_str = 'fashion'
      data_str = 'mehp_fmnist_apr23'
      setups = ['mehp_nonDP', 'mehp_eps=1']
      y_lims = (0.35, 0.8)
      y_step = 0.05

    ar_dict = collect_results()
    sr_conv_array = ar_dict['sr_conv_apr6']
    merf_adaboost_array = ar_dict['may10_merf_adaboost']
    sr_conv_array.array[:, 1:, :, 7, :, 0] = merf_adaboost_array.array[:, :, :, 0, :, 0]

    sb_array = ar_dict['sb']
    real_adaboost_array = ar_dict['may10_real_adaboost']
    sb_array.array[:, 0, :, 7, :, 0] = real_adaboost_array.array[:, 0, :, 0, :, 0]

    merged_array = sr_conv_array.merge([sb_array, ar_dict['full_mmd_jan7'],
                                        ar_dict[data_str],
                                        ar_dict['may14_gswgan'],
                                        ar_dict['may27_dpgan']],
                                       merge_dim='setups')

    queried_setups = ['real_data', 'full_mmd',
                      setups[0], setups[1],
                      'apr6_sr_conv_sig_0', 'apr6_sr_conv_sig_5',
                      'gswgan', 'dpcgan', 'may27_dpgan']
    setup_names = ['real data', 'full MMD $\epsilon=\infty$',
                   'DP-HP (ours) $\epsilon=\infty$', 'DP-HP (ours) $\epsilon=1$',
                   'DP-MERF $\epsilon=\infty$', 'DP-MERF $\epsilon=1$',
                   'GS-WGAN $\epsilon=10$', 'DP-CGAN $\epsilon=9.6$', 'DP-GAN $\epsilon=9.6$']
    plt.figure()
    ax = plt.subplot(111)
    plt.xscale('log')
    plt.xticks(DEFAULT_RATIOS[::-1], DEFAULT_RATIO_KEYS[::-1])
    print('data', data_id)
    for s_idx, s in enumerate(queried_setups):
      sub_mat = merged_array.get({'data_ids': [data_id], 'setups': [s], 'models': DEFAULT_MODELS, 'eval_metrics': [metric]})

      print(f'setup: {s}')
      sub_mat = np.mean(sub_mat, axis=1)  # average over models
      mean_y_100 = np.mean(sub_mat, axis=1)[0]
      mean_y_opt = np.max(np.mean(sub_mat, axis=1))
      sdev_y_100 = np.std(sub_mat, axis=1)[0]
      print(f'acc mean: {mean_y_100}, sdev: {sdev_y_100}, opt: {mean_y_opt}')

      plot_with_variance(DEFAULT_RATIOS, sub_mat, color=DEFAULT_COLORS[s_idx], label=setup_names[s_idx])

    plt.xlabel('# samples generated')
    plt.ylabel('accuracy')

    y_ticks = np.arange(y_lims[0], y_lims[1] + y_step, y_step)
    h_lines = np.arange(y_lims[0] + y_step, y_lims[1], y_step)
    plt.yticks(y_ticks)
    plt.hlines(h_lines, xmin=DEFAULT_RATIOS[-1], xmax=DEFAULT_RATIOS[0], linestyles='dotted')
    plt.ylim(y_lims)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{save_str.capitalize()} MNIST downstream accuracy under subsampling', loc='center', pad=10.)
    plt.savefig(f'plot_order100_{save_str}_subsampling.png')

if __name__ == '__main__':
    downstream_model_comparison()
    subsampled_accuracies()
