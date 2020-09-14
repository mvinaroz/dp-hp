import os
import shutil
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from aux import plot_mnist_batch, NamedArray
from sklearn.metrics import roc_curve, auc


def expand_vector(v, tgt_vec):
  # expand v to the number of dimensions of tgt_vec. I'm sure there is a nice way to do this but this works as well
  tgt_dims = len(tgt_vec.shape)
  if tgt_dims == 2:
    return v[:, None]
  elif tgt_dims == 3:
    return v[:, None, None]
  elif tgt_dims == 4:
    return v[:, None, None, None]
  elif tgt_dims == 5:
    return v[:, None, None, None, None]
  elif tgt_dims == 6:
    return v[:, None, None, None, None, None]
  else:
    return ValueError


def named_array_test():
  a = np.asarray(list(range(125)))
  a = np.reshape(a, (5, 5, 5))
  name_ids = [str(k) for k in range(5)]
  dim_names = ['a', 'b', 'c']
  idx_names = {'a': name_ids, 'b': name_ids, 'c': name_ids}
  named_arr1 = NamedArray(a, dim_names, idx_names)

  q1 = {'a': ['4', '2'], 'b': ['0', '1'], 'c': ['3', '4']}
  get1 = named_arr1.get(q1)

  q2 = {'b': ['1', '3', '0'], 'c': ['1', '4']}
  get2 = named_arr1.get(q2)

  print(a)
  print(get1)
  print(get2)

  a = np.asarray(list(range(1000, 1060)))
  a = np.reshape(a, (5, 3, 4))
  dim_names = ['a', 'b', 'c']
  idx_names = {'a': ['1', '2', '3', '7', '9'], 'b': ['10', '11', '12'], 'c': ['0', '1', '2', '3']}
  named_arr2 = NamedArray(a, dim_names, idx_names)

  merged_array = named_arr1.merge(named_arr2, merge_dim='b')
  print(merged_array.idx_names)
  print(merged_array.array)


def collect_arp4_grid():
  log_dir = 'logs/gen/apr4_grid_log/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  for idx in range(216):
    run_dir = f'logs/gen/apr3_sr_conv_grid_{idx}/'

    final_ep = [50, 20, 5]
    for ep in final_ep:
      run_plot_path = run_dir + f'samples_ep{ep}.png'
      tgt_plot_path = log_dir + f'run{idx}_ep{ep}.png'
      if os.path.exists(run_plot_path):
        shutil.copy(run_plot_path, tgt_plot_path)
        break


def collect_apr4_sr_conv():
  log_dir = 'logs/gen/apr5_conv_sig_overview/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  for mode in ['', '_bigmodel']:
    for data in {'d', 'f'}:
      for sig in [0, 2.5, 5, 10, 25, 50]:
        for run in range(5):
          run_dir = f'logs/gen/apr4_sr_conv{mode}_sig_{sig}_{data}{run}/'
          run_plot_path = run_dir + f'samples_ep20.png'
          tgt_plot_path = log_dir + f'data_{data}_{mode}_sig{sig}_run{run}_ep20.png'
          if os.path.exists(run_plot_path):
            shutil.copy(run_plot_path, tgt_plot_path)


def collect_apr6_sr_conv():
  log_dir = 'logs/gen/apr5_conv_sig_overview/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  for mode in ['', '_bigmodel']:
    for data in {'d', 'f'}:
      for sig in [0, 2.5, 5, 10, 25, 50]:
        for run in range(5):
          run_dir = f'logs/gen/apr4_sr_conv{mode}_sig_{sig}_{data}{run}/'
          run_plot_path = run_dir + f'samples_ep20.png'
          tgt_plot_path = log_dir + f'data_{data}_{mode}_sig{sig}_run{run}_ep20.png'
          if os.path.exists(run_plot_path):
            shutil.copy(run_plot_path, tgt_plot_path)


def collect_apr6_noconv_grid():
  log_dir = 'logs/gen/apr6_noconv_grid_overview/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  for idx in range(216):
    run_dir = f'logs/gen/apr6_sr_noconv_grid_{idx}/'

    final_ep = [50, 20, 5]
    for ep in final_ep:
      run_plot_path = run_dir + f'samples_ep{ep}.png'
      tgt_plot_path = log_dir + f'run{idx}_ep{ep}.png'
      if os.path.exists(run_plot_path):
        shutil.copy(run_plot_path, tgt_plot_path)
        break


def collect_apr6_better_conv():
  log_dir = 'logs/gen/apr6_conv_overview/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  max_ep = '5'
  for data in {'d', 'f'}:
    for sig in [0, 5, 25]:
      for run in range(5):
        run_dir = f'logs/gen/apr6_sr_conv_sig{sig}_{data}{run}/'
        run_plot_path = run_dir + f'samples_ep{max_ep}.png'
        tgt_plot_path = log_dir + f'data_{data}_sig{sig}_run{run}_ep{max_ep}.png'
        if os.path.exists(run_plot_path):
          shutil.copy(run_plot_path, tgt_plot_path)


def plot_dpcgan_data():
  # mat = np.load('dp-cgan-synth-mnist-eps1.0.npz')
  # mat = np.load('dmnist-sig5-eps1.0.npz')
  mat = np.load('old_plots_and_synth_datasets/dp-cgan-synth-mnist-eps=9.6.npz')
  data = mat['data']
  print(data.shape)
  labels = mat['labels']
  mat_select = []
  for idx in range(10):
    ids = np.where(labels[:, idx] == 1)[0][:10]
    print(ids)
    mat_select.append(data[ids])
  mat_select = np.concatenate(mat_select, axis=0)
  print(mat_select.shape)
  plot_mnist_batch(mat_select, 10, 10, 'dmnist-eps9.6-plot', save_raw=False)


def dpcgan_dummmy_eval():
  targets = np.zeros(60000)
  targets[:6000] = 1.
  print(sum(targets))
  preds = np.zeros(60000)
  # preds[:12000] = .5
  # preds[6000:12000] = 1.

  # preds[6600:12000] = 1.
  # preds[:600] = 1.

  preds[6000:12000] = .51
  preds[:6000] = .49

  # preds[:12000] = .5

  print(sum(preds))
  fpr, tpr, thresholds = roc_curve(targets, preds)

  print(fpr, tpr, thresholds, auc(fpr, tpr))


def collect_sep14_real_mmd_grid():
  log_dir = 'logs/gen/sep14_realmmd_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  scores = []
  for run in range(64):
    run_score = f'logs/gen/sep14_realmmd_{run}/final_score'
    if os.path.exists(run_score):
      with open(run_score) as f:
        scores.append((run, f.readline()))

  with open(log_dir + 'scores') as f:
    for idx, score in scores:
      print(f'{idx}: {score}')
      f.write(f'{idx}: {score}')

# import numpy as np
# import os
# run_types = [20, 30, 40, 50, 60, 70]
# run_seeds = [1, 2, 3, 4, 5]
# for run in run_types:
#   run_results = []
#   for seed in run_seeds:
#     path = f'logs/gen/aug3_sig{run}_s{seed}/final_score'
#     if os.path.exists(path):
#       with open(path) as f:
#         line = f.readlines()[0]
#         assert line[:5] == 'acc: '
#         run_results.append(float(line[5:]))
#     else:
#       print(f'{path} not found')
#   print(f'{run}: {np.mean(np.asarray(run_results))}')
#   print(run_results)


if __name__ == '__main__':
  collect_sep14_real_mmd_grid()
  # dpcgan_dummmy_eval()
  # 'dpmerf-high-eps-f0'
  # mat = np.load('logs/gen/dpmerf-high-eps-d4/synth_eval/sub0.1_bagging_log.npz')
  # mat = np.load('logs/gen/dpmerf-high-eps-f4/synth_eval/sub0.1_bagging_log.npz')
  # plot_dpcgan_data()
  # print(mat['accuracies'])
  # collect_arp4_grid()
  # collect_apr4_sr_conv()
  # collect_apr6_better_conv()
  # collect_apr6_noconv_grid()
