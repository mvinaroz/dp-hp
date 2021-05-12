import os
import shutil
# import matplotlib
# matplotlib.use('Agg')  # to plot without Xserver
# import matplotlib.pyplot as plt
import numpy as np
# from torchvision import datasets
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
  # mat = np.load('dp-cgan-synth-code_balanced-eps1.0.npz')
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
  log_dir = 'logs/gen/sep14_realmmd2_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  scores = []
  run_range = ['00', '01', '02', '03', '04', '05', '06', '07', '08',  '09'] + [str(k) for k in range(10, 28)]
  for run in run_range:
    run_score = f'logs/gen/sep14_realmmd2_{run}/final_score'
    if os.path.exists(run_score):
      with open(run_score) as f:
        scores.append((run, f.readline()))

  for idx, score in scores:
    print(f'{idx}: {score}')

  with open(log_dir + 'scores', mode='w') as f:
    for idx, score in scores:
      f.write(f'{idx}: {score}')


def collect_sep21_nonp_kmeans_grid():
  log_dir = 'logs/gen/sep21_kmeans_grid_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  scores = []
  run_range = [str(k) for k in range(28)]
  for run in run_range:
    run_score = f'logs/gen/sep21_kmeans_grid_{run}/final_score'
    if os.path.exists(run_score):
      with open(run_score) as f:
        scores.append((run, f.readline()))

  for idx, score in scores:
    print(f'{idx}: {score}')

  with open(log_dir + 'scores', mode='w') as f:
    for idx, score in scores:
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


def collect_oct4_dpcgan_grid():
  log_dir = '../../dpcgan/logs/oct4_synd_2d_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  for run in range(90):
    run_dir = f'../../dpcgan/logs/dp-cgan-synth-2d-disc_k5_n100000_row5_col5_noise0.2-eps9.6/syn2d_grid_{run}/'
    run_plot_path = run_dir + f'gen_data.png.png'
    tgt_plot_path = log_dir + f'gen_data_{run}.png'
    if os.path.exists(run_plot_path):
      shutil.copy(run_plot_path, tgt_plot_path)


def collect_oct4_dpcgan_grid_scores():
  for run in range(90):
    run_file = f'../../dpcgan/joblogs/oct4_dpcgan_grid_{run}.out.txt'


    if os.path.exists(run_file):

      with open(run_file) as f:
        lines = f.readlines()
        if len(lines) > 0 and lines[-1].startswith('gen samples eval score: '):
          score = lines[-1].split()[-1]
          print(f'{run}: {score}')
        else:
          print(f'{run} wrong format')
    else:

      print(f'{run} not found')


def gather_syn_plots(log_dir, run_dir_fun, run_plot_name, log_plot_name_fun, n_runs):
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  for run in range(n_runs):

    run_plot_path = run_dir_fun(run) + run_plot_name
    tgt_plot_path = log_dir + log_plot_name_fun(run)
    if os.path.exists(run_plot_path):
      shutil.copy(run_plot_path, tgt_plot_path)
    else:
      print(f'{run} not found')


def collect_oct5_dpcgan_grid():
  gather_syn_plots(log_dir='../../dpcgan/logs/oct5_synd_2d_summary/',
                   run_dir_fun=lambda x: f'../../dpcgan/logs/dp-cgan-synth-2d-disc_k5_n100000_row5_col5_noise0.2-eps9.6/syn2d_grid_oct5_{x}/',
                   run_plot_name='gen_data.png',
                   log_plot_name_fun=lambda x: f'gen_data_{x}.png',
                   n_runs=108)


  for run in range(108):
    run_file = f'../../dpcgan/joblogs/oct5_dpcgan_grid_{run}.out.txt'

    if os.path.exists(run_file):

      with open(run_file) as f:
        lines = f.readlines()
        if len(lines) > 0 and lines[-1].startswith('gen samples eval score: '):
          score = lines[-1].split()[-1]
          print(f'{run}: {score}')
        else:
          print(f'{run} wrong format')
    else:

      print(f'{run} not found')


def collect_oct7_dpgan_grid_scores_and_plots():
  log_dir = '../../dpgan-alternative/synth_data/oct7_synd_2d_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

    for run in range(162):
      run_dir = f'../../dpgan-alternative/synth_data/oct7_grid_{run}/'
      run_plot_path = run_dir + f'data_plot.png'
      tgt_plot_path = log_dir + f'gen_data_{run}.png'
      if os.path.exists(run_plot_path):
        shutil.copy(run_plot_path, tgt_plot_path)

  for run in range(81):
    run_file = f'../../dpgan-alternative/joblogs/oct7_dpgan_syn2d_grid_{run}.out.txt'

    if os.path.exists(run_file):
      with open(run_file) as f:
        lines = f.readlines()
        score_lines = [l.split()[-1] for l in lines if l.startswith('likelhood: ')]

        if len(score_lines) == 2:
          print(f'{2*run}: {score_lines[0]}')
          print(f'{2*run+1}: {score_lines[1]}')
        else:
          print(f'{2*run}, {2*run+1} wrong format')
    else:
      print(f'{run} not found')


def collect_oct8_dpmerf_grid_scores_and_plots():
  log_dir = 'logs/gen/oct8_dpmerf_syn2d_grid_summary/'
  gather_syn_plots(log_dir,
                   run_dir_fun=lambda x: f'logs/gen/oct8_dpmerf_syn2d_grid{x}/',
                   run_plot_name='plot_gen.png',
                   log_plot_name_fun=lambda x: f'plot_gen_{x}.png',
                   n_runs=298)

  for run in range(298):
    run_file = f'logs/gen/oct8_dpmerf_syn2d_grid{run}/final_score'

    if os.path.exists(run_file):
      with open(run_file) as f:
        lines = f.readlines()

        if len(lines) == 1:
          print(f'{run}: {lines[0].rstrip()}')
        else:
          print(f'{run} wrong format')
    else:
      print(f'{run} not found')


def collect_oct8_dpcgan_grid():
  gather_syn_plots(log_dir='../../dpcgan/logs/oct8_synd_2d_summary/',
                   run_dir_fun=lambda x: f'../../dpcgan/logs/dp-cgan-synth-2d-norm_k5_n100000_row5_col5_noise0.2-eps1.0/syn2d_grid_oct8_{x}/',
                   run_plot_name='gen_data.png',
                   log_plot_name_fun=lambda x: f'gen_data_{x}.png',
                   n_runs=24)

  for run in range(24):
    run_file = f'../../dpcgan/joblogs/oct8_dpcgan_grid_{run}.out.txt'

    if os.path.exists(run_file):

      with open(run_file) as f:
        lines = f.readlines()
        if len(lines) > 0 and lines[-1].startswith('gen samples eval score: '):
          score = lines[-1].split()[-1]
          print(f'{run}: {score}')
        else:
          print(f'{run} wrong format')
    else:

      print(f'{run} not found')


def collect_oct8_dpmerf_log_likelihoods():

  for run in range(64):
    run_file = f'joblogs/oct8_dpmerf_syn2d_grid_{run}.out.txt'

    if os.path.exists(run_file):

      with open(run_file) as f:
        print('job', run)
        lines = [line for line in f.readlines() if line.startswith('Score of evaluation function: ')]

        if len(lines) > 0:
          for line in lines:
            print(line.split()[-1])
        else:
          print(f'{run} wrong format')
    else:

      print(f'{run} not found')


def collect_oct9_dpcgan_grid():
  gather_syn_plots(log_dir='../../dpcgan/logs/oct9_synd_2d_summary/',
                   run_dir_fun=lambda x: f'../../dpcgan/logs/dp-cgan-synth-2d-norm_k5_n100000_row5_col5_noise0.2-eps9.6/syn2d_grid_oct9_{x}/',
                   run_plot_name='gen_data.png',
                   log_plot_name_fun=lambda x: f'gen_data_{x}.png',
                   n_runs=8)

  for run in range(8):
    run_file = f'../../dpcgan/joblogs/oct9_dpcgan_grid_{run}.out.txt'

    if os.path.exists(run_file):

      with open(run_file) as f:
        lines = f.readlines()
        if len(lines) > 0 and lines[-1].startswith('gen samples eval score: '):
          score = lines[-1].split()[-1]
          print(f'{run}: {score}')
        else:
          print(f'{run} wrong format')
    else:

      print(f'{run} not found')


def collect_oct10_dpcgan_grid():
  gather_syn_plots(log_dir='../../dpcgan/logs/oct10_synd_2d_summary/',
                   run_dir_fun=lambda x: f'../../dpcgan/logs/dp-cgan-synth-2d-norm_k5_n100000_row5_col5_noise0.2-eps0.0/syn2d_grid_oct10_{x}/',
                   run_plot_name='gen_data.png',
                   log_plot_name_fun=lambda x: f'gen_data_{x}.png',
                   n_runs=48)

  for run in range(48):
    run_file = f'../../dpcgan/joblogs/oct10_dpcgan_nondp_grid_{run}.out.txt'

    if os.path.exists(run_file):

      with open(run_file) as f:
        lines = f.readlines()
        if len(lines) > 0 and lines[-1].startswith('gen samples eval score: '):
          score = lines[-1].split()[-1]
          print(f'{run}: {score}')
        else:
          print(f'{run} wrong format')
    else:

      print(f'{run} not found')


def collect_oct9_dpmerf_mnist_scores():
  log_dir = 'logs/gen/oct9_mnist_grid_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  scores = []
  run_range = [str(k) for k in range(216)]
  for run in run_range:
    run_score = f'logs/gen/oct9_mnist_grid_{run}/final_score'
    if os.path.exists(run_score):
      with open(run_score) as f:
        scores.append((run, f.readline().rstrip()))

  for idx, score in scores:
    print(f'{idx}: {score}')

  with open(log_dir + 'scores', mode='w') as f:
    for idx, score in scores:
      f.write(f'{idx}: {score}')


def collect_oct12_dpmerf_mnist_scores():
  log_dir = 'logs/gen/oct12_mnist_grid_summary/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  scores = []
  # run_range = [str(k) for k in range(216)]
  # for run in run_range:
  for data in ['d', 'f']:
    for eps in [0, 5, 25]:
      for seed in range(5):
        run_score = f'logs/gen/oct12_eps_{data}{eps}_s{seed}/final_score'
        if os.path.exists(run_score):
          with open(run_score) as f:
            score_tup = (f'{data}_eps{eps}_s{seed}', f.readline().rstrip())
            scores.append(score_tup)
          print(f'{score_tup[0]}: {score_tup[1]}')
        else:
          print(f'{data}_eps{eps}_s{seed}: not found')

  with open(log_dir + 'scores', mode='w') as f:
    for idx, score in scores:
      f.write(f'{idx}: {score}')


def collect_may11_dpmehp_mnist_scores():
  log_dir = '../dp_mehp/logs/gen/may11_digits_hp_scores/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  # run_range = [str(k) for k in range(216)]
  # for run in run_range:
  eval_files = ['sub0.1_bernoulli_nb_log.npz', 'sub0.1_gaussian_nb_log.npz',
                'sub0.1_logistic_reg_log.npz', 'sub0.1_random_forest_log.npz']

  for run_setting, n_runs in [('', 84), ('_continuous', 112)]:
    scores = np.zeros((n_runs, 4))
    for run in range(n_runs):

      eval_dir = f'../dp_mehp/logs/gen/may11_digits_hp{run_setting}_grid_{run}/digits/synth_eval/'

      for f_idx, f in enumerate(eval_files):
        eval_file = os.path.join(eval_dir, f)
        if os.path.exists(eval_file):
          scores[run, f_idx] = np.load(eval_file)['accuracies'][1]
        else:
          print(f'hp{run_setting} run {run} file {f}: not found')

    np.save(os.path.join(log_dir, f'scores{run_setting}.npy'), scores)
    max_vals = np.max(scores, axis=0)
    max_ids = np.argmax(scores, axis=0)
    max_avg_val = np.max(np.mean(scores, axis=1))
    max_avg_id = np.argmax(np.mean(scores, axis=1))
    print(f'results for hp{run_setting}:')
    print(f'best scores: {max_vals} at runs {max_ids}')
    print(f'best average: {max_avg_val} at run {max_avg_id}')


if __name__ == '__main__':
  # collect_sep21_nonp_kmeans_grid()
  # collect_oct4_dpcgan_grid()
  # collect_oct4_dpcgan_grid_scores()
  # collect_oct7_dpgan_grid_scores_and_plots()
  # collect_oct5_dpcgan_grid()
  # collect_oct8_dpgan_grid_scores_and_plots()
  # collect_oct8_dpcgan_grid()
  # collect_oct8_dpmerf_log_likelihoods()
  # collect_oct10_dpcgan_grid()
  # collect_oct9_dpmerf_mnist_scores()
  # collect_oct12_dpmerf_mnist_scores()
  collect_may11_dpmehp_mnist_scores()
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
