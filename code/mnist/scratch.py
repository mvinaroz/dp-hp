import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from aux import plot_mnist_batch

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


def aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str):
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
            mat = np.load(load_file)
            for e_idx, e in enumerate(eval_metrics):
              score = mat[e][1]
              all_the_results[d_idx, s_idx + 1, r_idx, m_idx, run_idx, e_idx] = score

              if s == setup_with_real_data:
                all_the_results[d_idx, 0, r_idx, m_idx, run_idx, e_idx] = mat[e][0]

  np.save(f'{save_str}_full_results', all_the_results)

  for e_idx, e in enumerate(eval_metrics):
    print('metric:', e)
    for d_idx, d in enumerate(data_ids):
      print('data:', d)
      for s_idx, s in enumerate(['real_data'] + setups):
        print('setup:', s)
        for r_idx, r in enumerate(sub_ratios):
          print('sub_ratio:', r, 'mean:', np.mean(all_the_results[d_idx, s_idx, r_idx, :, :, e_idx]))
          # print(all_the_results[d_idx, s_idx, r_idx, :, :, e_idx])

  np.save(f'{save_str}_mean_results', np.mean(all_the_results, axis=(3, 4)))


def aggregate_subsample_tests_paper_setups():
  data_ids = ['d', 'f']
  setups = ['dpcgan-', 'dpmerf-ae-', 'dpmerf-low-eps-', 'dpmerf-med-eps-', 'dpmerf-high-eps-',
            'dpmerf-nonprivate-']
  sub_ratios = ['0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  setup_with_real_data = 'dpmerf-high-eps'
  save_str = 'paper'
  aggregate_subsample_tests(data_ids, setups, sub_ratios, models, runs, eval_metrics, setup_with_real_data, save_str)


def aggregate_subsample_tests_renorm_test():
  data_ids = ['']
  setups = ['dmnist_rescale_release_off_', 'dmnist_rescale_release_on_', 'dmnist_rescale_release_clip_']
  sub_ratios = ['0.1', '0.01', '0.001']
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  eval_metrics = ['accuracies']
  setup_with_real_data = 'dmnist_rescale_release_off_'
  save_str = 'renorm'
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

  np.save(f'{save_str}_full_results', all_the_results)

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

  np.save(f'{save_str}_mean_results', np.mean(all_the_results, axis=(3, 4)))



def plot_subsampling_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps',
            'dpmerf-nonprivate']
  sub_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  # models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
  #           'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  mean_mat = np.load('paper_mean_results.npy')

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
      plt.savefig(f'plot_{d}_{e}.png')


def plot_subsampling_logreg_performance():
  data_ids = ['d', 'f']
  setups = ['real_data', 'dpcgan', 'dpmerf-ae', 'dpmerf-low-eps', 'dpmerf-med-eps', 'dpmerf-high-eps']
  sub_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
  model_idx = 1
  models = ['logistic_reg', 'random_forest', 'gaussian_nb', 'bernoulli_nb', 'linear_svc', 'decision_tree', 'lda',
            'adaboost', 'mlp', 'bagging', 'gbm', 'xgboost']
  # runs = [0, 1, 2, 3, 4]
  eval_metrics = ['accuracies', 'f1_scores']
  all_mat = np.load('all_the_results.npy')
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
  mean_mat = np.load('renorm_mean_results.npy')

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


if __name__ == '__main__':
  # dpcgan_plot()
  # dpgan_plot()
  # collect_synth_benchmark_results()
  aggregate_subsample_tests_paper_setups()
  # aggregate_subsample_tests_renorm_test()
  plot_subsampling_performance()
  # plot_subsampling_logreg_performance()
  # plot_renorm_performance()
  # extract_numpy_data_mats()
  # aggregate_mar12_setups()


# python3 train_dp_generator.py --ae-enc-spec 300,100 --ae-dec-spec 100 --ae-load-dir logs/ae/d0_1_0/ --log-name d0_2_0 -bs 500 -lr 1e-3 -denc 10 -dcode 10 -ep 7 --gen-spec 100,100 --ae-no-bias --d-rff 10000 --rff-sigma 80 --ae-bn --batch-norm --gen-labels --uniform-labels -noise 0.7 --synth-mnist
