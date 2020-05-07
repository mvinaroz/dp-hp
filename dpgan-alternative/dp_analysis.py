# This is for computing the cumulative privacy loss of our algorithm
# We use the analytic moments accountant method by Wang et al
# (their github repo is : https://github.com/yuxiangw/autodp)
# by changing the form of upper bound on the Renyi DP, resulting from
# several Gaussian mechanisms we use given a mini-batch.
from autodp import rdp_acct, rdp_bank
import time
import numpy as np

def conservative_analysis():
  """ input arguments """

  # (1) privacy parameters for four types of Gaussian mechanisms
  sigma = 10.

  # (2) desired delta level
  delta = 1e-5

  n_epochs = 10  # 5 for DP-MERF and 17 for DP-MERF+AE
  batch_size = 64  # the same across experiments
  acct = rdp_acct.anaRDPacct()

  n_data_by_class = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

  start_time = time.time()
  subset_count = 0
  for n_data in n_data_by_class:

    steps_per_epoch = int(np.ceil(n_data / batch_size))
    n_steps = steps_per_epoch * n_epochs
    sampling_rate = batch_size / n_data

    epoch_last_batch_size = n_data % batch_size
    epoch_last_sampling_rate = epoch_last_batch_size / n_data

    # old_time = start_time
    old_time = time.time()
    for i in range(1, n_steps + 1):
      sampling_rate_i = epoch_last_sampling_rate if i % steps_per_epoch == 0 else sampling_rate
      acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), sampling_rate_i)
      if i % steps_per_epoch == 0:
        new_time = time.time()
        epochs_done = i // steps_per_epoch
        t_used = new_time - old_time
        t_total = new_time - start_time
        t_total_min = t_total / 60
        print(f'Epoch {epochs_done} done - Time used: {t_used:.2f}, Total: {t_total:.2f} ({t_total_min:.2f} minutes)')
        old_time = new_time

      if i == n_steps:
        pre_eps_time = time.time()
        subset_count += 1
        print("[", i, "]Privacy loss is", (acct.get_eps(delta)))
        post_eps_time = time.time()
        print('time to get_eps: ', post_eps_time - pre_eps_time)
        old_time = post_eps_time
    print(f'data subset {subset_count} done')


def main():
    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma = 0.7

    # (2) desired delta level
    delta = 1e-5

    # (5) number of training steps
    n_epochs = 17  # 5 for DP-MERF and 17 for DP-MERF+AE
    batch_size = 500  # the same across experiments

    n_data = 60000  # fixed for mnist
    steps_per_epoch = n_data // batch_size
    n_steps = steps_per_epoch * n_epochs
    # n_steps = 1

    # (6) sampling rate
    prob = batch_size / n_data
    # prob = 1

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    eps_seq = []

    for i in range(1, n_steps+1):
        acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), prob)
        if i % steps_per_epoch == 0 or i == n_steps:
            eps_seq.append(acct.get_eps(delta))
            print("[", i, "]Privacy loss is", (eps_seq[-1]))


if __name__ == '__main__':
    # main()
    conservative_analysis()