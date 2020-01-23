# This is for computing the cumulative privacy loss of our algorithm
# We use the analytic moments accountant method by Wang et al
# (their github repo is : https://github.com/yuxiangw/autodp)
# by changing the form of upper bound on the Renyi DP, resulting from
# several Gaussian mechanisms we use given a mini-batch.
from autodp import rdp_acct, rdp_bank


def main():

    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma = 0.5

    # (2) desired delta level
    delta = 1e-5

    # (5) number of training steps
    batch_size = 1000
    n_epochs = 40
    n_data = 60000
    steps_per_epoch = n_data // n_data
    n_steps = steps_per_epoch * n_epochs

    # (6) sampling rate
    prob = batch_size / n_data

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    eps_seq = []
    print_every_n = 100
    for i in range(1, n_steps+1):
        acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), prob)
        eps_seq.append(acct.get_eps(delta))
        if i % print_every_n == 0 or i == n_steps:
            print("[", i, "]Privacy loss is", (eps_seq[-1]))

    print("Composition of 1000 subsampled Gaussian mechanisms gives ", (acct.get_eps(delta), delta))


if __name__ == '__main__':
    main()
