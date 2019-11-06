"""" test a simple generating training using MMD for relatively simple datasets """
# Mijung wrote on Nov 6, 2019

import numpy as np
import matplotlib.pyplot as plt

# generate data from 2D Gaussian for sanity check
def generate_data(mean_param, cov_param, n):
    how_many_Gaussians = np.shape(mean_param, 1)
    data_samps = []

    for i in np.range(0,how_many_Gaussians):
        new_samps = np.random.multivariate_normal(mean_param[:, i], cov_param[:, :, i], np.int(n / how_many_Gaussians))
        data_samps = [data_samps, new_samps]
    return data_samps

def main():

    n = 1000 # number of data points
    num_Gaussians = 3
    input_dim = 2
    mean_param = np.zeros(input_dim, num_Gaussians)
    cov_param = np.zeros(input_dim, input_dim, num_Gaussians)

    mean_param[:, 0] = [2, 2]
    mean_param[:, 1] = [-1, 2]
    mean_param[:, 2] = [1, -3]

    cov_param[:, :, 0] = 2 * np.eye(input_dim)
    cov_param[:, :, 1] = 1 * np.eye(input_dim)
    cov_param[:, :, 2] = 0.2 * np.eye(input_dim)

    data_samps = generate_data(mean_param, cov_param, n)
    print(data_samps)
    plt.plot(data_samps, 'o')
    plt.show()



if __name__ == '__main__':
    main()