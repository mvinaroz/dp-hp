"""" test a simple generating training using MMD for relatively simple datasets """
# Mijung wrote on Nov 6, 2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import feature as feature
import util as util
import kernel as kernel

# generate data from 2D Gaussian for sanity check
def generate_data(mean_param, cov_param, n):

    how_many_Gaussians = mean_param.shape[1]
    dim_Gaussians = mean_param.shape[0]
    data_samps = np.zeros((n, dim_Gaussians))

    for i in np.arange(0,how_many_Gaussians):
        print(i)

        how_many_samps = np.int(n/how_many_Gaussians)
        new_samps = np.random.multivariate_normal(mean_param[:, i], cov_param[:, :, i], how_many_samps)
        data_samps[(i*how_many_samps):((i+1)*how_many_samps),:] = new_samps
        print((i*how_many_samps))
        print(((i+1)*how_many_samps))

    idx = np.random.permutation(n)
    shuffled_x = data_samps[idx,:]

    return shuffled_x

def Gaussian_RF(sigma2, n_features, X):
    mean_emb = RFF_Gauss(sigma2, n_features, X)
    return mean_emb

def distance_RF(mean_emb1, mean_emb2):
    distance_RF_eval = torch.dist(mean_emb1, mean_emb2, p=2)**2
    return distance_RF_eval

def RFF_Gauss(sigma2, n_features, X):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html
    n, d = X.size()
    draws = n_features // 2

    sigma2 = torch.Tensor([sigma2])
    W = torch.randn(draws, d) / torch.sqrt(sigma2 * 2.0 ** 0.5)
    # n x draws
    # XWT = X.dot(W.T)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    # n_features = torch.Tensor([n_features])
    # Z = torch.hstack((Z1, Z2)) * torch.sqrt(1.0 / n_features)
    Z = torch.cat((Z1, Z2),1) * torch.sqrt(1.0/torch.Tensor([n_features]))
    return Z


class Generative_Model(nn.Module):

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
    # def __init__(self, input_dim, hidden_dim):
        super(Generative_Model, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output


def main():

    n = 2000 # number of data points divisable by num_Gassians
    num_Gaussians = 3
    input_dim = 2
    mean_param = np.zeros((input_dim, num_Gaussians))
    cov_param = np.zeros((input_dim, input_dim, num_Gaussians))

    mean_param[:, 0] = [6, 2]
    mean_param[:, 1] = [-1, 2]
    mean_param[:, 2] = [4, -3]

    cov_param[:, :, 0] = 0.5 * np.eye(input_dim)
    cov_param[:, :, 1] = 1 * np.eye(input_dim)
    cov_param[:, :, 2] = 0.2 * np.eye(input_dim)

    data_samps = generate_data(mean_param, cov_param, n)

    # print(data_samps)
    # plt.plot(data_samps[:,0], data_samps[:,1], 'o')
    # plt.show()

    # n = 100
    # d = 3
    # data_samps = np.random.randn(n, d) * 4.0 + np.random.rand(n, d) * 2

    # test how to use RFF for computing the kernel matrix
    med = util.meddistance(data_samps)
    sigma2 = med**2

    print('Median heuristic distance (squared): {}'.format(sigma2))

    # # Gaussian kernel
    # k = kernel.KGauss(sigma2=sigma2)
    # K = k.eval(data_samps, data_samps)
    #
    # random Fourier features
    n_features = 10

    # fmap = feature.RFFKGauss(sigma2=sigma2, n_features=num_features)
    #
    # Phi = fmap(data_samps)
    # Kapprox = Phi.dot(Phi.T)

    """ training a Generator via minimizing MMD """
    hidden_dim_1 = 10
    hidden_dim_2 = 5
    output_dim = 2
    model = Generative_Model(input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, output_dim=output_dim)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    mini_batch_size = 100
    how_many_epochs = 200
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        print('epoch number is ', epoch)
        running_loss = 0.0

        for i in range(how_many_iter):

            # print(i)
            # get the inputs
            inputs = data_samps[i*mini_batch_size:(i+1)*mini_batch_size,:]
            inputs_to_model = torch.randn((mini_batch_size, input_dim))
            # print(inputs.shape)
            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(torch.Tensor(inputs_to_model))
            # labels = torch.Tensor(labels)
            # loss = F.binary_cross_entropy(outputs, labels)
            # loss = loss_function(outputs, labels)

            mean_emb1 = Gaussian_RF(sigma2, n_features, torch.Tensor(inputs))
            mean_emb2 = Gaussian_RF(sigma2, n_features, outputs)

            loss = distance_RF(mean_emb1, mean_emb2)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        training_loss_per_epoch[epoch] = running_loss/n

    plt.figure(1)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')
    plt.show()

    # Diff = K - Kapprox
    # print(np.sqrt(np.sum(Diff**2)))
    # Reldiff = np.abs(Diff / (1e-3 + K))
    #
    #
    # plt.figure(1)
    # plt.imshow(Reldiff, interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure(2)
    # plt.hist(Reldiff.reshape(-1))
    # plt.xlabel('Relative errors')
    # plt.ylabel('Count')
    # plt.show()

    # training the generative model using the MMD loss






if __name__ == '__main__':
    main()

