"""" test a simple generating training using MMD for relatively simple datasets """
# Mijung wrote on Nov 6, 2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
# from torch.nn.parameter import Parameter
# from feature import mmd2_biased
import util
# import kernel as kernel
import feature
import scipy
# from torch.distributions.normal import Normal
from abc import ABCMeta, abstractmethod

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

# def distance_RF(mean_emb1, mean_emb2):
#     # mean_emb1_avg = torch.mean(mean_emb1,0)
#     # mean_emb2_avg = torch.mean(mean_emb2,0)
#     distance_RF_eval = torch.norm(mean_emb1 - mean_emb2, p=2)**2
#     # distance_RF_eval = torch.dist(mean_emb1_avg, mean_emb2_avg, p=2)**2
#     return distance_RF_eval

# def RFF_Gauss(sigma2, n_features, X):
#     """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
#     # Fourier transform formula from
#     # http://mathworld.wolfram.com/FourierTransformGaussian.html
#     n, d = X.size()
#     draws = n_features // 2
#
#     # sigma2 = torch.Tensor([sigma2])
#     # W = torch.randn(draws, d) / torch.sqrt(sigma2)
#     W = np.random.randn(draws, d) / np.sqrt(sigma2)
#     W = torch.Tensor(W)
#     # m = Normal()
#
#     # n x draws
#     # XWT = X.dot(W.T)
#     XWT = torch.mm(X, torch.t(W))
#     Z1 = torch.cos(XWT)
#     Z2 = torch.sin(XWT)
#
#     # n_features = torch.Tensor([n_features])
#     # Z = torch.hstack((Z1, Z2)) * torch.sqrt(1.0 / n_features)
#     Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
#     return Z
#


class FeatureMap(object):
    """Abstract class for a feature map function"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def gen_features(self, X):
        """Generate D features for each point in X.
        - X: nxd data matrix
        Return a n x D numpy array.
        """
        pass

    @abstractmethod
    def num_features(self, X=None):
        """
        Return the number of features that this map will generate for X.
        X is optional.
        """
        pass

    def __call__(self, X):
        return self.gen_features(X)



class RFFKGauss(FeatureMap):
    """
    A FeatureMap to construct random Fourier features for a Gaussian kernel.
    """
    def __init__(self, sigma2, n_features, seed=1):
        """
        n_features: number of random Fourier features. The total number of
            dimensions will be n_features. Internally draw n_features/2
            frequency components. n_features has to be even.
        """

        self.sigma2 = sigma2
        self.n_features = n_features
        self.seed =  seed

    def gen_features(self, X):
        # The following block of code is deterministic given seed.
        # Fourier transform formula from
        # http://mathworld.wolfram.com/FourierTransformGaussian.html
        with util.NumpySeedContext(seed=self.seed):
            n, d = X.shape

            draws = self.n_features//2
            W = np.random.randn(draws, d)/np.sqrt(self.sigma2)
            # n x draws
            XWT = X.dot(W.T)
            Z1 = np.cos(XWT)
            Z2 = np.sin(XWT)
            Z = np.hstack((Z1, Z2))*np.sqrt(2.0/self.n_features)

            #     n, d = X.size()
            #     draws = n_features // 2
            #
            #     # sigma2 = torch.Tensor([sigma2])
            #     # W = torch.randn(draws, d) / torch.sqrt(sigma2)
            #     W = np.random.randn(draws, d) / np.sqrt(sigma2)
            #     W = torch.Tensor(W)
            #     # m = Normal()
            #
            #     # n x draws
            #     # XWT = X.dot(W.T)
            #     XWT = torch.mm(X, torch.t(W))
            #     Z1 = torch.cos(XWT)
            #     Z2 = torch.sin(XWT)
            #
            #     # n_features = torch.Tensor([n_features])
            #     # Z = torch.hstack((Z1, Z2)) * torch.sqrt(1.0 / n_features)
            #     Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
            #     return Z

        return Z

    def num_features(self, X=None):
        return self.n_features

#
# def RFF_Gauss(sigma2, n_features, X, W):
#     """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
#     # Fourier transform formula from
#     # http://mathworld.wolfram.com/FourierTransformGaussian.html
#     # n, d = X.size()
#     # draws = n_features // 2
#
#     # sigma2 = torch.Tensor([sigma2])
#     # W = torch.randn(draws, d) / torch.sqrt(sigma2)
#     # n x draws
#     # XWT = X.dot(W.T)
#     XWT = torch.mm(X, torch.t(W))
#     Z1 = torch.cos(XWT)
#     Z2 = torch.sin(XWT)
#
#     # n_features = torch.Tensor([n_features])
#     # Z = torch.hstack((Z1, Z2)) * torch.sqrt(1.0 / n_features)
#     Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
#     return Z


# def main():
#     # debugging my linear-time MMD computation
#     n = 100
#     d = 2
#     data_samps = np.random.randn(n, d) * 4.0
#     sigma2 = 1
#     n_features = 6
#     draws = n_features//2
#     W = np.random.randn(draws, d)/np.sqrt(sigma2)
#     Z_from_PT = torch.mean(RFF_Gauss(sigma2, n_features, torch.Tensor(data_samps), torch.Tensor(W)), axis=0)
#     print('Z from pytorch code is', Z_from_PT)
#
#     fm = feature.RFFKGauss(sigma2, n_features=n_features, W=W)
#     observed_mean_feature = np.mean(fm(data_samps), axis=0)
#     print('observed_mean_feature from python code is', observed_mean_feature)




class Generative_Model(nn.Module):
    #I'm going to define my own Model here following how I generated this dataset

    def __init__(self, input_dim, how_many_Gaussians, mini_batch_size):
    # def __init__(self, input_dim, hidden_dim):
        super(Generative_Model, self).__init__()

        # number of parameters = how_many_Gaussians*(input_dim + 1)

        number_of_parameters = how_many_Gaussians * (input_dim + 1)
        # self.parameter = Parameter(2*torch.randn(number_of_parameters),requires_grad=True) # this parameter lies

        self.parameter = Parameter(torch.randn(number_of_parameters), requires_grad=True)  # this parameter lies
        self.n = mini_batch_size
        self.input_dim = input_dim
        self.how_many_Gaussians = how_many_Gaussians

    def forward(self, x):

        n = self.n
        dim_Gaussians = self.input_dim
        how_many_Gaussians = self.how_many_Gaussians

        # mean_param = torch.zeros((dim_Gaussians, how_many_Gaussians))

        parameters = self.parameter
        mean_param = torch.reshape(parameters[0:how_many_Gaussians*dim_Gaussians], (dim_Gaussians, how_many_Gaussians))
        var_Gaussian = F.softplus(parameters[how_many_Gaussians*dim_Gaussians:how_many_Gaussians * (dim_Gaussians + 1)])

        data_samps = torch.zeros((n, dim_Gaussians))
        how_many_samps = np.int(n / how_many_Gaussians)

        for i in np.arange(0, how_many_Gaussians):
            print(i)
            mean = mean_param[:,i].repeat(how_many_samps,1)
            new_samps = mean + torch.sqrt(var_Gaussian[i])*x[(i * how_many_samps):((i + 1) * how_many_samps), :]
            data_samps[(i * how_many_samps):((i + 1) * how_many_samps), :] = new_samps
            print((i * how_many_samps))
            print(((i + 1) * how_many_samps))

        idx = torch.randperm(n)
        shuffled_x = data_samps[idx, :]

        return shuffled_x

def main():

    n = 3000 # number of data points divisable by num_Gassians
    num_Gaussians = 1
    input_dim = 2
    mean_param = np.zeros((input_dim, num_Gaussians))
    cov_param = np.zeros((input_dim, input_dim, num_Gaussians))

    mean_param[:, 0] = [0.9, 0.8]
    # mean_param[:, 1] = [-0.2, 0.1]
    # mean_param[:, 2] = [-0.8, -0.7]

    cov_param[:, :, 0] = 0.01 * np.eye(input_dim)
    # cov_param[:, :, 1] = 0.02 * np.eye(input_dim)
    # cov_param[:, :, 2] = 0.04 * np.eye(input_dim)

    data_samps = generate_data(mean_param, cov_param, n)

    print(data_samps)
    plt.plot(data_samps[:,0], data_samps[:,1], 'o')
    plt.show()

    # n = 100
    # d = 3
    # data_samps = np.random.randn(n, d) * 4.0 + np.random.rand(n, d) * 2

    # test how to use RFF for computing the kernel matrix
    med = util.meddistance(data_samps)
    sigma2 = med**2

    # # print('Median heuristic distance (squared): {}'.format(sigma2))
    # sigma2 = 0.01

    # # Gaussian kernel
    # k = kernel.KGauss(sigma2=sigma2)
    # K = k.eval(data_samps, data_samps)
    #
    # random Fourier features
    n_features = 6

    # fmap = feature.RFFKGauss(sigma2=sigma2, n_features=num_features)
    #
    # Phi = fmap(data_samps)
    # Kapprox = Phi.dot(Phi.T)

    """ training a Generator via minimizing MMD """
    # hidden_dim_1 = 10
    # hidden_dim_2 = 5
    # output_dim = 2
    # input_dim_z = 2
    mini_batch_size = 50
    model = Generative_Model(input_dim=input_dim, how_many_Gaussians=num_Gaussians, mini_batch_size=mini_batch_size)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    how_many_epochs = 100
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    fm = feature.RFFKGauss(sigma2, n_features=n_features)

    observed_mean_feature = np.mean(fm(data_samps), axis=0)

    mean_emb1 = torch.mean(Gaussian_RF(sigma2, n_features, torch.Tensor(data_samps)), axis=0)


    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        print('epoch number is ', epoch)
        running_loss = 0.0

        for i in range(how_many_iter):

            # for p in model.parameters():
            #     p.data.clamp_(-0.5, 0.5)

            # print(i)
            # get the inputs

            # inputs = data_samps
            # inputs = data_samps[i*mini_batch_size:(i+1)*mini_batch_size,:]
            # # inputs_to_model = torch.randn((mini_batch_size, input_dim_z))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs_to_model = torch.randn((mini_batch_size, input_dim))
            # inputs_to_model = torch.randn((input_dim, mini_batch_size))
            outputs = model(torch.Tensor(inputs_to_model))
            # labels = torch.Tensor(labels)
            # loss = F.binary_cross_entropy(outputs, labels)
            # loss = loss_function(outputs, labels)

            pseudo_mean_feature = np.mean(fm(outputs.detach().numpy()), axis=0)
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature, ord=2) ** 2
            print('from scipy the loss is', dis)

            mean_emb2 = torch.mean(Gaussian_RF(sigma2, n_features, outputs), axis=0)


            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2
            print('loss with random data', loss.detach().numpy())



            """ this is for debugging """
            samps_from_corr_dist = generate_data(mean_param, cov_param, mini_batch_size)

            pseudo_mean_feature_crr = np.mean(fm(samps_from_corr_dist), axis=0)
            dis = scipy.linalg.norm(observed_mean_feature - pseudo_mean_feature_crr, ord=2) ** 2
            print('from scipy the loss from true data is', dis)

            mean_emb2_crr = torch.mean(Gaussian_RF(sigma2, n_features, torch.Tensor(samps_from_corr_dist)), axis=0)
            loss_crr = torch.norm(mean_emb1 - mean_emb2_crr, p=2)**2
            print('loss with true data', loss_crr.detach().numpy())


            # loss = mmd2_biased(inputs, outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        training_loss_per_epoch[epoch] = running_loss/n

    plt.figure(1)
    plt.subplot(121)
    plt.plot(data_samps[:, 0], data_samps[:, 1], 'o')
    # plt.plot(training_loss_per_epoch)
    # plt.title('MMD as a function of epoch')
    # plt.show()

    plt.subplot(122)
    model.eval()
    # generated_samples = model(torch.randn((mini_batch_size, input_dim_z)))
    generated_samples = model(torch.randn((n, input_dim)))
    generated_samples = generated_samples.detach().numpy()
    plt.plot(generated_samples[:,0], generated_samples[:,1], 'x')
    # plt.plot(data_samps[:,0], data_samps[:,1], 'o')
    plt.show()

    plt.figure(2)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')
    plt.show()

    from_model_params = list(model.parameters())
    estimated_params = from_model_params[0]
    estimated_mean_params = torch.reshape(estimated_params[0:num_Gaussians * input_dim], (input_dim, num_Gaussians))
    estimated_var_params = F.softplus(estimated_params[num_Gaussians * input_dim:num_Gaussians * (input_dim + 1)])
    estimated_mean_params = estimated_mean_params.detach().numpy()
    estimated_var_params = estimated_var_params.detach().numpy()


    print('true mean : ', mean_param)
    print('estimated mean : ', estimated_mean_params)
    print('estimated var: ', estimated_var_params)


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

