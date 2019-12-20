"""" test a simple generating training using MMD for relatively simple datasets """
""" with generating labels together with the input features """
# Mijung wrote on Dec 20, 2019

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    labels = np.zeros((n, how_many_Gaussians))

    for i in np.arange(0,how_many_Gaussians):

        how_many_samps = np.int(n/how_many_Gaussians)
        new_samps = np.random.multivariate_normal(mean_param[:, i], cov_param[:, :, i], how_many_samps)
        data_samps[(i*how_many_samps):((i+1)*how_many_samps),:] = new_samps

        labels[(i*how_many_samps):((i+1)*how_many_samps),i] = 1


    idx = np.random.permutation(n)
    shuffled_x = data_samps[idx,:]
    shuffled_y = labels[idx,:]

    return shuffled_x, shuffled_y


def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
    return Z


class Generative_Model(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, n_classes):
            super(Generative_Model, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size
            self.n_classes = n_classes

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.relu(output)
            output = self.fc3(output)

            output_features = output[:, 0:-self.n_classes]
            output_labels = output[:, -self.n_classes:]
            # output[:,-self.n_classes:] = self.softmax(output[:, -self.n_classes:])
            output_total = torch.cat((output_features, output_labels), 1)
            return output_total

def main():

    n = 5000 # number of data points divisable by num_Gassians
    num_Gaussians = 2
    input_dim = 2
    mean_param = np.zeros((input_dim, num_Gaussians))
    cov_param = np.zeros((input_dim, input_dim, num_Gaussians))

    mean_param[:, 0] = [2, 8]
    mean_param[:, 1] = [-10, -4]
    # mean_param[:, 2] = [-8, -7]

    cov_mat = np.empty((2,2))
    cov_mat[0,0] = 1
    cov_mat[1,1] = 4
    cov_mat[0,1] = 0.2
    cov_mat[1,0] = 0.2
    cov_param[:, :, 0] = cov_mat
    cov_param[:, :, 1] = cov_mat

    # cov_param[:, :, 0] = 2*np.eye(input_dim)
    # cov_param[:, :, 1] = 1 * np.eye(input_dim)
    # cov_param[:, :, 2] = 4 * np.eye(input_dim)

    data_samps, true_labels = generate_data(mean_param, cov_param, n)

    # print(data_samps)
    # plt.plot(data_samps[:,0], data_samps[:,1], 'o')
    # plt.show()

    # n = 100
    # d = 3
    # data_samps = np.random.randn(n, d) * 4.0 + np.random.rand(n, d) * 2

    # test how to use RFF for computing the kernel matrix
    med = util.meddistance(data_samps)
    sigma2 = med**2
    print('length scale from median heuristic is', sigma2)

    # sigma2 = 150 # larger than the value from the median heuristic

    # # print('Median heuristic distance (squared): {}'.format(sigma2))
    # sigma2 = 0.1

    # # Gaussian kernel
    # k = kernel.KGauss(sigma2=sigma2)
    # K = k.eval(data_samps, data_samps)
    #
    # random Fourier features
    n_features = 200
    n_classes = num_Gaussians

    # fmap = feature.RFFKGauss(sigma2=sigma2, n_features=num_features)
    #
    # Phi = fmap(data_samps)
    # Kapprox = Phi.dot(Phi.T)

    """ training a Generator via minimizing MMD """
    mini_batch_size = 1000

    input_size = 10
    hidden_size_1 = 100
    hidden_size_2 = 50
    output_size = input_dim + n_classes

    # model = Generative_Model(input_dim=input_dim, how_many_Gaussians=num_Gaussians)
    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size, n_classes = n_classes)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    how_many_epochs = 500
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    # fm = feature.RFFKGauss(sigma2, n_features=n_features)
    # observed_mean_feature = np.mean(fm(data_samps), axis=0)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)
    mean_emb1_input_features = torch.mean(RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq), 0)

    mean_emb_label_normalizer = torch.sqrt(1.0/torch.Tensor([n_classes]))
    mean_emb1_labels = torch.mean(torch.Tensor(true_labels),0)*mean_emb_label_normalizer

    mean_emb1 = torch.cat((mean_emb1_input_features, mean_emb1_labels),0)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            # for p in model.parameters():
            #     p.data.clamp_(-1.0, 1.0)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(torch.randn((mini_batch_size, input_size)))

            samp_input_features = outputs[:,0:input_dim]
            samp_labels = outputs[:,-n_classes:]

            mean_emb2_input_features = torch.mean(RFF_Gauss(n_features, samp_input_features, W_freq), 0)
            mean_emb2_labels = torch.mean(samp_labels,0)*mean_emb_label_normalizer
            mean_emb2 = torch.cat((mean_emb2_input_features, mean_emb2_labels), 0)

            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2
            # loss = mmd2_biased(inputs, outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if running_loss<=1e-4:
            break
        print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss

    plt.figure(1)
    plt.subplot(121)
    plt.plot(data_samps[:, 0], data_samps[:, 1], 'o')
    # plt.plot(training_loss_per_epoch)
    # plt.title('MMD as a function of epoch')
    # plt.show()

    plt.subplot(122)
    model.eval()
    # generated_samples = model(torch.randn((mini_batch_size, input_dim_z)))
    # generated_samples = model(torch.randn((n, input_size)))
    # generated_samples = generated_samples.detach().numpy()
    generated_samples = samp_input_features.detach().numpy()
    # plt.plot(generated_samples[:,0], generated_samples[:,1], 'o')
    # print('mean of samples is', np.mean(generated_samples,0))
    # print('true mean is', mean_param[:,0])
    #
    # print('cov of samples is', np.cov(generated_samples.transpose()))
    # print('true cov is', cov_mat)

    generated_labels = samp_labels.detach().numpy()
    labl = np.argmax(generated_labels, axis=1)
    plt.scatter(generated_samples[:,0], generated_samples[:,1], c=labl, label=labl)


    plt.figure(2)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')

    #
    # from_model_params = list(model.parameters())
    # estimated_params = from_model_params[0]
    # estimated_mean_params = torch.reshape(estimated_params[0:num_Gaussians * input_dim], (input_dim, num_Gaussians))
    # estimated_var_params = F.softplus(estimated_params[num_Gaussians * input_dim:num_Gaussians * (input_dim + 1)])
    # estimated_mean_params = estimated_mean_params.detach().numpy()
    # estimated_var_params = estimated_var_params.detach().numpy()
    #
    #
    # print('true mean : ', mean_param)
    # print('estimated mean : ', estimated_mean_params)
    # print('estimated var: ', estimated_var_params)

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

