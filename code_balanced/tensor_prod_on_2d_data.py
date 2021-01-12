# author: Mijung Park
# purpose: to test the quality of approximation using Hermite Polynomials
#          on the simulated data drawn from 2D Gaussians
# date: 07 Jan 2021

import numpy as np
import kernel as kernel
from aux import meddistance
from scipy.special import eval_hermite
from scipy.special import factorial
from scipy import optimize
from synth_data_2d import make_dataset, plot_data
from architectures import Generative_Model
from mmd_real import get_real_mmd_loss
from mmd_approx_eigen import eval_hermite_pytorch
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
# from mmd_real import mmd2_loss
from kernel import mmd2_biased
# import matplotlib
# import tkinter
# matplotlib.use('TkAgg')

def feature_map_HP(order, x, rho, input_dim, device):
    n_data = x.shape[0]
    phi_x_mat = torch.zeros((n_data, order+1, input_dim)) # mean embedding of real data
    for axis in np.arange(input_dim):
        # print(axis)
        x_axis = x[:, axis]
        x_axis = x_axis[:, np.newaxis]
        phi_x_axis, eigen_vals_axis = mean_embedding_HP(order, x_axis,rho,device)
        phi_x_mat[:, :, axis] = phi_x_axis

    return phi_x_mat

def mean_embedding_HP(k, x, rho, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1))
    eigen_funcs_x = eigen_func(k, rho, x, device)  # output dim: number of datapoints by number of degree
    phi_x = torch.einsum('ij,j-> ij', eigen_funcs_x, np.sqrt(eigen_vals)) # number of datapoints by order
    # n_data = eigen_funcs_x.shape[0]
    # mean_phi_x = torch.sum(phi_x,0)/n_data

    return phi_x, eigen_vals

def eigen_func(k, rho, x, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at, size: number of data points by input_dimension
    orders = torch.arange(0, k + 1)
    H_k = eval_hermite_pytorch(x, k+1, device, return_only_last_term=False)
    H_k = H_k[:,:,0]
    # H_k = eval_hermite(orders, x)  # input arguments: degree, where to evaluate at.
    # output dim: number of datapoints by number of degree
    exp_trm = torch.exp(-rho / (1 + rho) * (x ** 2))  # output dim: number of datapoints by 1
    N_k = (2 ** orders) * (factorial(orders)) * np.sqrt((1 - rho) / (1 + rho))
    eigen_funcs = 1 / np.sqrt(N_k) * (H_k * exp_trm)  # output dim: number of datapoints by number of degree
    return eigen_funcs


def main():
    device = 'cpu'
    """
    1. Produce data
    """
    n_data = 1000
    n_classes = 1
    data_samples, label_samples, eval_func, class_centers = make_dataset(n_classes=n_classes,
                                                                         n_samples=n_data,
                                                                         n_rows=1,
                                                                         n_cols=1,
                                                                         noise_scale=0.2,
                                                                         discrete=False)
    plot_data(data_samples, label_samples, 'synth_2d_data_plot', center_frame=True, title='')
    # print(eval_func(data_samples, label_samples))

    """ 
    2. Train a generative model for producing synthetic data 
    """
    mini_batch_size = 200
    input_size = 10 # size of inputs to the generator
    hidden_size_1 = 100
    hidden_size_2 = 50
    input_dim = 2 # dimension of data
    output_size = input_dim + n_classes

    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size, n_classes = n_classes)


    # """
    # 3. Using the data, test if the pytorch version of hermite polynomial is the same as the scipy version.
    # """
    # k = 10
    # axis = 0
    # orders = np.arange(0, k+1)
    # x_axis = data_samples[:100,axis]
    # x_axis = x_axis[:,np.newaxis]
    # H_k = eval_hermite(orders,x_axis)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(H_k[:,:])
    #
    # device = 'cpu'
    # H_k_prime = eval_hermite_pytorch(torch.from_numpy(x_axis), k+1, device, return_only_last_term=False)
    # H_k_prime = H_k_prime.detach().numpy()
    # plt.subplot(212)
    # plt.plot(H_k_prime[:, :, 0])
    # plt.show()
    # # np.sum((H_k-H_k_prime)**2)
    # # seems to be the same with some floating point differences.

    """
    4. Determine up to which order we want to keep
    """
    # choose an appropriate length scale.
    med = meddistance(data_samples, subsample=5000)
    sigma2 = med**2
    alpha = 1 / (2.0 * sigma2)
    # from this: alpha = rho / (1- rho**2), identify what rho is
    sol = optimize.minimize_scalar(lambda r: (alpha - r / (1 - r ** 2)) ** 2, bounds=(0, 1), method='bounded')
    rho = sol.x
    print(med, alpha, rho)
    k = 100
    eigen_vals = (1 - rho) * (rho ** np.arange(0, k + 1))
    print('eigen values are ', eigen_vals)
    eigen_val_threshold = 1e-6
    idx_keep = eigen_vals > eigen_val_threshold
    keep_eigen_vals = eigen_vals[idx_keep]
    print('keep_eigen_vals are ', keep_eigen_vals)
    order = len(keep_eigen_vals)
    print('The number of orders for Hermite Polynomials is', order)

    """ 
    5. Compute the data mean embedding by approximating a Gaussian kernel using HPs.
    """
    x = torch.Tensor(data_samples)
    fm_x = feature_map_HP(order, x, rho, input_dim, device) # n_data by order by input_dim
    mean_emb1 = torch.mean(fm_x,0)
    # emb1_lables = torch.nn.functional.one_hot(torch.tensor(label_samples)) # n_data by n_classes
    # outer_emb1 = torch.einsum('kip,kj->kipj', [fm_x, emb1_lables])
    # mean_emb1 = torch.mean(outer_emb1, 0) # order by input_dim by n_classes

    """
    5. Training the generator
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    how_many_epochs = 100
    how_many_iter = np.int(n_data/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(torch.randn((mini_batch_size, input_size)))

            samp_input_features = outputs[:,0:input_dim]
            samp_labels = outputs[:,-n_classes:]

            """
            1. Test the usual MMD^2
            """
            loss = mmd2_loss(data_samples, samp_input_features, sigma2)

            """ computing mean embedding of generated samples """
            # fm_x = feature_map_HP(order, x, rho, input_dim, device)  # n_data by order by input_dim
            # emb1_lables = torch.nn.functional.one_hot(torch.tensor(label_samples))  # n_data by n_classes
            # outer_emb1 = torch.einsum('kip,kj->kipj', fm_x, emb1_lables)
            # mean_emb1 = torch.mean(outer_emb1, 0)  # order by input_dim by n_classes

            fm_x_prime = feature_map_HP(order, samp_input_features, rho, input_dim, device)
            mean_emb2 = torch.mean(fm_x_prime,0)
            # emb2_labels = samp_labels
            # outer_emb2 = torch.einsum('kip,kj->kipj', fm_x_prime, emb2_labels)
            # mean_emb2 = torch.mean(outer_emb2, 0) # order by input_dim by n_classes
            #

            """
            Compare approximate to real
            """
            # # The right thing to do
            # a1 = torch.einsum('xy, ay -> xa', fm_x[:,:,0], fm_x[:,:,0])
            # a2 = torch.einsum('xy, ay -> xa', fm_x[:, :, 1], fm_x[:, :, 1])
            # xx = torch.mean(torch.einsum('xy,xy->xy', a1, a2))
            #
            # a1 = torch.einsum('xy, ay -> xa', fm_x_prime[:,:,0], fm_x_prime[:,:,0])
            # a2 = torch.einsum('xy, ay -> xa', fm_x_prime[:, :, 1], fm_x_prime[:, :, 1])
            # yy = torch.mean(torch.einsum('xy,xy->xy', a1, a2))
            #
            # a1 = torch.einsum('xy, ay -> xa', fm_x[:,:,0], fm_x_prime[:,:,0])
            # a2 = torch.einsum('xy, ay -> xa', fm_x[:, :, 1], fm_x_prime[:, :, 1])
            # cross = torch.mean(torch.einsum('xy,xy->xy', a1, a2))
            #
            # loss_1 = xx + yy - 2*cross

            # approx_xx = torch.dot(mean_emb1[:,0], mean_emb1[:, 0])*torch.dot(mean_emb1[:, 1], mean_emb1[:, 1])
            #
            # loss =  approx_xx + approx_yy - 2*approx_xy

            # approx_yy = torch.dot(mean_emb2[:,0], mean_emb2[:,0])*torch.dot(mean_emb2[:,1], mean_emb2[:,1])
            # approx_xy = torch.dot(mean_emb1[:,0], mean_emb2[:,0])*torch.dot(mean_emb1[:,1], mean_emb2[:,1])
            # approx_xx = torch.dot(mean_emb1[:,0], mean_emb1[:, 0])*torch.dot(mean_emb1[:, 1], mean_emb1[:, 1])
            #
            # loss =  approx_xx + approx_yy - 2*approx_xy

            # Gaussian_kernel = kernel.KGauss(sigma2=sigma2)
            # Kxy = np.mean(Gaussian_kernel(data_samples, samp_input_features.detach().numpy()))
            #
            # print('approximation_xy ', approx_xy)
            # print('true1 ', Kxy)
            #
            # Kyy = np.mean(Gaussian_kernel(samp_input_features.detach().numpy(), samp_input_features.detach().numpy()))
            # print('approximation_yy', approx_yy)
            # print('true2 ', Kyy)


            # MMD_prox = torch.zeros(n_classes)
            # for axis in torch.arange(n_classes):
            #     # mean_phi_x_prime = np.sum(phi_x_prime, 0) / n_data
            #     # dot_prod_axis = np.dot(mean_phi_x, mean_phi_x_prime)
            #     phi_x_prime = mean_emb2[:,:,axis]
            #     phi_x = mean_emb1[:,:,axis]
            #     MMD_data = torch.prod(torch.diag(torch.matmul(phi_x.T, phi_x)))
            #     MMD_pseudo = torch.prod(torch.diag(torch.matmul(phi_x_prime.T, phi_x_prime)))
            #     MMD_cross = torch.prod(torch.diag(torch.matmul(phi_x.T, phi_x_prime)))
            #     MMD_prox[axis] = MMD_data + MMD_pseudo - 2*MMD_cross
            #
            # loss = torch.norm(MMD_prox, p=2)**2
            # loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

            # loss = torch.norm(mean_emb1[:,0]-mean_emb2[:,0], p=2) + torch.norm(mean_emb1[:,1]-mean_emb2[:,1], p=2) + torch.norm(mean_emb1[:,2]-mean_emb2[:,2], p=2)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # if running_loss<=1e-4:
        #     break
        print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss

    # for plotting
    outputs = model(torch.randn((n_data, input_size)))
    samp_input_features = outputs[:, 0:input_dim]
    samp_labels = outputs[:, -n_classes:]
    samp_labels = torch.argmax(samp_labels,1)
    plot_data(samp_input_features.detach().numpy(), samp_labels.detach().numpy(), 'generated_2d_data_plot', center_frame=True, title='')












if __name__ == '__main__':
    main()