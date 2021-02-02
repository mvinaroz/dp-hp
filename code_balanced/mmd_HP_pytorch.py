import numpy as np
from scipy.special import factorial
from mmd_approx_eigen import eval_hermite_pytorch
import torch
import kernel as k


def get_hp_losses(train_loader, device, n_labels, order, rho):
    data_acc    =   []
    label_acc   =   []
    i   =   0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        data_acc.append(data)
        label_acc.append(labels)
        i   +=1
    # print('Size of train_loader: ', i, "and size of labels", labels.size())
        
    s1              =   np.hstack([len(data_acc)*data.shape[0], data.shape[1:]]).astype(int)
    s2              =   np.hstack([len(label_acc)*labels.shape[0], labels.shape[1:]]).astype(int)
    data_tensor     =   torch.zeros(list(s1), device=device)
    label_tensor    =   torch.zeros(list(s2), device=device)
    torch.cat(data_acc, out=data_tensor)
    torch.cat(label_acc, out=label_tensor)
    data_tensor     =   torch.flatten(data_tensor, start_dim=1)
    # print(label_tensor.size)
    def hp_loss(gen_enc, gen_labels):
        return mmd_loss_hp_approx(data_tensor, label_tensor, gen_enc, gen_labels, n_labels, order, rho, device)
    return hp_loss
          
          
def mmd_loss_hp_approx(data_enc, data_labels, gen_enc, gen_labels, n_labels, order, rho, device, labels_to_one_hot=False):
    if (labels_to_one_hot==True):
        # set gen labels to scalars from one-hot
        _, data_labels = torch.max(data_labels, dim=1)
    
    _, gen_labels = torch.max(gen_labels, dim=1)

    # for each label, take the associated encodings
    # print('Number of labels:', n_labels)
    mmd_sum = 0
    mmd_real    =   0
    for idx in range(n_labels):
      idx_data_enc = data_enc[data_labels == idx]
      idx_gen_enc = gen_enc[gen_labels == idx]
      # print('Data_enc Shape:', idx_data_enc.shape)
      a         = mmd_hp(idx_data_enc, idx_gen_enc, order, rho, device)
      mmd_sum   +=a
      #mmd_real  +=b
      
    #print('Real MMD is ', mmd_real)
    return mmd_sum

def mmd_hp(x, x_prime, order, rho, device):

    mat_xx = mmd_prod_kernel_across_dimension_wHP(x, x, order, rho, device)
    mat_xy = mmd_prod_kernel_across_dimension_wHP(x, x_prime, order, rho, device)
    mat_yy = mmd_prod_kernel_across_dimension_wHP(x_prime, x_prime, order, rho, device)

    m = x.shape[0]
    n = x_prime.shape[0]

    e_kxx = (torch.sum(mat_xx) - torch.sum(mat_xx.diag()))/(m*(m-1))
    e_kyy = (torch.sum(mat_yy) - torch.sum(mat_yy.diag())) / (n*(n-1))
    e_kxy = torch.sum(mat_xy)/(m*n)
    
    mmd_approx = e_kxx + e_kyy - 2.0*e_kxy

    sigma2  =   (1-rho**2)/(2*rho)
    
    #I add the real kernel to see the differences
    # Gaussian_kernel = k.KGauss(sigma2=rho)
    # Kxy             = np.mean(Gaussian_kernel(x.cpu().detach().numpy(), x_prime.cpu().detach().numpy()))
    # Kxx             = np.mean(Gaussian_kernel(x.cpu().detach().numpy(), x.cpu().detach().numpy()))
    # Kyy             = np.mean(Gaussian_kernel(x_prime.cpu().detach().numpy(), x_prime.cpu().detach().numpy()))
    
    # mmd_real    =   Kxx+Kxy-2.0*Kxy
    
    return mmd_approx#, mmd_real=0

def mmd_hp_norm(x, x_prime, order, rho, device):
    return 0
    
    
    
def mmd_prod_kernel_across_dimension_wHP(x, x_prime, order, rho, device):
    n_data, input_dim = x.shape
    n_generated_data = x_prime.shape[0]

    # phi_x_mat = torch.zeros((n_data, order+1, input_dim))
    # phi_x_prime_mat = torch.zeros((n_generated_data, order+1, input_dim))
    matmat = torch.ones((n_data, n_generated_data), device=device)
    for axis in np.arange(input_dim):
        # print(axis)
        x_axis = x[:, axis]
        x_axis = x_axis[:, np.newaxis]
        phi_x_axis, eigen_vals_axis = feature_map_HP(order, x_axis,rho,device)
        # phi_x_mat[:, :, axis] = phi_x_axis # number of datapoints by order

        x_prime_axis = x_prime[:, axis]
        x_prime_axis = x_prime_axis[:, np.newaxis]
        phi_x_prime_axis, eigen_vals_prime_axis = feature_map_HP(order, x_prime_axis,rho,device)
        print(phi_x_prime_axis.size())
        # phi_x_prime_mat[:, :, axis] = phi_x_prime_axis # number of datapoints by order
        # print('Size of Phi_x_axis:', phi_x_axis.size(), "and size of Phi_x_prime_axis: ", phi_x_prime_axis.size(), "and Size of matmat: ", matmat.size())
        matmat = matmat * torch.einsum('ab, cb -> ac', phi_x_axis, phi_x_prime_axis) # size:  # datapoints in x by # datapoints in x_prime

    return matmat

def feature_map_HP(k, x, rho, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1, device=device))
    eigen_funcs_x = eigen_func(k, rho, x, device)  # output dim: number of datapoints by number of degree
    phi_x = torch.einsum('ij,j-> ij', eigen_funcs_x, torch.sqrt(eigen_vals)) # number of datapoints by order
    # n_data = eigen_funcs_x.shape[0]
    # mean_phi_x = torch.sum(phi_x,0)/n_data

    return phi_x, eigen_vals

def eigen_func(k, rho, x, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at, size: number of data points by input_dimension
    orders = torch.arange(0, k + 1, device=device)
    H_k = eval_hermite_pytorch(x, k+1, device, return_only_last_term=False)
    H_k = H_k[:,:,0]
    # H_k = eval_hermite(orders, x)  # input arguments: degree, where to evaluate at.
    # output dim: number of datapoints by number of degree
    rho     =   torch.tensor(rho, dtype=float, device=device)
    exp_trm = torch.exp(-rho / (1 + rho) * (x ** 2))  # output dim: number of datapoints by 1
    N_k = (2 ** orders) * ((orders+1).to(torch.float).lgamma().exp()) * torch.sqrt(((1 - rho) / (1 + rho)))
    eigen_funcs = 1 / torch.sqrt(N_k) * (H_k * exp_trm)  # output dim: number of datapoints by number of degree
    return eigen_funcs
