import numpy as np
from scipy.special import factorial
from mmd_approx_eigen import eval_hermite_pytorch
import torch
import kernel as k
from aux import meddistance
import math
#s

def get_hp_losses(train_loader, device, n_labels, order, rho, bs, smp_mult, mmd_computation, sampling_rate, sr_me_division,  single_release=True, sample_dims=False, heuristic_sigma=True):
    # print('Sampling Rate is ', sampling_rate)    
    if (smp_mult):
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
        
        if (heuristic_sigma):
            med             =   meddistance(data_tensor.detach().cpu().numpy())
            alpha = 1 / (2.0 * (med**2))
            xi = -1/2/alpha+np.sqrt(1/alpha**2+4)/2
        else:
            xi  =   rho
        n_data, dim_data    =   data_tensor.shape
        batch_num           =   np.zeros([1])
        np.ceil(n_data/bs, out=batch_num)
        batch_num   =   int(batch_num)
        rchoice     =   np.zeros(shape=[batch_num, int(np.floor(dim_data*sampling_rate)) ])
        
        for j in range(batch_num):
            rchoice[j, :]     =   np.random.choice(np.arange(dim_data), size=int(np.floor(dim_data*sampling_rate)))
            data_ten          =   data_tensor[:, rchoice[j, :]]
            mean1             =   mean_embedding_proxy(data_ten, label_tensor, order, xi, device, n_labels, sr_me_division, False)
            
        
    # print(label_tensor.size)
        def hp_loss(gen_enc, gen_labels):
            if (sample_dims):
                n_data, dim_data    =   data_tensor.shape
                # rchoice     =   np.random.choice(np.arange(dim_data), size=int(np.floor(dim_data*sampling_rate)))
                # data_ten    =   data_tensor[:, rchoice]
                gen_enc     =   gen_enc[:, rchoice[0, :]]
            # print('loss')
            if (mmd_computation=='mean_emb'):
                return mmd_mean_embedding(data_ten, label_tensor, gen_enc, gen_labels, n_labels, order, xi, device, sr_me_division=sr_me_division)
            elif (mmd_computation=='cross'):
                return mmd_loss_hp_approx(data_ten, label_tensor, gen_enc, gen_labels, n_labels, order, xi, device, sr_me_division=sr_me_division)
        hp_loss_minibatch   =   None        
    elif (single_release):
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
        
        if (heuristic_sigma):
            med             =   meddistance(data_tensor.detach().cpu().numpy())
            alpha = 1 / (2.0 * (med**2))
            xi = -1/2/alpha+np.sqrt(1/alpha**2+4)/2
        else:
            xi  =   rho
        if (sample_dims):
            n_data, dim_data    =   data_tensor.shape
            rchoice     =   np.random.choice(np.arange(dim_data), size=int(np.floor(dim_data*sampling_rate)))
            data_ten    =   data_tensor[:, rchoice]
        if (mmd_computation=='mean_emb'):
            mean1   =   mean_embedding_proxy(data_ten, label_tensor, order, xi, device, n_labels, sr_me_division, False)
    # print(label_tensor.size)
        def hp_loss(gen_enc, gen_labels):
            if (sample_dims):
                n_data, dim_data    =   data_tensor.shape
                gen_enc     =   gen_enc[:, rchoice]
            # print('loss')
            if (mmd_computation=='mean_emb'):
                return mmd_mean_embedding([], [], gen_enc, gen_labels, n_labels, order, xi, device, sr_me_division=sr_me_division, known_mean1=True, mean1=mean1)
            elif (mmd_computation=='cross'):
                return mmd_loss_hp_approx(data_ten, label_tensor, gen_enc, gen_labels, n_labels, order, xi, device, sr_me_division=sr_me_division)
        hp_loss_minibatch   =   None
    else:
        def hp_loss_minibatch(data_enc, labels, gen_enc, gen_labels):
            if (heuristic_sigma):
                med             =   meddistance(data_enc.detach().cpu().numpy())
                alpha = 1 / (2.0 * (med**2))
                xi = -1/2/alpha+np.sqrt(1/alpha**2+4)/2
            else:
                xi  =   rho
            if (sample_dims):
                n_data, dim_data    =   data_enc.shape
                rchoice     =   np.random.choice(np.arange(dim_data), size=int(np.floor(dim_data*sampling_rate)))
                data_enc    =   data_enc[:, rchoice]
                gen_enc     =   gen_enc[:, rchoice]
            # print('Size is ', int(np.floor(dim_data*sampling_rate)))
            # print('Total Size is ', dim_data)
            if (mmd_computation=='mean_emb'):
                return mmd_mean_embedding(data_enc, labels, gen_enc, gen_labels, n_labels, order, xi, device)
            elif (mmd_computation=='cross'):
                return mmd_loss_hp_approx(data_enc, labels, gen_enc, gen_labels, n_labels, order, xi, device)
        hp_loss     =   None
    return hp_loss, hp_loss_minibatch
          
def mmd_mean_embedding(data_enc, data_labels, gen_enc, gen_labels, n_labels, order, rho, device, labels_to_one_hot=False, sr_me_division=1, known_mean1=False, mean1=[]):
    if (known_mean1==False):
        mean1   =   mean_embedding_proxy(data_enc, data_labels, order, rho, device, n_labels, sr_me_division, labels_to_one_hot)
    mean2   =   mean_embedding_proxy(gen_enc, gen_labels, order, rho, device, n_labels, 1, labels_to_one_hot) #Generated data are not that big. Hence, I set sr_me_division=1 for them.
    
    return torch.norm(mean1-mean2)
    
    
    
def mean_embedding_proxy(data, label, order, rho, device, n_labels, sr_me_division, labels_to_one_hot=False):
    size_data, _    =   data.shape
    if (labels_to_one_hot==True or len(label.shape)!=1):
        # set gen labels to scalars from one-hot
        _, label = torch.max(label, dim=1)
    
    _, dim_data    =   data.shape    
    # for each label, take the associated encodings
    # print('Number of labels:', n_labels)
    mean_size   =   torch.hstack([torch.tensor(n_labels, dtype=int), (order+1)*torch.ones(size=[dim_data,], dtype=int)])
    # print('Mean Size is ', mean_size)
    # print(mean_size)
    mean_proxy  =   torch.zeros(tuple((mean_size.numpy()).tolist()), device=device)
    # mmd_real    =   0
    num_data_idx    =   torch.zeros([n_labels, ], device=device)
    for t in range(sr_me_division):
        lb          =   int(t*np.floor(size_data/sr_me_division)) #Lower Bound of indices in each data batch
        ub          =   int(np.min([(t+1)*np.floor(size_data/sr_me_division),size_data]))
        print('ub is ', ub, ' and lb is ', lb)
        subdata     =   data[lb:ub, :] 
        sublabel    =   label[lb:ub][:]  
        for idx in range(n_labels):
          # print(data.shape)
          # print('Indexed dimensions are ', (sublabel==idx).shape)
          idx_data_enc          =   subdata[sublabel == idx][:]
          num_data_id  , _      =   idx_data_enc.shape
          num_data_idx[idx]     +=  num_data_id
          for idx_data in range(num_data_id):
              # print(idx_data)
              mp    =  tensor_fmap_hp(idx_data_enc[idx_data, :], order, rho, device)
              # print(uu.size())
              mean_proxy[idx, :]    +=    mp
        print('Mean Proxy Shape is ', mean_proxy.shape, ' and num_data_idx shape is ', num_data_idx.shape)
        mean_proxy  =   torch.div(mean_proxy.T,num_data_idx).T
          
    return mean_proxy

def tensor_fmap_hp(data, order, rho, device):
    data_dim    =    data.size()[0]
    # print('Data dim is ' , data_dim)
    # print('Data shape is ', data.shape)
    # print('Data[0] shape is ', data[0].shape)
    
    fmap, _    =   feature_map_HP(order, data[0].unsqueeze(0).unsqueeze(0), rho, device)
    fmap        =   fmap[0, :]
    for dim in range(data_dim-1):
        fmap_dim, _    =   feature_map_HP(order, data[dim+1].unsqueeze(0).unsqueeze(0), rho, device)
        fmap_dim    =   fmap_dim[0, :]
        # print('Fmap_dim is ', fmap_dim)
        if (dim!=0):
            fmap    =   torch.einsum('i...l, j->i...lj', fmap, fmap_dim)#matmul(fmap.unsqueeze(dim+1), fmap_dim.unsqueeze(0))
        else:
            fmap    =   torch.einsum('i, j->ij', fmap, fmap_dim)
    return fmap
        
       
def mmd_loss_hp_approx(data_enc, data_labels, gen_enc, gen_labels, n_labels, order, rho, device, labels_to_one_hot=False, sr_me_division=1):
    if (labels_to_one_hot==True):
        # set gen labels to scalars from one-hot
        _, data_labels = torch.max(data_labels, dim=1)
    
    _, gen_labels = torch.max(gen_labels, dim=1)

    # for each label, take the associated encodings
    # print('Number of labels:', n_labels)
    mmd_sum = 0
    # mmd_real    =   0
    print(rho)
    for idx in range(n_labels):
# <<<<<<< HEAD
# =======
# # <<<<<<< HEAD
# >>>>>>> fb6b1f328211ba72c7242b8f8106b26695871f35
        if (torch.sum(data_labels==idx)>0 and torch.sum(data_labels==idx)>0):
            # print('The SUM is ', torch.sum(data_labels==idx))
            idx_data_enc = data_enc[data_labels == idx]
            idx_gen_enc = gen_enc[gen_labels == idx]
            # print('Data_enc Shape:', idx_data_enc.shape)
            a         = mmd_hp(idx_data_enc, idx_gen_enc, order, rho, device, sr_me_division)
            # if (math.isnan(a)):
            #     print(data_enc)
            mmd_sum   +=a
# <<<<<<< HEAD
      # idx_data_enc = data_enc[data_labels == idx]
      # idx_gen_enc = gen_enc[gen_labels == idx]
      # # print('Data_enc Shape:', idx_data_enc.shape)
      # a         = mmd_hp(idx_data_enc, idx_gen_enc, order, rho, device)
      # # if (math.isnan(a)):
      # #     print(data_enc)
      # mmd_sum   +=a
      #mmd_real  +=b
# =======
# =======
#       idx_data_enc = data_enc[data_labels == idx]
#       idx_gen_enc = gen_enc[gen_labels == idx]
#       # print('Data_enc Shape:', idx_data_enc.shape)
#       a         = mmd_hp(idx_data_enc, idx_gen_enc, order, rho, device)
#       # if (math.isnan(a)):
#       #     print(data_enc)
#       mmd_sum   +=a
# >>>>>>> d8b62a786b3a97a84413c87f1181c112dc44689c
#       #mmd_real  +=b
# >>>>>>> fb6b1f328211ba72c7242b8f8106b26695871f35

          
    #print('Real MMD is ', mmd_real)
    return mmd_sum

def mmd_hp(x, x_prime, order, rho, device, sr_me_division):

    mat_xx = mmd_prod_kernel_across_dimension_wHP(x, x, order, rho, device)
    mat_xy = mmd_prod_kernel_across_dimension_wHP(x, x_prime, order, rho, device)
    mat_yy = mmd_prod_kernel_across_dimension_wHP(x_prime, x_prime, order, rho, device)

    m = x.shape[0]
    n = x_prime.shape[0]

    # e_kxx = (torch.sum(mat_xx) - torch.sum(mat_xx.diag()))/(m*(m-1)) #Unbiased kernel estimator
    e_kxx = (torch.sum(mat_xx) )/(m*(m))
    # e_kyy = (torch.sum(mat_yy) - torch.sum(mat_yy.diag())) / (n*(n-1))
    e_kyy = (torch.sum(mat_yy)) / (n*(n))
    e_kxy = torch.sum(mat_xy)/(m*n)

    # if (len(x)==1):
    #     print(x)
    
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
        # print(phi_x_prime_axis.size())
        # phi_x_prime_mat[:, :, axis] = phi_x_prime_axis # number of datapoints by order
        # print('Size of Phi_x_axis:', phi_x_axis.size(), "and size of Phi_x_prime_axis: ", phi_x_prime_axis.size(), "and Size of matmat: ", matmat.size())
        # print('Firs size is ', phi_x_axis.shape, ' and second size is ', phi_x_prime_axis.size)
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
    # print('Device of Rho is ', rho.device, 'and device of x is ', x.device)
    exp_trm = torch.exp(-rho / (1 + rho) * (x ** 2))  # output dim: number of datapoints by 1
    N_k = (2 ** orders) * ((orders+1).to(torch.float).lgamma().exp()) * torch.sqrt(((1 - rho) / (1 + rho)))
    eigen_funcs = 1 / torch.sqrt(N_k) * (H_k * exp_trm)  # output dim: number of datapoints by number of degree
    return eigen_funcs


# u=(mean_embedding_proxy(torch.randn(size=[100, 4], device=torch.device('cpu')), torch.zeros([100], device=torch.device('cpu')), 4
#                      , torch.tensor(0.5, device=torch.device('cpu'))
#                      , torch.device('cpu'), 1))