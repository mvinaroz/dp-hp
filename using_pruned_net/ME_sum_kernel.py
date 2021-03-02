import torch
import numpy as np
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from all_aux_files import FCCondGen, ConvCondGen, flatten_features, meddistance
from all_aux_files import eval_hermite_pytorch
from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data, flatten_features, log_gen_data

def load_data(data_name, batch_size):
    transform_digits = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    transform_fashion = transforms.Compose([transforms.ToTensor()])

    if data_name == 'digits':
        train_dataset = datasets.MNIST(root='data', train=True, transform=transform_digits, download=True)
    elif data_name == 'fashion':
        train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform_fashion, download=True)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=0,shuffle=True)
    return train_loader

def find_rho(sigma2):
    alpha = 1 / (2.0 * sigma2)
    rho = -1 / 2 / alpha + np.sqrt(1 / alpha ** 2 + 4) / 2
    return rho

def find_order(rho,eigen_val_threshold):
    k = 100
    eigen_vals = (1 - rho) * (rho ** np.arange(0, k + 1))
    idx_keep = eigen_vals > eigen_val_threshold
    keep_eigen_vals = eigen_vals[idx_keep]
    print('keep_eigen_vals are ', keep_eigen_vals)
    order = len(keep_eigen_vals)
    print('The number of orders for Hermite Polynomials is', order)
    return order

def ME_with_HP(x, order, rho, device):
    n_data, input_dim = x.shape

    # reshape x, such that x is a long vector
    x_flattened = x.view(-1)
    x_flattened = x_flattened[:,None]
    phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, torch.tensor(rho), device)
    phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order+1)
    phi_x = torch.mean(phi_x, axis=0)
    phi_x = phi_x.view(-1) # size: input_dim*(order+1)

    # phi_x_mat = torch.zeros(input_dim*(order+1))
    # for axis in torch.arange(input_dim):
    #     # print(axis)
    #     x_axis = x[:, axis]
    #     x_axis = x_axis[:, None]
    #     phi_x_axis, eigen_vals_axis = feature_map_HP(order, x_axis, torch.tensor(rho), device)
    #     me = torch.mean(phi_x_axis,axis=0)
    #     phi_x_mat[axis*(order+1):(axis+1)*(order+1)] = me
    #     # phi_x_mat.append(me[:,None])
    # phi_x_mat = phi_x_mat[:,None]
    # phi_x_mat = phi_x_mat.to(device)

    return phi_x


def feature_map_HP(k, x, rho, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at
    # print('device', device)
    eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1))
    eigen_vals = eigen_vals.to(device)
    # print('eigen_vals', eigen_vals)
    eigen_funcs_x = eigen_func(k, rho, x, device)  # output dim: number of datapoints by number of degree
    # print('eigen_funcs', eigen_funcs_x)

    phi_x = torch.einsum('ij,j-> ij', eigen_funcs_x, torch.sqrt(eigen_vals))  # number of datapoints by order
    # n_data = eigen_funcs_x.shape[0]
    # mean_phi_x = torch.sum(phi_x,0)/n_data

    return phi_x, eigen_vals


def eigen_func(k, rho, x, device):
    # k: degree of polynomial
    # rho: a parameter (related to length parameter)
    # x: where to evaluate the function at, size: number of data points by input_dimension
    # print('device', device)
    orders = torch.arange(0, k + 1, device=device)
    H_k = eval_hermite_pytorch(x, k + 1, device, return_only_last_term=False)
    H_k = H_k[:, :, 0]
    # H_k = eval_hermite(orders, x)  # input arguments: degree, where to evaluate at.
    # output dim: number of datapoints by number of degree
    # rho = torch.tensor(rho, dtype=float, device=device)
    # print('Device of Rho is ', rho.device, 'and device of x is ', x.device)
    exp_trm = torch.exp(-rho / (1 + rho) * (x ** 2))  # output dim: number of datapoints by 1
    N_k = (2 ** orders) * ((orders + 1).to(torch.float).lgamma().exp()) * torch.sqrt(((1 - rho) / (1 + rho)))
    eigen_funcs = 1 / torch.sqrt(N_k) * (H_k * exp_trm)  # output dim: number of datapoints by number of degree
    return eigen_funcs


def main():

    data_name = 'digits' # 'digits' or 'fashion'
    method = 'sum_kernel' # sum_kernel or a_Gaussian_kernel
    single_release = True
    model_name = 'CNN' # CNN or FC
    report_intermidiate_result = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Load data to test """
    batch_size = 100
    train_loader = load_data(data_name, batch_size)
    n_train_data = 60000

    """ Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    n_epochs = 20
    if model_name == 'FC':
        model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=False, batch_norm=True).to(device)
    elif model_name == 'CNN':
        model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)

    """ Training """
    # set the scale length
    num_iter = np.int(n_train_data/batch_size)
    sigma2_arr = 0.05
    # sigma2_arr = np.zeros((np.int(num_iter), feature_dim))
    # for batch_idx, (data, labels) in enumerate(train_loader):
    #     # print('batch idx', batch_idx)
    #     data, labels = data.to(device), labels.to(device)
    #     data = flatten_features(data) # minibatch by feature_dim
    #     data_numpy = data.detach().cpu().numpy()
    #     for dim in np.arange(0,feature_dim):
    #         med = meddistance(np.expand_dims(data_numpy[:,dim],axis=1))
    #         sigma2 = med ** 2
    #         sigma2_arr[batch_idx, dim] = sigma2


    base_dir = 'logs/gen/'
    log_dir = base_dir + data_name + method + model_name + '/'
    log_dir2 = data_name + method + model_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    sigma2 = np.mean(sigma2_arr)
    rho = find_rho(sigma2)
    ev_thr = 1e-3  # eigen value threshold, below this, we wont consider for approximation
    order = find_order(rho, ev_thr)
    or_thr = 30
    if order>or_thr:
        order = or_thr
    if single_release:
        data_embedding = torch.zeros(feature_dim*(order+1), n_classes, num_iter)
        for batch_idx, (data, labels) in enumerate(train_loader):
            # print(batch_idx)
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)
            for idx in range(n_classes):
                idx_data = data[labels == idx]
                phi_data = ME_with_HP(idx_data, order, rho, device)
                data_embedding[:,idx, batch_idx] = phi_data
        data_embedding = torch.mean(data_embedding,axis=2)


    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            optimizer.zero_grad()
            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            if single_release:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes))
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device)
            else:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes))
                data_embedding = torch.zeros((feature_dim * (order+1), n_classes))
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_data = data[labels == idx]
                    data_embedding[:, idx] = ME_with_HP(idx_data, order, rho, device)
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device)

            loss = torch.mean((data_embedding - synth_data_embedding) ** 2)

            loss.backward()
            optimizer.step()
        # end for

        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_train_data, loss.item()))

        log_gen_data(model, device, epoch, n_classes, log_dir)
        # scheduler.step()

        if report_intermidiate_result:
            """ now we save synthetic data and test them on logistic regression """
            syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                                       n_data=n_train_data,
                                                                       n_labels=n_classes)

            dir_syn_data = log_dir + data_name + '/synthetic_mnist'
            if not os.path.exists(dir_syn_data):
                os.makedirs(dir_syn_data)

            np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
            final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=0.1, custom_keys='logistic_reg')
            print('on logistic regression, accuracy is', final_score)

    #     end if
    # end for

    #########################################################################3
    """ Once we have a trained generator, we store synthetic data from it and test them on logistic regression """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=n_train_data,
                                                               n_labels=n_classes)

    dir_syn_data = log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
    final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=0.1, custom_keys='logistic_reg')
    print('on logistic regression, accuracy is', final_score)


if __name__ == '__main__':
    main()
