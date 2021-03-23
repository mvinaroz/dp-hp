### this script is to test DP-MEHP on tabular data

import torch
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from all_aux_files import find_rho, find_order, ME_with_HP
from all_aux_files import log_args
# from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data, flatten_features, log_gen_data
from autodp import privacy_calibrator
import matplotlib
matplotlib.use('Agg')
import argparse
from all_aux_files_tab_data import data_loading, Generative_Model_homogeneous_data, Generative_Model_heterogeneous_data, heuristic_for_length_scale
from sklearn.preprocessing import OneHotEncoder

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='digits', help='options are digits or fashion')

    # OPTIMIZATION
    parser.add_argument("--batch-rate", type=float, default=0.1)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

    # MODEL DEFINITION
    parser.add_argument('--model-name', type=str, default='FC', help='either CNN or FC')

    # DP SPEC
    parser.add_argument('--is-private', default=False, help='produces a DP mean embedding of data')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in (epsilon, delta)-DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta in (epsilon, delta)-DP')

    # OTHERS
    parser.add_argument('--single-release', action='store_true', default=True, help='produce a single data mean embedding')  # Here usually we have action and default be True
    parser.add_argument('--heuristic-sigma', action='store_true', default=False)
    parser.add_argument('--kernel-length', type=float, default=0.001, help='')
    parser.add_argument('--order-hermite', type=int, default=50, help='')

    ar = parser.parse_args()
    preprocess_args(ar)
    log_args(ar.log_dir, ar)

    return ar

def preprocess_args(ar):

    """ name the directories """
    base_dir = 'logs/gen/'
    log_name = ar.data_name + '_' + ar.model_name + '_' + 'single_release=' + str(ar.single_release) +  \
               '_' + 'order=' + str(ar.order_hermite) + '_' + 'private=' + str(ar.is_private) + '_' \
               + 'epsilon=' + str(ar.epsilon) + '_' + 'delta=' + str(ar.delta) + '_' \
               + 'heuristic_sigma=' + str(ar.heuristic_sigma) + '_' + 'kernel_length=' + str(ar.kernel_length) + '_' \
                + 'bs=' + str(ar.batch_size) + '_' + 'lr=' + str(ar.lr) + '_' \
               + 'nepoch=' + str(ar.epochs)

    ar.log_name = log_name
    ar.log_dir = base_dir + log_name + '/'
    if not os.path.exists(ar.log_dir):
        os.makedirs(ar.log_dir)

def main():

    # load settings
    ar = get_args()
    torch.manual_seed(ar.seed)

    data_name = ar.data_name
    single_release = ar.single_release
    report_intermediate_result = ar.report_intermediate_result
    model_name = ar.model_name
    n_epochs = ar.epochs

    private = ar.is_private # this flag can be true or false only when single_release is true.
    if private:
        epsilon = ar.epsilon
        delta = ar.delta

    subsampling_rate_for_synthetic_data = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # specify heterogeneous dataset or not
    heterogeneous_datasets = ['cervical', 'adult', 'census', 'intrusion', 'covtype']
    homogeneous_datasets = ['epileptic','credit','isolet', 'adult_cat']

    """ Load data to test """
    X_train, X_test, y_train, y_test, n_classes, num_categorical_inputs, num_numerical_inputs = data_loading(data_name, ar.undersampled_rate, ar.seed)

    # one-hot encoding of labels.
    n, input_dim = X_train.shape
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = onehot_encoder.fit_transform(y_train)

    ######################################
    # MODEL
    batch_size = np.int(np.round(ar.batch_rate * n))
    print("minibatch: ", batch_size)
    input_size = 10 + 1
    hidden_size_1 = 4 * input_dim
    hidden_size_2 = 2 * input_dim
    output_size = input_dim

    if data_name in homogeneous_datasets:

        model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                  hidden_size_2=hidden_size_2,
                                                  output_size=output_size, dataset=data_name).to(device)

    else: # data_name in heterogeneous_datasets:

        model = Generative_Model_heterogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                                    hidden_size_2=hidden_size_2,
                                                    output_size=output_size,
                                                    num_categorical_inputs=num_categorical_inputs,
                                                    num_numerical_inputs=num_numerical_inputs).to(device)

    """ set the scale length """
    num_iter = np.int(n / batch_size)
    if ar.heuristic_sigma:
        sigma2 = heuristic_for_length_scale(data_name, X_train, num_numerical_inputs, input_dim, heterogeneous_datasets)
        # for now, let's use the same length scale on every coordinate of the data
        # for this, we either need to normalize the data or use a separate length scale to each coordinate
        # Credit and Census datasets are affected by this. Also, heterogeous datasets.
        sigma2 = np.mean(sigma2)
    else:
        sigma2 = ar.kernel_length
    print('sigma2 is', sigma2)

    rho = find_rho(sigma2)
    ev_thr = 1e-6  # eigen value threshold, below this, we wont consider for approximation
    order = find_order(rho, ev_thr)
    or_thr = ar.order_hermite
    if order>or_thr:
        order = or_thr
        print('chosen order is', order)


    ########## Start from here again ##########
        
    if single_release:
        print('single release is', single_release)
        print('computing mean embedding of data')
        data_embedding = torch.zeros(input_dim*(order+1), n_classes, num_iter, device=device)
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)
            for idx in range(n_classes):
                idx_data = data[labels == idx]
                phi_data = ME_with_HP(idx_data, order, rho, device, n_train_data)
                data_embedding[:,idx, batch_idx] = phi_data
        data_embedding = torch.sum(data_embedding, axis=2)
        print('done with computing mean embedding of data')

        if private:
            print('we add noise to the data mean embedding as the private flag is true')
            k = 1 # how many compositions we do, here it's just once.
            privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
            privacy_param = privacy_param['sigma']
            print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param)
            std = (2 * privacy_param * np.sqrt(feature_dim) / n_train_data)
            noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

            print('before perturbation, mean and variance of data mean embedding are %f and %f ' %(torch.mean(data_embedding), torch.std(data_embedding)))
            data_embedding = data_embedding + noise
            print('after perturbation, mean and variance of data mean embedding are %f and %f ' % (torch.mean(data_embedding), torch.std(data_embedding)))

        else:
            print('we do not add noise to the data mean embedding as the private flag is false')


    """ Training """
    optimizer = torch.optim.Adam(list(model.parameters()), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    score_mat = np.zeros(n_epochs)

    print('start training the generator')
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)

            gen_code, gen_labels = model.get_code(batch_size, device)
            gen_samples = model(gen_code) # batch_size by 784

            if single_release:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, batch_size)
            else:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_data = data[labels == idx]
                    data_embedding[:, idx] = ME_with_HP(idx_data, order, rho, device, batch_size)
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, batch_size)

            loss = torch.sum((data_embedding - synth_data_embedding)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # end for

        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_train_data, loss.item()))

        log_gen_data(model, device, epoch, n_classes, ar.log_dir)
        scheduler.step()

        if report_intermediate_result:
            """ now we save synthetic data and test them on logistic regression """
            syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                                       n_data=n_train_data,
                                                                       n_labels=n_classes)

            dir_syn_data = ar.log_dir + ar.data_name + '/synthetic_mnist'
            if not os.path.exists(dir_syn_data):
                os.makedirs(dir_syn_data)

            np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
            final_score = test_gen_data(ar.log_name + '/' + data_name, data_name, subsample=subsampling_rate_for_synthetic_data, custom_keys='logistic_reg')
            print('on logistic regression, accuracy is', final_score)
            score_mat[epoch - 1] = final_score

    #     end if
    # end for

    if report_intermediate_result:
        dir_max_score = ar.log_dir + data_name + '/score_mat'
        np.save(dir_max_score+'score_mat', score_mat)
        print('scores among the training runs are', score_mat)

    """ now we save synthetic data of size 60K and test them on logistic regression """
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=n_train_data,
                                                               n_labels=n_classes)

    dir_syn_data = ar.log_dir + data_name + '/synthetic_mnist'
    if not os.path.exists(dir_syn_data):
        os.makedirs(dir_syn_data)

    np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
    final_score = test_gen_data(ar.log_name + '/' + data_name, data_name, subsample=1.0, custom_keys='logistic_reg')
    print('on logistic regression with 60K synthetic datapoints,  the accuracy is', final_score)

if __name__ == '__main__':
    main()
