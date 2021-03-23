### this script is to test DP-MEHP on tabular data

import torch
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from all_aux_files import FCCondGen, ConvCondGen
from all_aux_files import find_rho, find_order, ME_with_HP
from all_aux_files import log_args
# from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data, flatten_features, log_gen_data
from autodp import privacy_calibrator
import matplotlib
matplotlib.use('Agg')
import argparse
from all_aux_files_tab_data import data_loading, Generative_Model_homogeneous_data, Generative_Model_heterogeneous_data, heuristic_for_length_scale
from all_aux_files_tab_data import test_models
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='epileptic', \
                        help='choose among cervical, adult, census, intrusion, covtype, epileptic, credit, isolet')

    # OPTIMIZATION
    parser.add_argument("--batch-rate", type=float, default=0.5)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
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
    parser.add_argument('--kernel-length', type=float, default=0.01, help='')
    parser.add_argument('--order-hermite', type=int, default=100, help='')
    parser.add_argument("--undersampled-rate", type=float, default=1.0)

    parser.add_argument('--classifiers', nargs='+', type=int, help='list of integers',
                      default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    ar = parser.parse_args()
    preprocess_args(ar)
    log_args(ar.log_dir, ar)

    return ar
# kernel-length 0.1
# roc mean across methods is 0.587
# prc mean across methods is 0.321

# kernel-length 0.01
# roc mean across methods is 0.623
# prc mean across methods is 0.356

def preprocess_args(ar):

    """ name the directories """
    base_dir = 'logs/gen/'
    log_name = ar.data_name + '_' + ar.model_name + '_' + 'single_release=' + str(ar.single_release) +  \
               '_' + 'order=' + str(ar.order_hermite) + '_' + 'private=' + str(ar.is_private) + '_' \
               + 'epsilon=' + str(ar.epsilon) + '_' + 'delta=' + str(ar.delta) + '_' \
               + 'heuristic_sigma=' + str(ar.heuristic_sigma) + '_' + 'kernel_length=' + str(ar.kernel_length) + '_' \
                + 'br=' + str(ar.batch_rate) + '_' + 'lr=' + str(ar.lr) + '_' \
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
    n_epochs = ar.epochs
    model_name = ar.model_name
    if ar.is_private:
        epsilon = ar.epsilon
        delta = ar.delta

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # specify heterogeneous dataset or not
    heterogeneous_datasets = ['cervical', 'adult', 'census', 'intrusion', 'covtype']
    homogeneous_datasets = ['epileptic','credit','isolet']

    """ Load data to test """
    X_train, X_test, y_train, y_test, n_classes, num_categorical_inputs, num_numerical_inputs = data_loading(data_name, ar.undersampled_rate, ar.seed)

    # one-hot encoding of labels.
    n, input_dim = X_train.shape

    # one hot encode the labels
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = np.expand_dims(y_train, 1)
    true_labels = onehot_encoder.fit_transform(y_train)

    # standardize the inputs
    print('normalizing the data')
    # X_train = preprocessing.scale(X_train)
    # X_test = preprocessing.scale(X_test)
    X_train = preprocessing.minmax_scale(X_train, feature_range=(0, 1), axis=0, copy=True)
    X_test = preprocessing.minmax_scale(X_test, feature_range=(0, 1), axis=0, copy=True)

    ######################################
    # MODEL
    batch_size = np.int(np.round(ar.batch_rate * n))
    print("minibatch: ", batch_size)
    input_size = 10 + 1
    # hidden_size_1 = 500
    # hidden_size_2 = 500
    hidden_size_1 = 4 * input_dim
    hidden_size_2 = 2 * input_dim
    output_size = input_dim

    # if model_name == 'FC':
    #     model = FCCondGen(input_size, '500,500', output_size, n_classes, use_sigmoid=True, batch_norm=True).to(device)
    # elif model_name == 'CNN':
    #     model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)

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
    if ar.heuristic_sigma:
        print('we use the median heuristic for length scale')
        sigma = heuristic_for_length_scale(data_name, X_train, num_numerical_inputs, input_dim, heterogeneous_datasets)
        sigma2 = np.mean(sigma**2)
        print('sigma2 value is', sigma2)
    else:
        sigma2 = ar.kernel_length
    print('sigma2 is', sigma2)

    rho = find_rho(sigma2)
    ev_thr = 1e-10  # eigen value threshold, below this, we wont consider for approximation
    order = find_order(rho, ev_thr)
    or_thr = ar.order_hermite
    if order>or_thr:
        order = or_thr
        print('chosen order is', order)


    ########## data mean embedding ##########
    """ compute the weights """
    print('computing mean embedding of data: (1) compute the weights')
    unnormalized_weights = np.sum(true_labels, 0)
    weights = unnormalized_weights / np.sum(unnormalized_weights) # weights = m_c / n
    print('\n weights with no privatization are', weights, '\n')

    if ar.is_private:
        print("private")
        k = 2 # because we add noise to the weights and means separately.
        privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
        print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

        sensitivity_for_weights = np.sqrt(2) / n  # double check if this is sqrt(2) or 2
        noise_std_for_weights = privacy_param['sigma'] * sensitivity_for_weights
        weights = weights + np.random.randn(weights.shape[0]) * noise_std_for_weights
        weights[weights < 0] = 1e-3  # post-processing so that we don't have negative weights.
        print('weights after privatization are', weights)

    """ compute the means """
    print('computing mean embedding of data: (2) compute the mean')
    data_embedding = torch.zeros(input_dim*(order+1), n_classes, device=device)
    for idx in range(n_classes):
        idx_data = X_train[y_train.squeeze()==idx,:]
        phi_data = ME_with_HP(torch.Tensor(idx_data), order, rho, device, n)
        data_embedding[:,idx] = phi_data # this includes 1/n factor inside
    print('done with computing mean embedding of data')

    if ar.is_private:
        print('we add noise to the data mean embedding as the private flag is true')
        std = (2 * privacy_param * np.sqrt(input_dim) / n)
        noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

        print('before perturbation, mean and variance of data mean embedding are %f and %f ' %(torch.mean(data_embedding), torch.std(data_embedding)))
        data_embedding = data_embedding + noise
        print('after perturbation, mean and variance of data mean embedding are %f and %f ' % (torch.mean(data_embedding), torch.std(data_embedding)))

    # the final mean embedding of data is,
    data_embedding = data_embedding / torch.Tensor(weights).to(device) # this means, 1/n * n/m_c, so 1/m_c

    """ Training """
    optimizer = torch.optim.Adam(list(model.parameters()), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    print('start training the generator')
    num_iter = np.int(n / batch_size)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()

        for i in range(num_iter):

            """ (1) produce labels uniformly across different classes """
            label_input = torch.multinomial(1 / n_classes * torch.ones(n_classes), batch_size, replacement=True).type(torch.FloatTensor)

            if data_name in homogeneous_datasets:  # In our case, if a dataset is homogeneous, then it is a binary dataset.

                label_input = label_input.to(device)
                feature_input = torch.randn((batch_size, input_size - 1)).to(device)
                input_to_model = torch.cat((feature_input, label_input[:, None]), 1)

            else:  # heterogeneous data

                label_input = torch.cat((label_input, torch.arange(len(weights), out=torch.FloatTensor()).unsqueeze(0)),1)  # to avoid no labels
                label_input = label_input.transpose_(0, 1)
                label_input = label_input.to(device)
                feature_input = torch.randn((batch_size + len(weights), input_size - 1)).to(device)
                input_to_model = torch.cat((feature_input, label_input), 1)

            """ (2) produce data """
            outputs = model(input_to_model)

            """ (3) compute synthetic data's mean embedding """
            weights_syn = torch.zeros(n_classes) # weights = m_c / n
            syn_data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)
            for idx in range(n_classes):
                weights_syn[idx] = torch.sum(label_input == idx)
                idx_syn_data = outputs[label_input == idx]
                phi_syn_data = ME_with_HP(idx_syn_data, order, rho, device, batch_size)
                syn_data_embedding[:, idx] = phi_syn_data  # this includes 1/n factor inside

            weights_syn = weights_syn / torch.sum(weights_syn)
            syn_data_embedding = syn_data_embedding / torch.Tensor(weights_syn).to(device)

            loss = torch.sum(data_embedding - syn_data_embedding) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
        scheduler.step()

    """ Once the training step is over, we produce 60K samples and test on downstream tasks """
    """ now we save synthetic data of size 60K and test them on logistic regression """
    #######################################################################33
    if data_name in heterogeneous_datasets:

        """ draw final data samples """
        # (1) generate labels
        label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
        label_input = label_input.transpose_(0, 1)
        label_input = label_input.to(device)

        # (2) generate corresponding features
        feature_input = torch.randn((n, input_size - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input), 1)
        outputs = model(input_to_model)

        # (3) round the categorial features
        output_numerical = outputs[:, 0:num_numerical_inputs]
        output_categorical = outputs[:, num_numerical_inputs:]
        output_categorical = torch.round(output_categorical)

        output_combined = torch.cat((output_numerical, output_categorical), 1)

        generated_input_features_final = output_combined.cpu().detach().numpy()
        generated_labels_final = label_input.cpu().detach().numpy()

        roc, prc = test_models(generated_input_features_final, generated_labels_final, X_test, y_test, n_classes, "generated", ar.classifiers)
        roc_return, prc_return = roc, prc

    else:  # homogeneous datasets

        """ draw final data samples """
        # weights[1] represents the fraction of the positive labels in the dataset,
        # and we would like to generate a similar fraction of the postive/negative datapoints
        label_input = (1 * (torch.rand((n)) < weights[1])).type(torch.FloatTensor)
        label_input = label_input.to(device)

        feature_input = torch.randn((n, input_size - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
        outputs = model(input_to_model)

        # if arguments.save_generated:
        #     save_generated_samples(outputs)

        samp_input_features = outputs

        label_input_t = torch.zeros((n, n_classes))
        idx_1 = (label_input == 1.).nonzero()[:, 0]
        idx_0 = (label_input == 0.).nonzero()[:, 0]
        label_input_t[idx_1, 1] = 1.
        label_input_t[idx_0, 0] = 1.

        samp_labels = label_input_t

        generated_input_features_final = samp_input_features.cpu().detach().numpy()
        generated_labels_final = samp_labels.cpu().detach().numpy()
        generated_labels = np.argmax(generated_labels_final, axis=1)

        f1 = test_models(generated_input_features_final, generated_labels, X_test, y_test, n_classes, "generated", ar.classifiers)
        # test_models(X_tr, y_tr, X_te, y_te, n_classes, datasettype, args):


if __name__ == '__main__':
    main()
