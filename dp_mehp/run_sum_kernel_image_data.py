### this script is to test DP-MEHP on MNIST and F-MNIST data

import torch
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from all_aux_files import FCCondGen, ConvCondGen, find_rho, find_order, ME_with_HP
from all_aux_files import get_dataloaders, log_args, datasets_colletion_def, test_results_subsampling_rate
from all_aux_files import synthesize_data_with_uniform_labels, test_gen_data, flatten_features, log_gen_data
from autodp import privacy_calibrator
import matplotlib
matplotlib.use('Agg')
import argparse
from all_aux_files_tab_data import heuristic_for_length_scale
from all_aux_files_tab_data import ME_with_HP_tab, find_rho_tab

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import  LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost
from collections import defaultdict, namedtuple
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import accuracy_score

def get_args():

    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='digits', help='options are digits or fashion')

    # OPTIMIZATION
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
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
    parser.add_argument('--report-intermediate-result', default=False, help='test synthetic data on logistic regression at every epoch')
    parser.add_argument('--heuristic-sigma', action='store_true', default=False)
    parser.add_argument("--separate-kernel-length", action='store_true', default=False)  # heuristic-sigma has to be "True", to enable separate-kernel-length
    parser.add_argument('--kernel-length', type=float, default=0.01, help='')
    parser.add_argument('--order-hermite', type=int, default=100, help='')
    parser.add_argument('--sampling_rate_synth', type=float, default=0.1, help='')
    parser.add_argument('--skip-downstream-model', action='store_false', default=False, help='')

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


def load_data(data_name, batch_size):
    transform_digits = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    transform_fashion = transforms.Compose([transforms.ToTensor()])

    if data_name == 'digits':
        train_dataset = datasets.MNIST(root='data', train=True, transform=transform_digits, download=True)
    elif data_name == 'fashion':
        train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform_fashion, download=True)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers=0,shuffle=True)
    return train_loader

def main():

    # load settings
    ar = get_args()
    torch.manual_seed(ar.seed)

    data_name = ar.data_name
    single_release = ar.single_release
    report_intermediate_result = ar.report_intermediate_result
    batch_size = ar.batch_size
    test_batch_size = ar.test_batch_size
    model_name = ar.model_name
    n_epochs = ar.epochs

    private = ar.is_private # this flag can be true or false only when single_release is true.
    if private:
        epsilon = ar.epsilon
        delta = ar.delta

    # method = 'sum_kernel' # sum_kernel or a_Gaussian_kernel
    # loss_type = 'MEHP'
    subsampling_rate_for_synthetic_data = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Load data to test """
    # train_loader = load_data(data_name, batch_size)
    if data_name == 'fashion':
        data_pkg = get_dataloaders(data_name, batch_size, test_batch_size, True, False, [], [])
    elif data_name == 'digits':
        normalize = 1
        data_dir = 'data'
        use_cuda = 1 if torch.cuda.is_available() else 0
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transforms_list = [transforms.ToTensor()]
        if normalize:
          mnist_mean = 0.1307
          mnist_sdev = 0.3081
          transforms_list.append(transforms.Normalize((mnist_mean,), (mnist_sdev,)))
        prep_transforms = transforms.Compose(transforms_list)
        trn_data = datasets.MNIST(data_dir, train=True, download=True, transform=prep_transforms)
        tst_data = datasets.MNIST(data_dir, train=False, transform=prep_transforms)
        train_loader = torch.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)

        train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                               'train_data', 'test_data',
                                                               'n_features', 'n_data', 'n_labels', 'eval_func'])
        n_features = 784
        n_data = 60_000
        n_labels = 10
        eval_func = None

        data_pkg = train_data_tuple_def(train_loader, test_loader, trn_data, tst_data, n_features, n_data, n_labels, eval_func)

        def prep_data(dataset):
            x, y = dataset.data.numpy(), dataset.targets.numpy()
            x = np.reshape(x, (-1, 784))
            # x = np.reshape(x, (-1, 784)) / 255
            x = (x-mnist_mean)/mnist_sdev
            return x, y

        x_trn, y_trn = prep_data(trn_data)
        x_tst, y_tst = prep_data(tst_data)
        np.savez('data/MNIST/numpy_dmnist.npz', x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst)

    train_loader = data_pkg.train_loader
    n_train_data = 60000

    """ Define a generator """
    input_size = 5 # dimension of z
    feature_dim = 784
    n_classes = 10
    if model_name == 'FC':
        model = FCCondGen(input_size, '500,500', feature_dim, n_classes, use_sigmoid=False, batch_norm=True, use_clamp=True).to(device)
    elif model_name == 'CNN':
        # if data_name=='fashion':
        model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
        # else: # digits with smaller kernel size
        #     model = ConvCondGen(input_size, '500,500', n_classes, '16,8', '3,3', use_sigmoid=True, batch_norm=True).to(
        #     device)


    """ set the scale length """
    num_iter = np.int(n_train_data / batch_size)


    if ar.heuristic_sigma:
        if ar.separate_kernel_length:
            sigma_all = []
            for batch_idx, (data, labels) in enumerate(train_loader):
                if batch_idx <=5:
                    print('batch_idx', batch_idx)
                    data, labels = data.to(device), labels.to(device)
                    data = flatten_features(data)
                    sigma = heuristic_for_length_scale(data_name, data.cpu().detach().numpy(), [], feature_dim, [])
                    sigma_all.append(sigma)

            sigma = np.mean(sigma_all, 0)
            sigma2 = sigma ** 2
        else:
            if data_name=='digits':
                sigma2 = 0.05
            elif data_name=='fashion':
                sigma2 = 0.07
    else:
        sigma2 = ar.kernel_length
    # print('sigma2 is', sigma2)


    if ar.separate_kernel_length:
        rho = find_rho_tab(sigma2)
    else:
        rho = find_rho(sigma2)
    # ev_thr = 1e-6  # eigen value threshold, below this, we wont consider for approximation
    # order = find_order(rho, ev_thr)
    # or_thr = ar.order_hermite
    # if order>or_thr:
    #     order = or_thr
    #     print('chosen order is', order)
    order = ar.order_hermite

    if single_release:
        print('single release is', single_release)
        print('computing mean embedding of data')
        data_embedding = torch.zeros(feature_dim*(order+1), n_classes, num_iter, device=device)
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            data = flatten_features(data)
            for idx in range(n_classes):
                idx_data = data[labels == idx]
                if ar.separate_kernel_length:
                    phi_data = ME_with_HP_tab(idx_data, order, rho, device, n_train_data)
                else:
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
                    if ar.separate_kernel_length:
                        synth_data_embedding[:, idx] = ME_with_HP_tab(idx_synth_data, order, rho, device, batch_size)
                    else:
                        synth_data_embedding[:, idx] = ME_with_HP(idx_synth_data, order, rho, device, batch_size)
            else:
                synth_data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                data_embedding = torch.zeros((feature_dim * (order+1), n_classes), device=device)
                _, gen_labels_numerical = torch.max(gen_labels, dim=1)
                for idx in range(n_classes):
                    idx_data = data[labels == idx]
                    if ar.separate_kernel_length:
                        data_embedding[:, idx] = ME_with_HP_tab(idx_data, order, rho, device, batch_size)
                    else:
                        data_embedding[:, idx] = ME_with_HP(idx_data, order, rho, device, batch_size)
                    idx_synth_data = gen_samples[gen_labels_numerical == idx]
                    if ar.separate_kernel_length:
                        synth_data_embedding[:, idx] = ME_with_HP_tab(idx_synth_data, order, rho, device, batch_size)
                    else:
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

    # if data_name == 'fashion':

    # """ load synthetic data for training """
    # file_name = 'logs/gen/digits_FC_single_release=True_order=100_private=False_epsilon=1.0_delta=1e-05_heuristic_sigma=False_kernel_length=0.02_bs=200_lr=0.01_nepoch=0/digits/synthetic_mnist.npz'
    # td = np.load(file_name)
    # X_tr = td['data']
    # y_tr = td['labels'].squeeze()
    #
    # """ load real data for testing """
    # d = np.load('data/MNIST/numpy_dmnist.npz')
    # X_te = d['x_test'].reshape(10000, 784)
    # y_te = d['y_test']

    test_results_subsampling_rate(ar.data_name, ar.log_name + '/' + ar.data_name, ar.log_dir, ar.skip_downstream_model, ar.sampling_rate_synth)

    # elif data_name == 'digits':
    #
    #     """ load synthetic data for training """
    #     # file_name = 'logs/gen/digits_FC_single_release=True_order=200_private=False_epsilon=1.0_delta=1e-05_heuristic_sigma=False_kernel_length=0.0005_bs=200_lr=0.01_nepoch=10/digits/synthetic_mnist.npz'
    #     # td = np.load(file_name)
    #     # X_tr = td['data']
    #     # y_tr = td['labels'].squeeze()
    #     X_tr = syn_data
    #     y_tr = syn_labels.squeeze()
    #
    #     """ load real data for testing """
    #     d = np.load('data/MNIST/numpy_dmnist.npz')
    #     X_te = d['x_test'].reshape(10000, 784)
    #     y_te = d['y_test']
    #
    #     acc_arr = []
    #     """ test classifiers """
    #     models_to_test = np.array(
    #         [
    #          LogisticRegression(solver='lbfgs', max_iter=50000, multi_class='auto'),
    #          GaussianNB(),
    #          BernoulliNB(binarize=0.5),
    #          LinearSVC(),
    #          DecisionTreeClassifier(),
    #          LinearDiscriminantAnalysis(),
    #          AdaBoostClassifier(),
    #          BaggingClassifier(),
    #          RandomForestClassifier(),
    #          GradientBoostingClassifier(subsample=0.1, n_estimators=50),
    #          MLPClassifier(),
    #         xgboost.XGBClassifier(disable_default_eval_metric=True, learning_rate=0.5)
    #         ])
    #
    #     for model in models_to_test:
    #
            # print('\n', type(model))
            # model.fit(X_tr, y_tr)
            # pred = model.predict(X_te)  # test on real data
            # acc = accuracy_score(pred, y_te)
    #
    #         prior_class = 1/n_classes*np.ones(n_classes)
    #
    #         if str(model)[0:10] == 'GaussianNB':
    #             print('training again')
    #
    #             model = GaussianNB(var_smoothing=0.2, priors=prior_class)
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc1 = accuracy_score(pred, y_te)
    #
    #             acc = max(acc, acc1)
    #
    #
    #         elif str(model)[0:12] == 'DecisionTree':
    #
    #             print('training again')
    #             model = DecisionTreeClassifier(criterion='gini', class_weight='balanced', max_depth=100, max_leaf_nodes=100)
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc1 = accuracy_score(pred, y_te)
    #             print('DT acc1', acc1)
    #
    #             acc = max(acc, acc1)
    #
    #
    #         elif str(model)[0:8] == 'AdaBoost':
    #
    #             model = AdaBoostClassifier(n_estimators=1000, learning_rate=5.0, random_state=0)
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc5 = accuracy_score(pred, y_te)
    #             print('adaboost acc5', acc5)
    #
    #             model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.7, random_state=0, algorithm='SAMME.R')
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc6 = accuracy_score(pred, y_te)
    #             print('adaboost acc6', acc6)
    #
    #             acc = max(acc, acc5, acc6)
    #
    #         elif str(model)[0:7] == 'Bagging':
    #
    #             model = BaggingClassifier(n_estimators=200, warm_start=True, max_features=40)  # improved
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc1 = accuracy_score(pred, y_te)
    #             print('bagging acc1', acc1)
    #
    #             acc = max(acc, acc1)
    #
    #         elif str(model)[0:9] == 'LinearSVC':
    #
    #             model = LinearSVC(max_iter=10000, tol=1e-16, loss='hinge',multi_class = 'crammer_singer', C=0.0001)
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc1 = accuracy_score(pred, y_te)
    #             print('linearSVC acc1', acc1)
    #
    #             acc = max(acc, acc1)
    #
    #         elif str(model)[0:16] == 'GradientBoosting':
    #
    #             model = GradientBoostingClassifier(subsample=0.5, n_estimators=100, learning_rate=0.05)
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc4 = accuracy_score(pred, y_te)
    #             print('GradientBoosting acc4', acc4)
    #
    #             model = GradientBoostingClassifier(subsample=0.5, n_estimators=100, learning_rate=0.001)
    #             model.fit(X_tr, y_tr)
    #             pred = model.predict(X_te)  # test on real data
    #             acc6 = accuracy_score(pred, y_te)
    #             print('GradientBoosting acc6', acc6)
    #
    #             acc = max(acc, acc4, acc6)
    #
    #
    #
    #         print("accuracy on test data is %.3f" % (acc))
    #         acc_arr.append(acc)
    #
    #
    #     acc_avg = np.mean(acc_arr)
    #     print("------\n mean acc across classifiers is %.3f\n" % acc_avg)
    #



if __name__ == '__main__':
    main()
