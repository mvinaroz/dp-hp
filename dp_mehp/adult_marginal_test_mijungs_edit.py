import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
#import util
import random
import argparse
import seaborn as sns
sns.set()
# %matplotlib inline
from autodp import privacy_calibrator
from all_aux_files_tab_data import  find_rho_tab, heuristic_for_length_scale, ME_with_HP_tab
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings('ignore')
import os
from code_tab.marginals_eval import gen_data_alpha_way_marginal_eval
from code_tab.binarize_adult import binarize_data
from all_aux_files_tab_data import undersample

# ############################## generative models to use ###############################
class Generative_Model_homogeneous_data(nn.Module):
  def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, out_fun):
    super(Generative_Model_homogeneous_data, self).__init__()

    self.input_size = input_size
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    # self.hidden_size_3 = hidden_size_3
    self.output_size = output_size
    assert out_fun in ('lin', 'sigmoid', 'relu')

    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
    self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
    self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
    self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
    # self.bn3 = torch.nn.BatchNorm1d(self.hidden_size_3)
    # self.fc4 = torch.nn.Linear(self.hidden_size_3, self.output_size)
    # self.bn4 = torch.nn.BatchNorm1d(self.output_size)
    if out_fun == 'sigmoid':
      self.out_fun = nn.Sigmoid()
    elif out_fun == 'relu':
      self.out_fun = nn.ReLU()
    else:
      self.out_fun = nn.Identity()

  def forward(self, x):
    hidden = self.fc1(x)
    relu = self.relu(self.bn1(hidden))
    output = self.fc2(relu)
    relu = self.relu(self.bn2(output))
    output = self.fc3(relu)
    # relu = self.relu(self.bn3(output))
    # output = self.fc4(relu)
    output = self.out_fun(output)
    # output = torch.round(output) # so that we make the output as categorical
    return output


class Generative_Model_heterogeneous_data(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, categorical_columns,
                 binary_columns):
        super(Generative_Model_heterogeneous_data, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.categorical_columns = categorical_columns
        self.binary_columns = binary_columns

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.relu(self.bn2(output))
        output = self.fc3(output)

        output_binary = self.sigmoid(output[:, 0:len(self.binary_columns)])
        output_categorical = self.relu(output[:, len(self.binary_columns):])
        output_combined = torch.cat((output_binary, output_categorical), 1)
        # X = X[:, binary_columns + categorical_columns]

        return output_combined


# ####################################### beginning of main script #######################################
def rescale_dims(data):
  # assume min=0
  max_vals = np.max(data, axis=0)
  print('max vals:', max_vals)
  data = data / max_vals
  print('new max', np.max(data))
  return data, max_vals


def revert_scaling(data, base_scale):
  return data * base_scale


def main():
  args, device = parse_arguments()
  seed = np.random.randint(0, 1000)
  print('seed: ', seed)

  # print('Hermite polynomial order: ', args.order_hermite)

  random.seed(seed)
  ############################### data loading ##################################
  # print("adult_cat dataset")  # this is heterogenous

  if args.dataset_name=='adult':
      data = np.load(f"../data/real/sdgym_{args.dataset}_adult.npy")
  else: # for census data, we take the first
      data = np.load(f"../data/real/sdgym_{args.dataset}_census.npy")
      # n_subsampled_datapoints = 20000 # to do a quick test, later remove this
      # data = data[np.random.permutation(data.shape[0])][:n_subsampled_datapoints]
      # np.save('census_small.npy', data)


  if args.kernel == 'linear':
    data, unbin_mapping_info = binarize_data(data)
    # print('bin data shape', data.shape)
  else:
    unbin_mapping_info = None

  if args.norm_dims == 1:
    data, base_scale = rescale_dims(data)
  else:
    base_scale = None


  ###########################################################################
  # PREPARING GENERATOR

  X = data # without labels separated
  n_classes = 1

  n_samples, input_dim = X.shape

  ######################################
  # MODEL

  # model specifics
  batch_size = np.int(np.round(args.batch_rate * n_samples))
  # print("minibatch: ", batch_size)

  input_size = 5
  hidden_size_1 = 400 * input_dim
  hidden_size_2 = 100 * input_dim

  # hidden_size_3 = 10 * input_dim

  output_size = input_dim
  out_fun = 'relu' if args.kernel == 'gaussian' else 'sigmoid'

  model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                            hidden_size_2=hidden_size_2,
                                            output_size=output_size,
                                            out_fun=out_fun).to(device)

  ####################### estimating length scale for each dimensoin ##################
  sigma2 = np.median(X, 0)
  sigma2[sigma2==0] = 0.9
  if args.dataset_name == 'census':
      hp = args.hyperparam
      if hp==0:
          sigma2 = sigma2
      else:
          sigma2 = hp*np.sqrt(sigma2)
  else:
      if args.dataset=='simple':
          sigma2 = 0.2*np.sqrt(sigma2)
      else:
          sigma2 = sigma2

  rho = find_rho_tab(sigma2)
  order = args.order_hermite

  ########## data mean embedding ##########

  if args.is_private:
      # print("private")
      delta = 1e-5
      k = 1  # because we add noise to the weights and means separately.
      privacy_param = privacy_calibrator.gaussian_mech(args.epsilon, delta, k=k)
      # print(f'eps,delta = ({args.epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

  """ compute the means """
  # print('computing mean embedding of data:')
  data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)

  chunk_size = 250
  emb_sum = 0
  for idx in range(n_samples // chunk_size + 1):
      data_chunk = data[idx * chunk_size:(idx + 1) * chunk_size].astype(np.float32)
      chunk_emb = ME_with_HP_tab(torch.Tensor(data_chunk).to(device), order, rho, device, n_samples)
      emb_sum += chunk_emb

  data_embedding[:,0] = emb_sum


  if args.is_private:
      std = (2 * privacy_param['sigma'] / n_samples)
      noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

      # print('before perturbation, mean and variance of data mean embedding are %f and %f ' % (
      # torch.mean(data_embedding), torch.std(data_embedding)))
      data_embedding = data_embedding + noise
      # print('after perturbation, mean and variance of data mean embedding are %f and %f ' % (
      # torch.mean(data_embedding), torch.std(data_embedding)))

  """ Training """
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  num_iter = np.int(n_samples / batch_size)

  # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
  for epoch in range(args.epochs):  # loop over the dataset multiple times
      model.train()

      for i in range(num_iter):

          feature_input = torch.randn((batch_size, input_size)).to(device)
          input_to_model = feature_input

          """ (2) produce data """
          outputs = model(input_to_model)

          """ (3) compute synthetic data's mean embedding """
          syn_data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)
          for idx in range(n_classes):
              idx_syn_data = outputs
              phi_syn_data = ME_with_HP_tab(idx_syn_data, order, rho, device, batch_size)
              syn_data_embedding[:, idx] = phi_syn_data  # this includes 1/n factor inside

          loss = torch.sum((data_embedding - syn_data_embedding)**2)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
      # print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))

      if (epoch>0) & (epoch % 50 == 0): # every 20th epoch we evaluate the quality of the data
          
          print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
          
          """ draw final data samples """
          if args.dataset_name == 'census':
              chunk_size = 2000
              generated_data = np.zeros(((n_samples//chunk_size)*chunk_size, input_dim))

              # n_samples = 199523
              # generated_data.shape = 198000, 41
              for idx in range(n_samples // chunk_size):
                  # print('%d of generating samples out of %d' %(idx, n_samples // chunk_size))
                  feature_input = torch.randn((chunk_size, input_size)).to(device)
                  input_to_model = feature_input
                  outputs = model(input_to_model)
                  samp_input_features = outputs
                  generated_data[idx * chunk_size:(idx + 1) * chunk_size,:] = samp_input_features.cpu().detach().numpy()

              generated_input_features_final = generated_data

          else:
              feature_input = torch.randn((n_samples, input_size)).to(device)
              input_to_model = feature_input
              outputs = model(input_to_model)

              samp_input_features = outputs

              generated_input_features_final = samp_input_features.cpu().detach().numpy()

          ##################################################################################################################

          if args.norm_dims == 1:
            generated_input_features_final = revert_scaling(generated_input_features_final, base_scale)

          # run marginals test
          if args.dataset_name == 'census':
              save_file = f"census_{args.dataset}_gen_eps_{args.epsilon}_{args.kernel}_kernel_" \
                          f"batch_rate_{args.batch_rate}_hp_{args.order_hermite}.npy"
              if args.save_data:
                  # save generated samples
                  path_gen_data = f"../data/generated/census"
                  os.makedirs(path_gen_data, exist_ok=True)
                  data_save_path = os.path.join(path_gen_data, save_file)
                  np.save(data_save_path, generated_input_features_final)
                  # print(f"Generated data saved to {path_gen_data}")
              else:
                  data_save_path = save_file

              real_data = f'../data/real/sdgym_{args.dataset}_census.npy'
              # real_data = f'census_small.npy'
              alpha = 3
              # then subsample datapoints, because this dataset is huge
              gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                           real_data_path=real_data,
                                           discretize=True,
                                           alpha=alpha,
                                           verbose=False,
                                           unbinarize=args.kernel == 'linear',
                                           unbin_mapping_info=unbin_mapping_info,
                                           # n_subsampled_datapoints=10000,
                                           gen_data_direct=generated_input_features_final)

              alpha = 4
              gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                                 real_data_path=real_data,
                                                 discretize=True,
                                                 alpha=alpha,
                                                 verbose=False,
                                                 unbinarize=args.kernel == 'linear',
                                                 unbin_mapping_info=unbin_mapping_info,
                                                 # n_subsampled_datapoints=1000,
                                                 gen_data_direct=generated_input_features_final)
          else:
              save_file = f"adult_{args.dataset}_gen_eps_{args.epsilon}_{args.kernel}_kernel_" \
                          f"batch_rate_{args.batch_rate}_hp_{args.order_hermite}.npy"
              if args.save_data:
                  # save generated samples
                  path_gen_data = f"../data/generated/adult"
                  os.makedirs(path_gen_data, exist_ok=True)
                  data_save_path = os.path.join(path_gen_data, save_file)
                  np.save(data_save_path, generated_input_features_final)
                  print(f"Generated data saved to {path_gen_data}")
              else:
                  data_save_path = save_file

              real_data = f'../data/real/sdgym_{args.dataset}_adult.npy'
              alpha = 3
              # real_data = 'numpy_data/sdgym_bounded_adult.npy'
              gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                               real_data_path=real_data,
                                               discretize=True,
                                               alpha=alpha,
                                               verbose=True,
                                               unbinarize=args.kernel == 'linear',
                                               unbin_mapping_info=unbin_mapping_info,
                                               gen_data_direct=generated_input_features_final)

              alpha = 4
              # real_data = 'numpy_data/sdgym_bounded_adult.npy'
              gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                               real_data_path=real_data,
                                               discretize=True,
                                               alpha=alpha,
                                               verbose=True,
                                               unbinarize=args.kernel == 'linear',
                                               unbin_mapping_info=unbin_mapping_info,
                                               gen_data_direct=generated_input_features_final)

###################################################################################################


def parse_arguments():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  args = argparse.ArgumentParser()

  args.add_argument('--order-hermite', type=int, default=100, help='')
  args.add_argument('--epochs', type=int, default=400)
  # args.add_argument("--batch-rate", type=float, default=0.1) # for adult data

  args.add_argument("--batch-rate", type=float, default=0.1)  # for census data
  args.add_argument("--lr", type=float, default=0.0001)

  args.add_argument("--hyperparam", type=float, default=1.0)
  
  args.add_argument('--is-private', default=True, help='produces a DP mean embedding of data')
  args.add_argument("--epsilon", type=float, default=1.0)
  args.add_argument("--dataset", type=str, default='simple', choices=['bounded', 'simple'])
  args.add_argument("--dataset_name", type=str, default='census', choices=['census', 'adult'])
  args.add_argument('--kernel', type=str, default='gaussian', choices=['gaussian', 'linear'])
  # args.add_argument("--data_type", default='generated')  # both, real, generated
  args.add_argument("--save-data", type=int, default=1, help='save data if 1')

  
  #args.add_argument("--d_hid", type=int, default=200)
  args.add_argument("--norm-dims", type=int, default=0, help='normalize dimensions to same range if 1')

  arguments = args.parse_args()
  print("arg", arguments)
  return arguments, device


if __name__ == '__main__':
  main()
    
    
