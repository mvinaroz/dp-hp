### this is for training a single generator for all labels ###
""" with the analysis of """
### weights = weights + N(0, sigma**2*(sqrt(2)/N)**2)
### columns of mean embedding = raw + N(0, sigma**2*(2/N)**2)

import numpy as np
# import matplotlib.pyplot as plt
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
from all_aux_files_tab_data import  find_rho_tab, meddistance, heuristic_for_length_scale, ME_with_HP_tab
from all_aux_files import ME_with_HP

import warnings
warnings.filterwarnings('ignore')
import os
from code_tab.marginals_eval import gen_data_alpha_way_marginal_eval
from code_tab.binarize_adult import binarize_data

############################### kernels to use ###############################

""" we use a weighted polynomial kernel for labels """

def Feature_labels(labels, weights, device):

  weights = torch.Tensor(weights)
  weights = weights.to(device)

  labels = labels.to(device)

  weighted_labels_feature = labels/weights

  return weighted_labels_feature


# ############################## end of kernels ###############################

# ############################## generative models to use ###############################
class Generative_Model_homogeneous_data(nn.Module):
  def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, out_fun):
    super(Generative_Model_homogeneous_data, self).__init__()

    self.input_size = input_size
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    self.output_size = output_size
    assert out_fun in ('lin', 'sigmoid', 'relu')

    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
    self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
    self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
    self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
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
    output = self.relu(self.bn2(output))
    output = self.fc3(output)
    output = self.out_fun(output)
    return output


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

  print('Hermite polynomial order: ', args.order_hermite)

  random.seed(seed)
  ############################### data loading ##################################
  print("adult_cat dataset")  # this is heterogenous

  data = np.load(f"../data/real/sdgym_{args.dataset}_adult.npy")
  print('data shape', data.shape)
  
  true_labels =np.ones(data.shape[0]) #We set al labels to the same class (as we don't have it).
  true_labels = np.expand_dims(true_labels, 1)
  
  
  if args.kernel == 'linear':
    data, unbin_mapping_info = binarize_data(data)
    print('bin data shape', data.shape)
  else:
    unbin_mapping_info = None

  if args.norm_dims == 1:
    data, base_scale = rescale_dims(data)
  else:
    base_scale = None


  ###########################################################################
  # PREPARING GENERATOR
  n_samples, input_dim = data.shape

  ######################################
  # MODEL

  # model specifics
  batch_size = np.int(np.round(args.batch_rate * n_samples))
  print("minibatch: ", batch_size)
  input_size = 10 + 1

  hidden_size_1 = 4 * input_dim
  hidden_size_2 = 2 * input_dim

  output_size = input_dim
  out_fun = 'relu' if args.kernel == 'gaussian' else 'sigmoid'

  model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,
                                            hidden_size_2=hidden_size_2,
                                            output_size=output_size,
                                            out_fun=out_fun).to(device)

  # define details for training
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  #training_loss_per_iteration = np.zeros(args.iterations)

  ##########################################################################
  heterogeneous_datasets=[]
  num_numerical_inputs=[]
  
  """ computing mean embedding of subsampled true data """
  if args.kernel == 'gaussian':
    """ set the scale length """
    if args.heuristic_sigma:
        print('we use the median heuristic for length scale')
        sigma = heuristic_for_length_scale('adult', data, num_numerical_inputs, input_dim, heterogeneous_datasets)
        if args.separate_kernel_length:
            print('we use a separate length scale on each coordinate of the data')
            sigma2 = sigma**2
        else:
            sigma2 = np.median(sigma**2)
    else:
        sigma2 = args.kernel_length
    # print('sigma2 is', sigma2)
    
  else:
    raise ValueError


  rho = find_rho_tab(sigma2)
  order = args.order_hermite

  ########## data mean embedding ##########
  """ compute the weights """
  print('computing mean embedding of data: (1) compute the weights')
  
  unnormalized_weights = np.sum(true_labels, 0)
  weights = unnormalized_weights / np.sum(unnormalized_weights) # weights = m_c / n
  print('\n weights with no privatization are', weights, '\n')

  
  print("private")
  delta = 1e-5
  k = 2 # because we add noise to the weights and means separately.
  privacy_param = privacy_calibrator.gaussian_mech(args.epsilon, delta, k=k)
  sensitivity_for_weights = np.sqrt(2) / n_samples  # double check if this is sqrt(2) or 2
  noise_std_for_weights = privacy_param['sigma'] * sensitivity_for_weights
  weights = weights + np.random.randn(weights.shape[0]) * noise_std_for_weights
  weights[weights < 0] = 1e-3  # post-processing so that we don't have negative weights.
  weights = weights/sum(weights) # post-processing so the sum of weights equals 1.
  
  n_classes=1

  """ compute the means """
  print('computing mean embedding of data: (2) compute the mean')
  data_embedding = torch.zeros(input_dim*(order+1), n_classes, device=device)
  for idx in range(n_classes):
     #print(idx,'th-class')
     #idx_data = X_train[y_train.squeeze()==idx,:]
     if args.separate_kernel_length:
         phi_data = ME_with_HP_tab(torch.Tensor(data).to(device), order, rho, device, n_samples)
     else:
         phi_data = ME_with_HP(torch.Tensor(data).to(device), order, rho, device, n_samples)
     data_embedding[:,idx] = phi_data # this includes 1/n factor inside
  print('done with computing mean embedding of data')
  
  if args.is_private:
      # print('we add noise to the data mean embedding as the private flag is true')
      # std = (2 * privacy_param['sigma'] * np.sqrt(input_dim) / n)
      std = (2 * privacy_param['sigma'] / n_samples)
      noise = torch.randn(data_embedding.shape[0], data_embedding.shape[1], device=device) * std

      print('before perturbation, mean and variance of data mean embedding are %f and %f ' %(torch.mean(data_embedding), torch.std(data_embedding)))
      data_embedding = data_embedding + noise
      print('after perturbation, mean and variance of data mean embedding are %f and %f ' % (torch.mean(data_embedding), torch.std(data_embedding)))

  # the final mean embedding of data is,
  data_embedding = data_embedding / torch.Tensor(weights).to(device) # this means, 1/n * n/m_c, so 1/m_c
   
  """ Training """
  #optimizer = torch.optim.Adam(list(model.parameters()), lr=ar.lr)

  num_iter = np.int(n_samples / batch_size)

  for epoch in range(args.epochs):  # loop over the dataset multiple times
      model.train()

      for i in range(num_iter):

          #if data_name in homogeneous_datasets:  # In our case the features aren't the original ones (D all numerical features are discretized and D* features reduced to a max 15) )

          label_input = torch.multinomial(torch.Tensor([weights]), batch_size, replacement=True).type(torch.FloatTensor)
          label_input = label_input.transpose_(0, 1)
          label_input = label_input.squeeze()
          label_input = label_input.to(device)

          feature_input = torch.randn((batch_size, input_size - 1)).to(device)
          input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
 
          outputs = model(input_to_model)
          
          weights_syn = torch.zeros(n_classes) # weights = m_c / n
          syn_data_embedding = torch.zeros(input_dim * (order + 1), n_classes, device=device)
          for idx in range(n_classes):
              weights_syn[idx] = torch.sum(label_input == idx)
              idx_syn_data = outputs[label_input == idx]
              if args.separate_kernel_length:
                  phi_syn_data = ME_with_HP_tab(idx_syn_data, order, rho, device, batch_size)
              else:
                  phi_syn_data = ME_with_HP(idx_syn_data, order, rho, device, batch_size)
              syn_data_embedding[:, idx] = phi_syn_data  # this includes 1/n factor inside

          weights_syn = weights_syn / torch.sum(weights_syn)
          syn_data_embedding = syn_data_embedding / torch.Tensor(weights_syn).to(device)

          loss = torch.sum((data_embedding - syn_data_embedding)**2)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
      print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
      
  """ draw final data samples """
  
  #label_input = (1 * (torch.rand((n_samples)) < weights[1])).type(torch.FloatTensor)
  # label_input = torch.multinomial(1 / n_classes * torch.ones(n_classes), n, replacement=True).type(torch.FloatTensor)
  #label_input = label_input.to(device)
  
  feature_input = torch.randn((n_samples, input_size)).to(device)
  #feature_input = torch.randn((n_samples, input_size - 1)).to(device)
  print(feature_input.shape)
  #input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
  #outputs = model(input_to_model)
  outputs = model(feature_input)

  samp_input_features = outputs

  #label_input_t = torch.zeros((n_samples, n_classes))
  #idx_1 = (label_input == 1.).nonzero()[:, 0]
  #idx_0 = (label_input == 0.).nonzero()[:, 0]
  #label_input_t[idx_1, 1] = 1.
  #label_input_t[idx_0, 0] = 1.

  #samp_labels = label_input_t

  generated_input_features_final = samp_input_features.cpu().detach().numpy()
  print("The generated samples: ", generated_input_features_final)
  print("The generated samples shape: ", generated_input_features_final.shape)
  #generated_labels_final = samp_labels.cpu().detach().numpy()
  #generated_labels = np.argmax(generated_labels_final, axis=1)
  
  ##################################################################################################################

  if args.norm_dims == 1:
    generated_input_features_final = revert_scaling(generated_input_features_final, base_scale)

  save_file = f"adult_{args.dataset}_gen_eps_{args.epsilon}_{args.kernel}_kernel_" \
              f"batch_rate_{args.batch_rate}_hp_{args.order_hermite}.npy"
  if args.save_data:
    # save generated samples
    path_gen_data = f"../data/generated/rebuttal_exp"
    os.makedirs(path_gen_data, exist_ok=True)
    data_save_path = os.path.join(path_gen_data, save_file)
    np.save(data_save_path, generated_input_features_final)
    print(f"Generated data saved to {path_gen_data}")
  else:
    data_save_path = save_file

  # run marginals test
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

  #args.add_argument("--n_features", type=int, default=2000)
  args.add_argument('--order-hermite', type=int, default=100, help='')
  args.add_argument('--heuristic-sigma', action='store_true', default=True)
  args.add_argument("--separate-kernel-length", action='store_true', default=True)
  args.add_argument("--kernel-length", type=float, default=0.1)  
  #args.add_argument("--iterations", type=int, default=1000) 
  #args.add_argument("--batch_size", type=int, default=1000)
  args.add_argument('--epochs', type=int, default=1000)
  args.add_argument("--batch-rate", type=float, default=0.1)
  args.add_argument("--lr", type=float, default=0.01)
  
  args.add_argument('--is-private', default=True, help='produces a DP mean embedding of data')
  args.add_argument("--epsilon", type=float, default=1.0)
  args.add_argument("--dataset", type=str, default='simple', choices=['bounded', 'simple'])
  args.add_argument('--kernel', type=str, default='gaussian', choices=['gaussian', 'linear'])
  # args.add_argument("--data_type", default='generated')  # both, real, generated
  args.add_argument("--save-data", type=int, default=0, help='save data if 1')

  
  #args.add_argument("--d_hid", type=int, default=200)
  args.add_argument("--norm-dims", type=int, default=0, help='normalize dimensions to same range if 1')

  arguments = args.parse_args()
  print("arg", arguments)
  return arguments, device


if __name__ == '__main__':
  main()
    
    
