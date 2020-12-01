### this is for training a single generator for all labels ###
""" with the analysis of """
### weights = weights + N(0, sigma**2*(sqrt(2)/N)**2)
### columns of mean embedding = raw + N(0, sigma**2*(2/N)**2)

import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import argparse
import seaborn as sns
sns.set()
# %matplotlib inline
from autodp import privacy_calibrator

import warnings
warnings.filterwarnings('ignore')
import os
from marginals_eval import gen_data_alpha_way_marginal_eval
from binarize_adult import binarize_data

############################### kernels to use ###############################
""" we use the random fourier feature representation for Gaussian kernel """

def RFF_Gauss(n_features, X, W, device):
  """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

  W = torch.Tensor(W).to(device)
  X = X.to(device)
  XWT = torch.mm(X, torch.t(W)).to(device)
  Z1 = torch.cos(XWT)
  Z2 = torch.sin(XWT)
  Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
  return Z


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


def main():
  args, device = parse_arguments()
  seed = np.random.randint(0, 1000)
  print('seed: ', seed)

  print('number of features: ', args.n_features)

  random.seed(seed)
  ############################### data loading ##################################
  print("adult_cat dataset")  # this is heterogenous

  data = np.load(f"../data/real/sdgym_{args.dataset}_adult.npy")
  print('data shape', data.shape)

  if args.kernel == 'linear':
    data, unbin_mapping_info = binarize_data(data)
    print('bin data shape', data.shape)
  else:
    unbin_mapping_info = None


  ###########################################################################
  # PREPARING GENERATOR
  n_samples, input_dim = data.shape

  ######################################
  # MODEL

  # model specifics
  # mini_batch_size = np.int(np.round(batch_size * n))
  print("minibatch: ", args.batch_size)
  input_size = 10
  if args.d_hid is not None:
    hidden_size_1 = args.d_hid
    hidden_size_2 = args.d_hid
  else:
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
  training_loss_per_iteration = np.zeros(args.iterations)

  ##########################################################################

  """ computing mean embedding of subsampled true data """
  if args.kernel == 'gaussian':
    med = util.meddistance(data[:300])
    W_freq = np.random.randn(args.n_features // 2, input_dim) / med

    # aggregate embedding in chunks
    chunk_size = 250
    emb_sum = 0
    for idx in range(n_samples // chunk_size + 1):
      data_chunk = data[idx * chunk_size:(idx + 1) * chunk_size].astype(np.float32)
      chunk_emb = RFF_Gauss(args.n_features, torch.tensor(data_chunk), W_freq, device)
      emb_sum += torch.sum(chunk_emb, 0)

    mean_emb1 = emb_sum / n_samples
    # outer_emb1 = RFF_Gauss(args.n_features, torch.tensor(data), W_freq, device)

  elif args.kernel == 'linear':
    outer_emb1 = (torch.tensor(data) / np.sqrt(data.shape[1])).to(device)
    mean_emb1 = torch.mean(outer_emb1, 0)
    W_freq = None
  else:
    raise ValueError

  ####################################################
  # Privatising quantities if necessary

  """ privatizing weights """
  delta = 1e-5
  privacy_param = privacy_calibrator.gaussian_mech(args.epsilon, delta, k=1)
  print(f'eps,delta = ({args.epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

  sensitivity = 2 / n_samples
  noise_std_for_privacy = privacy_param['sigma'] * sensitivity

  noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
  noise = noise.to(device)

  mean_emb1 = mean_emb1 + noise

  ##################################################################################################################
  # TRAINING THE GENERATOR

  print('Starting Training')

  for iteration in range(args.iterations):  # loop over the dataset multiple times

    running_loss = 0.0

    """ computing mean embedding of generated data """
    optimizer.zero_grad()
    feature_input = torch.randn((args.batch_size, input_size)).to(device)
    outputs = model(feature_input)

    """ computing mean embedding of generated samples """
    if args.kernel == 'gaussian':
      emb2_input_features = RFF_Gauss(args.n_features, outputs, W_freq, device)
      pass
    elif args.kernel == 'linear':
      emb2_input_features = outputs / (torch.tensor(np.sqrt(data.shape[1], dtype=np.float32))).to(device)  # 8
    else:
      raise ValueError
    mean_emb2 = torch.mean(emb2_input_features, 0)
    loss = torch.norm(mean_emb1 - mean_emb2, p=2) ** 2

    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    if iteration % 100 == 0:
      print('iteration # and running loss are ', [iteration, running_loss])
      training_loss_per_iteration[iteration] = running_loss

  """ now generate samples from the trained network """

  feature_input = torch.randn((n_samples, input_size)).to(device)
  outputs = model(feature_input)
  gen_data = outputs.detach().cpu().numpy()
  save_file = f"adult_{args.dataset}_gen_eps_{args.epsilon}_{args.kernel}_kernel_" \
              f"it_{args.iterations}_features_{args.n_features}.npy"
  if args.save_data:
    # save generated samples
    path_gen_data = f"../data/generated/rebuttal_exp"
    os.makedirs(path_gen_data, exist_ok=True)
    data_save_path = os.path.join(path_gen_data, save_file)
    np.save(data_save_path, gen_data)
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
                                   gen_data_direct=gen_data)

  alpha = 4
  # real_data = 'numpy_data/sdgym_bounded_adult.npy'
  gen_data_alpha_way_marginal_eval(gen_data_path=data_save_path,
                                   real_data_path=real_data,
                                   discretize=True,
                                   alpha=alpha,
                                   verbose=True,
                                   unbinarize=args.kernel == 'linear',
                                   unbin_mapping_info=unbin_mapping_info,
                                   gen_data_direct=gen_data)

###################################################################################################


def parse_arguments():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  args = argparse.ArgumentParser()
  # args.add_argument("--n_features", type=int, default=2000)
  args.add_argument("--n_features", type=int, default=10000)
  args.add_argument("--iterations", type=int, default=8000)
  # args.add_argument("--batch_size", type=float, default=128)
  args.add_argument("--batch_size", type=float, default=1000)
  args.add_argument("--lr", type=float, default=1e-2)

  args.add_argument("--epsilon", type= float, default=1.0)
  args.add_argument("--dataset", type=str, default='simple', choices=['bounded', 'simple'])
  args.add_argument('--kernel', type=str, default='gaussian', choices=['gaussian', 'linear'])
  # args.add_argument("--data_type", default='generated')  # both, real, generated
  args.add_argument("--save_data", type=int, default=0, help='save data if 1')

  args.add_argument("--d_hid", type=int, default=500)

  arguments = args.parse_args()
  print("arg", arguments)
  return arguments, device


if __name__ == '__main__':
  main()
